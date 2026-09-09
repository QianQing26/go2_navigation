"""Train the independent SEA-Nav MotionEstimator."""

import argparse
import csv
import datetime
import json
import os
import shutil
import sys

import torch
import yaml
from torch.utils.data import DataLoader

from data import MotionDataset, NormalizationStats, load_dataset_metadata
from losses import motion_loss
from models import MotionEstimator
from utils.checkpoint import load_checkpoint, save_checkpoint
from utils.metrics import scalar_stats
from utils.seed import seed_everything
from utils.visualization import plot_loss_curve


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


def _path(value):
    value = os.path.expanduser(value)
    return value if os.path.isabs(value) else os.path.join(ROOT, value)


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=os.path.join(os.path.dirname(__file__), 'configs', 'default.yaml'))
    parser.add_argument('--device', default=None)
    parser.add_argument('--resume', default=None)
    parser.add_argument('--run_name', default=None)
    parser.add_argument('--artifacts_dir', default=os.path.join(os.path.dirname(__file__), 'artifacts'))
    parser.add_argument('--max_train_batches', type=int, default=None)
    parser.add_argument('--max_val_batches', type=int, default=None)
    parser.add_argument('--num_workers', type=int, default=None)
    parser.add_argument('--epochs', type=int, default=None)
    return parser.parse_args()


def _load_config(path):
    with open(path) as file:
        config = yaml.safe_load(file)
    return config


def _target_cfg(config, manifest):
    target = dict(config['target'])
    if target.get('ray_max') is None:
        dynamic_range = manifest.get('dynamic_ray_range')
        raw_range = manifest.get('ray_range_semantics', {}).get('raw_dataset_range_checked')
        target['ray_max'] = float((dynamic_range or raw_range)[1])
    manifest_horizon = manifest.get('gt_horizon')
    if manifest_horizon is not None and abs(float(target['horizon']) - float(manifest_horizon)) > 1.0e-6:
        raise ValueError('Config horizon does not match manifest gt_horizon')
    return target


def _make_model(config, manifest, normalization):
    model_cfg = config['model']
    angles = manifest.get('ray_angles_deg')
    return MotionEstimator(
        history_length=int(model_cfg['history_length']),
        num_rays=int(model_cfg['num_rays']),
        hidden_dim=int(model_cfg['hidden_dim']),
        temporal_pool=model_cfg.get('temporal_pool', 'attention'),
        use_ego_motion=bool(model_cfg.get('use_ego_motion', True)),
        angular_dilations=tuple(model_cfg.get('angular_dilations', [1, 2, 4])),
        ray_angles=angles,
        ray_max=normalization.ray_max,
        hit_epsilon=normalization.hit_epsilon,
        d_safe=normalization.d_safe,
        kappa=normalization.kappa,
    )


def _to_device(batch, device):
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def _run_epoch(model, loader, loss_cfg, device, optimizer=None, scaler=None, amp=False, max_batches=None):
    training = optimizer is not None
    model.train(training)
    totals = {}
    count = 0
    drift_pred, drift_gt = [], []
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        batch = _to_device(batch, device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        autocast_enabled = bool(amp and device.type == 'cuda')
        with torch.cuda.amp.autocast(enabled=autocast_enabled):
            prediction = model(
                batch['rays_hist'], batch['ego_hist'], batch['ray_hit_hist'],
                batch['current_fused_rays'],
            )
            losses = motion_loss(prediction, batch, loss_cfg)
        if training:
            if scaler is not None and autocast_enabled:
                scaler.scale(losses['total']).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), loss_cfg['_grad_clip'])
                scaler.step(optimizer)
                scaler.update()
            else:
                losses['total'].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), loss_cfg['_grad_clip'])
                optimizer.step()
        batch_count = int(batch['closing_gt'].shape[0])
        count += batch_count
        for key in ('total', 'closing', 'drift', 'closing_under', 'drift_optimistic'):
            totals[key] = totals.get(key, 0.0) + float(losses[key].detach()) * batch_count
        if not training:
            drift_pred.append(prediction['drift'].detach().float().cpu())
            drift_gt.append(batch['drift_target'].detach().float().cpu())
    if count == 0:
        raise RuntimeError('No batches processed')
    result = {key: value / count for key, value in totals.items()}
    if drift_pred:
        pred = torch.cat(drift_pred)
        target = torch.cat(drift_gt)
        result['drift_mae_normalized'] = float((pred - target).abs().mean())
    return result


def _write_csv(path, rows):
    if not rows:
        return
    fields = sorted({key for row in rows for key in row})
    with open(path, 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main(args):
    config_path = _path(args.config)
    config = _load_config(config_path)
    dataset_dir, manifest, splits, _ = load_dataset_metadata(_path(config['dataset']['path']))
    target_cfg = _target_cfg(config, manifest)
    if int(config['model']['num_rays']) != len(manifest['ray_angles_deg']):
        raise ValueError('model.num_rays does not match manifest ray count')
    seed_everything(config['train']['seed'])
    device = torch.device(args.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    run_name = args.run_name or config['train'].get('run_name') or datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = _path(os.path.join(args.artifacts_dir, run_name))
    os.makedirs(run_dir, exist_ok=True)
    shutil.copyfile(config_path, os.path.join(run_dir, 'config.yaml'))

    raw_train = MotionDataset(
        dataset_dir, 'train', normalization=None, target_cfg=target_cfg,
        history_length=config['model']['history_length'], ray_max=target_cfg['ray_max'],
    )
    normalization_path = os.path.join(run_dir, 'normalization.json')
    if args.resume and os.path.isfile(normalization_path):
        normalization = NormalizationStats.load(normalization_path)
    else:
        normalization = NormalizationStats.from_dataset(
            raw_train, target_cfg, target_cfg['ray_max']
        )
        normalization.save(normalization_path)
    datasets = {
        split: MotionDataset(
            dataset_dir, split, normalization=normalization,
            target_cfg=target_cfg, history_length=config['model']['history_length'],
            ray_max=target_cfg['ray_max'],
        ) for split in ('train', 'val', 'test')
    }
    loader_kwargs = {
        'batch_size': config['train']['batch_size'],
        'num_workers': int(args.num_workers if args.num_workers is not None else config['dataset']['num_workers']),
        'pin_memory': bool(config['dataset'].get('pin_memory', True)),
    }
    train_loader = DataLoader(datasets['train'], shuffle=True, **loader_kwargs)
    val_loader = DataLoader(datasets['val'], shuffle=False, **loader_kwargs)
    model = _make_model(config, manifest, normalization).to(device)
    print('[motion_estimator] device={} parameters={}'.format(
        device, sum(parameter.numel() for parameter in model.parameters())
    ), flush=True)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(config['train']['lr']),
        weight_decay=float(config['train']['weight_decay']),
    )
    epochs = int(args.epochs or config['train']['epochs'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    scaler = torch.cuda.amp.GradScaler(enabled=bool(config['train'].get('amp', True) and device.type == 'cuda'))
    start_epoch = 0
    best_metric = float('inf')
    if args.resume:
        checkpoint = load_checkpoint(args.resume, model, optimizer, scheduler, map_location=device)
        start_epoch = int(checkpoint.get('epoch', -1)) + 1
        best_metric = float(checkpoint.get('best_metric', best_metric))
    loss_cfg = dict(config['loss'])
    # Loss thresholds are specified in physical m/s in YAML, while the
    # regression targets are normalized by train-only P95 scales.
    loss_cfg['zero_threshold'] /= normalization.closing_scale
    loss_cfg['drift_danger_threshold'] /= normalization.drift_scale
    loss_cfg['_grad_clip'] = float(config['train']['grad_clip'])
    rows = []
    patience = 0
    for epoch in range(start_epoch, epochs):
        train_result = _run_epoch(
            model, train_loader, loss_cfg, device, optimizer, scaler,
            amp=bool(config['train'].get('amp', True)),
            max_batches=args.max_train_batches,
        )
        with torch.no_grad():
            val_result = _run_epoch(
                model, val_loader, loss_cfg, device,
                amp=bool(config['train'].get('amp', True)),
                max_batches=args.max_val_batches,
            )
        scheduler.step()
        # Drift MAE in normalized target units is stable for model selection.
        val_metric = val_result['drift_mae_normalized']
        row = {
            'epoch': epoch,
            'lr': optimizer.param_groups[0]['lr'],
            'train_loss': train_result['total'],
            'val_loss': val_result['total'],
            'val_drift_mae': val_metric,
            'train_closing': train_result['closing'],
            'train_drift': train_result['drift'],
            'val_closing': val_result['closing'],
            'val_drift': val_result['drift'],
        }
        rows.append(row)
        _write_csv(os.path.join(run_dir, 'train_log.csv'), rows)
        save_checkpoint(
            os.path.join(run_dir, 'last.pt'), model, optimizer, scheduler,
            epoch, best_metric, normalization, config,
        )
        if val_metric < best_metric:
            best_metric = val_metric
            patience = 0
            save_checkpoint(
                os.path.join(run_dir, 'best.pt'), model, optimizer, scheduler,
                epoch, best_metric, normalization, config,
            )
        else:
            patience += 1
        print('[motion_estimator] epoch={} train={:.5f} val={:.5f} drift_mae={:.5f}'.format(
            epoch, train_result['total'], val_result['total'], val_metric,
        ), flush=True)
        if patience >= int(config['train']['early_stopping_patience']):
            print('[motion_estimator] early stopping', flush=True)
            break
    with open(os.path.join(run_dir, 'metrics.json'), 'w') as file:
        json.dump({
            'best_val_drift_mae_normalized': best_metric,
            'epochs_completed': len(rows),
            'model_parameters': sum(parameter.numel() for parameter in model.parameters()),
            'dataset_dir': dataset_dir,
            'split_sample_counts': splits.get('sample_counts', {}),
        }, file, indent=2)
    os.makedirs(os.path.join(run_dir, 'figures'), exist_ok=True)
    plot_loss_curve(rows, os.path.join(run_dir, 'figures', 'loss_curve.png'))
    print('[motion_estimator] run_dir={}'.format(run_dir), flush=True)
    return run_dir


if __name__ == '__main__':
    # Allow direct execution: ``python motion_estimator/train.py``.
    sys.path.insert(0, os.path.dirname(__file__))
    main(_parse_args())
