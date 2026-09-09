import os

import torch


def save_checkpoint(path, model, optimizer, scheduler, epoch, best_metric, normalization, config):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'epoch': int(epoch),
        'best_metric': float(best_metric),
        'normalization': normalization.to_dict(),
        'model_config': model.config_dict(),
        'config': config,
    }, path)


def load_checkpoint(path, model, optimizer=None, scheduler=None, map_location='cpu'):
    checkpoint = torch.load(path, map_location=map_location)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None and checkpoint.get('optimizer_state_dict'):
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler is not None and checkpoint.get('scheduler_state_dict'):
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    return checkpoint
