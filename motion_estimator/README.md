# SEA-Nav MotionEstimator

This directory contains the independent offline MotionEstimator pipeline. It
does not import Isaac Gym and is not connected to PPO, the navigation policy,
or the CBF runtime.

The estimator consumes a clean synchronized history:

```text
R_t: [H, 41] physical rays, H=10
M_t: [H, 3] = [v_x^B, v_y^B, omega_z^B]
```

It predicts a per-ray finite-horizon closing field `c_delta [41]` and a
scalar composite safety-field drift `d_h_delta`.

## Targets

The loader computes the scalar target when it is absent from the dataset:

```text
h_i = rho_i - d_safe
h_LSE = -logsumexp(-kappa * h_i) / kappa
d_h_delta = (h_LSE_future_cf - h_LSE_current) / Delta
```

For the current SEA-Nav CBF implementation, the verified values are
`d_safe = 0.20 m`, `kappa = 10.0`, and `Delta = 0.10 s`. The final dataset
manifest confirms the horizon and ray range (`rho_max = 3 m`); the safety
constants are recorded in `configs/default.yaml` because they are not stored
in the dataset manifest.

The local closing loss uses `source_switch_mask == False` as its regression
mask. Near-zero targets receive reduced data weight, and approaching targets
receive a conservative underestimation penalty. The scalar drift head is
trained on all samples, with an additional penalty against optimistic
predictions when the true drift is negative.

## Dataset and layout

The default dataset is:

```text
datasets/motion_dataset_final
```

The loader reads one shard at a time and uses the existing episode-level
`splits.json`; it never performs a random frame-level split.

```text
motion_estimator/
├── configs/default.yaml
├── data/                 # shard loader and train-only normalization
├── models/               # CNN encoder and dual heads
├── losses/               # local/scalar/conservative losses
├── utils/                # metrics, checkpoints, seeding, plots
├── train.py
├── evaluate.py
└── inspect_dataset.py
```

## Commands

Inspect the final dataset:

```bash
python motion_estimator/inspect_dataset.py
```

Run a one-batch smoke test:

```bash
python motion_estimator/train.py \
  --config motion_estimator/configs/default.yaml \
  --device cuda \
  --num_workers 0 \
  --epochs 1 --max_train_batches 1 --max_val_batches 1
```

Run the first full model experiment:

```bash
python motion_estimator/train.py \
  --config motion_estimator/configs/default.yaml \
  --device cuda
```

Evaluate the best checkpoint:

```bash
python motion_estimator/evaluate.py \
  --checkpoint motion_estimator/artifacts/<run>/best.pt \
  --device cuda
```

View training curves while a run is active:

```bash
tensorboard --logdir motion_estimator/artifacts/<run>/tensorboard
```

Each run stores `config.yaml`, train-only `normalization.json`,
`train_log.csv`, `best.pt`, `last.pt`, `metrics.json`, and figures under
`motion_estimator/artifacts/<run>/`.

The checkpoint contains model, optimizer and scheduler state, epoch, best
metric, normalization, and model configuration. Deployment integration with
a predictive time-varying CBF is intentionally a later phase.
