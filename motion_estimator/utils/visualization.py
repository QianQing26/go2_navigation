"""Small matplotlib-only visualizations for training and test reports."""

import os

import numpy as np


def _plt():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    return plt


def plot_loss_curve(rows, path):
    plt = _plt()
    epochs = [row['epoch'] for row in rows]
    plt.figure(figsize=(8, 5))
    for key in ('train_loss', 'val_loss', 'val_drift_mae'):
        values = [row[key] for row in rows if key in row]
        if values:
            plt.plot(epochs[:len(values)], values, label=key)
    plt.xlabel('epoch')
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()


def plot_scatter(pred, target, path, xlabel, ylabel, title):
    plt = _plt()
    pred, target = np.asarray(pred).reshape(-1), np.asarray(target).reshape(-1)
    if pred.size > 100000:
        keep = np.linspace(0, pred.size - 1, 100000).astype(np.int64)
        pred, target = pred[keep], target[keep]
    plt.figure(figsize=(6, 6))
    plt.hexbin(target, pred, gridsize=60, mincnt=1, bins='log')
    limit = max(float(np.max(np.abs(target))), float(np.max(np.abs(pred))), 1.0e-3)
    plt.plot([-limit, limit], [-limit, limit], 'r--', label='y=x')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()


def plot_error_distribution(pred, target, path, title, xlabel):
    plt = _plt()
    error = np.asarray(pred).reshape(-1) - np.asarray(target).reshape(-1)
    plt.figure(figsize=(8, 5))
    plt.hist(error, bins=100)
    plt.xlabel(xlabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()


def plot_test_example(example, angles, path):
    plt = _plt()
    plt.figure(figsize=(9, 5))
    plt.plot(angles, example['closing_gt'], label='GT c')
    plt.plot(angles, example['closing_pred'], label='Pred c')
    plt.plot(angles, example['current_rays'], label='current ray', alpha=0.7)
    plt.xlabel('angle [deg]')
    plt.ylabel('value')
    plt.title('GT d_h={:.4f}, Pred d_h={:.4f}'.format(
        example['drift_gt'], example['drift_pred']
    ))
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()
