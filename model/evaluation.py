#!/usr/bin/env python3
"""
Evaluation metrics and visualization tools for the painting system.

This module provides:
  - SSIM (Structural Similarity Index)
  - LPIPS (Learned Perceptual Image Patch Similarity) - simplified
  - MSE (Mean Squared Error)
  - PSNR (Peak Signal-to-Noise Ratio)
  - Visualization tools for training progress and painting steps

All metrics work on both numpy arrays and PyTorch tensors.
"""
from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False
    torch = None
    nn = None
    F = None


# ---------------------------------------------------------------------------
# Image quality metrics
# ---------------------------------------------------------------------------

def _to_numpy(img: Union[np.ndarray, Any]) -> np.ndarray:
    """Convert input to numpy array in [0, 1] range, shape (H, W, 3)."""
    if _HAS_TORCH and isinstance(img, torch.Tensor):
        img = img.detach().cpu().numpy()
    img = np.asarray(img)
    if img.ndim == 4:
        img = img[0]  # remove batch dim
    if img.ndim == 3 and img.shape[0] in (1, 3):  # CHW -> HWC
        img = np.transpose(img, (1, 2, 0))
    if img.dtype == np.uint8:
        img = img.astype(np.float32) / 255.0
    return img


def mse(img1: Union[np.ndarray, Any], img2: Union[np.ndarray, Any]) -> float:
    """Mean Squared Error between two images.

    Args:
        img1, img2: images in [0, 1] range, any shape (H,W,3) or (3,H,W)

    Returns:
        MSE value (0 = identical, higher = more different)
    """
    a = _to_numpy(img1)
    b = _to_numpy(img2)
    return float(np.mean((a - b) ** 2))


def psnr(img1: Union[np.ndarray, Any], img2: Union[np.ndarray, Any],
         max_val: float = 1.0) -> float:
    """Peak Signal-to-Noise Ratio.

    Args:
        img1, img2: images in [0, max_val] range
        max_val: maximum possible pixel value (1.0 or 255.0)

    Returns:
        PSNR in dB (higher = better, inf = identical)
    """
    m = mse(img1, img2)
    if m == 0:
        return float('inf')
    return float(10.0 * np.log10(max_val ** 2 / m))


def ssim(img1: Union[np.ndarray, Any], img2: Union[np.ndarray, Any],
         window_size: int = 11, data_range: float = 1.0) -> float:
    """Structural Similarity Index (SSIM).

    Uses a Gaussian window of size `window_size`. Implementation follows
    the standard SSIM formula from Wang et al. (2004).

    Args:
        img1, img2: images in [0, data_range] range
        window_size: size of the Gaussian window (odd)
        data_range: dynamic range of pixel values

    Returns:
        SSIM value in [-1, 1] (1 = identical)
    """
    a = _to_numpy(img1).astype(np.float64)
    b = _to_numpy(img2).astype(np.float64)

    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")

    # Ensure 3D
    if a.ndim == 2:
        a = a[..., np.newaxis]
        b = b[..., np.newaxis]

    # Create Gaussian window
    sigma = 1.5
    coords = np.arange(window_size) - window_size // 2
    g = np.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    window = np.outer(g, g)

    # Constants for stability
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2

    # Compute per-channel SSIM and average
    ssim_values = []
    for c in range(a.shape[2]):
        mu1 = _convolve2d(a[..., c], window)
        mu2 = _convolve2d(b[..., c], window)
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2

        sigma1_sq = _convolve2d(a[..., c] ** 2, window) - mu1_sq
        sigma2_sq = _convolve2d(b[..., c] ** 2, window) - mu2_sq
        sigma12 = _convolve2d(a[..., c] * b[..., c], window) - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        ssim_values.append(ssim_map.mean())

    return float(np.mean(ssim_values))


def _convolve2d(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """2D convolution with 'valid' padding using scipy or manual."""
    try:
        from scipy.signal import convolve2d
        return convolve2d(img, kernel, mode='valid')
    except ImportError:
        # Manual implementation (slower but no scipy dependency)
        kh, kw = kernel.shape
        ih, iw = img.shape
        oh, ow = ih - kh + 1, iw - kw + 1
        out = np.zeros((oh, ow), dtype=np.float64)
        for i in range(oh):
            for j in range(ow):
                out[i, j] = np.sum(img[i:i+kh, j:j+kw] * kernel)
        return out


def lpips_simplified(img1: Union[np.ndarray, Any],
                     img2: Union[np.ndarray, Any]) -> float:
    """Simplified LPIPS-like perceptual distance.

    This is a lightweight approximation of LPIPS that uses multi-scale
    feature differences instead of a deep network. For the full LPIPS,
    install the `lpips` package.

    Returns:
        Perceptual distance (0 = identical, higher = more different)
    """
    a = _to_numpy(img1).astype(np.float32)
    b = _to_numpy(img2).astype(np.float32)

    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")

    # Multi-scale L2 distance
    total = 0.0
    weight = 1.0
    total_weight = 0.0
    cur_a, cur_b = a, b
    for scale in range(4):
        diff = np.mean((cur_a - cur_b) ** 2)
        total += weight * diff
        total_weight += weight
        weight *= 0.5
        if min(cur_a.shape[:2]) > 4:
            cur_a = cv2_resize(cur_a, 0.5)
            cur_b = cv2_resize(cur_b, 0.5)

    return float(total / total_weight)


def cv2_resize(img: np.ndarray, scale: float) -> np.ndarray:
    """Resize image using OpenCV."""
    import cv2
    h, w = img.shape[:2]
    new_h, new_w = int(h * scale), int(w * scale)
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def compute_all_metrics(img1: Union[np.ndarray, Any],
                        img2: Union[np.ndarray, Any]) -> Dict[str, float]:
    """Compute all available metrics at once.

    Returns:
        Dict with keys: 'mse', 'psnr', 'ssim', 'lpips'
    """
    return {
        'mse': mse(img1, img2),
        'psnr': psnr(img1, img2),
        'ssim': ssim(img1, img2),
        'lpips': lpips_simplified(img1, img2),
    }


# ---------------------------------------------------------------------------
# Visualization tools
# ---------------------------------------------------------------------------

class TrainingVisualizer:
    """Visualize training progress (losses, rewards, metrics over time)."""

    def __init__(self, save_dir: str = 'training_plots'):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.history: Dict[str, List[float]] = {
            'rewards': [],
            'critic_losses': [],
            'actor_losses': [],
            'ssim': [],
            'mse': [],
        }

    def record(self, **kwargs):
        """Record a training step.

        Keyword args can include: reward, critic_loss, actor_loss, ssim, mse
        """
        if 'reward' in kwargs:
            self.history['rewards'].append(kwargs['reward'])
        if 'critic_loss' in kwargs:
            self.history['critic_losses'].append(kwargs['critic_loss'])
        if 'actor_loss' in kwargs:
            self.history['actor_losses'].append(kwargs['actor_loss'])
        if 'ssim' in kwargs:
            self.history['ssim'].append(kwargs['ssim'])
        if 'mse' in kwargs:
            self.history['mse'].append(kwargs['mse'])

    def plot(self, show: bool = False):
        """Generate and save training plots."""
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
        except ImportError:
            print("[viz] matplotlib not available, skipping plots")
            return

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle('Training Progress')

        # Rewards
        if self.history['rewards']:
            axes[0, 0].plot(self.history['rewards'])
            axes[0, 0].set_title('Episode Rewards')
            axes[0, 0].set_xlabel('Episode')
            axes[0, 0].set_ylabel('Reward')

        # Losses
        if self.history['critic_losses']:
            axes[0, 1].plot(self.history['critic_losses'], label='Critic')
            axes[0, 1].plot(self.history['actor_losses'], label='Actor')
            axes[0, 1].set_title('Losses')
            axes[0, 1].set_xlabel('Step')
            axes[0, 1].set_ylabel('Loss')
            axes[0, 1].legend()

        # SSIM
        if self.history['ssim']:
            axes[1, 0].plot(self.history['ssim'])
            axes[1, 0].set_title('SSIM (higher is better)')
            axes[1, 0].set_xlabel('Episode')
            axes[1, 0].set_ylabel('SSIM')

        # MSE
        if self.history['mse']:
            axes[1, 1].plot(self.history['mse'])
            axes[1, 1].set_title('MSE (lower is better)')
            axes[1, 1].set_xlabel('Episode')
            axes[1, 1].set_ylabel('MSE')

        plt.tight_layout()
        path = os.path.join(self.save_dir, 'training_progress.png')
        plt.savefig(path, dpi=100)
        plt.close()
        print(f"[viz] Saved training plot to {path}")

    def save_history(self, path: str = None):
        """Save training history as JSON."""
        import json
        path = path or os.path.join(self.save_dir, 'history.json')
        with open(path, 'w') as f:
            json.dump(self.history, f, indent=2)
        print(f"[viz] Saved history to {path}")


class PaintingVisualizer:
    """Visualize painting steps (progressive canvas snapshots)."""

    def __init__(self, save_dir: str = 'painting_steps'):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.steps: List[Tuple[int, np.ndarray]] = []

    def record(self, step: int, canvas: Union[np.ndarray, Any]):
        """Record a canvas snapshot at a given step."""
        img = _to_numpy(canvas)
        self.steps.append((step, img.copy()))

    def save_grid(self, cols: int = 4, filename: str = 'painting_grid.png'):
        """Save a grid of painting steps."""
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
        except ImportError:
            print("[viz] matplotlib not available")
            return

        n = len(self.steps)
        if n == 0:
            return
        rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
        if rows == 1:
            axes = axes[np.newaxis, :] if cols > 1 else np.array([[axes]])
        elif cols == 1:
            axes = axes[:, np.newaxis]

        for idx, (step, img) in enumerate(self.steps):
            r, c = idx // cols, idx % cols
            if r < axes.shape[0] and c < axes.shape[1]:
                axes[r, c].imshow(img)
                axes[r, c].set_title(f'Step {step}')
                axes[r, c].axis('off')

        # Hide unused subplots
        for idx in range(n, rows * cols):
            r, c = idx // cols, idx % cols
            if r < axes.shape[0] and c < axes.shape[1]:
                axes[r, c].axis('off')

        plt.tight_layout()
        path = os.path.join(self.save_dir, filename)
        plt.savefig(path, dpi=100)
        plt.close()
        print(f"[viz] Saved painting grid to {path}")


# ---------------------------------------------------------------------------
# Comparison utilities
# ---------------------------------------------------------------------------

def compare_painting_methods(target: np.ndarray,
                             results: Dict[str, np.ndarray]) -> Dict[str, Dict[str, float]]:
    """Compare multiple painting methods against a target image.

    Args:
        target: target image (H, W, 3) in [0, 1]
        results: dict of {method_name: painted_image}

    Returns:
        Dict of {method_name: {metric_name: value}}
    """
    all_metrics = {}
    for name, img in results.items():
        all_metrics[name] = compute_all_metrics(target, img)
    return all_metrics


def print_comparison_table(metrics: Dict[str, Dict[str, float]]):
    """Print a formatted comparison table."""
    print(f"\n{'Method':<20} {'MSE':<10} {'PSNR':<10} {'SSIM':<10} {'LPIPS':<10}")
    print("-" * 60)
    for name, m in metrics.items():
        print(f"{name:<20} {m['mse']:<10.4f} {m['psnr']:<10.2f} "
              f"{m['ssim']:<10.4f} {m['lpips']:<10.4f}")


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_evaluator():
    """Create a metrics evaluator function."""
    return compute_all_metrics
