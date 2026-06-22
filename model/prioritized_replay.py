#!/usr/bin/env python3
"""
Prioritized Experience Replay buffer and enhanced training utilities.

This module implements:
  - PrioritizedReplayBuffer: experience replay with priority sampling
    based on TD-error (Schaul et al., 2015)
  - MultiDimensionalReward: combines SSIM + LPIPS + color + edge rewards
  - TrainingStability: gradient clipping, learning rate scheduling,
    early stopping, and training monitoring

These components enhance the DDPG training system with better sample
efficiency and training stability.
"""
from __future__ import annotations

import math
import os
import random
import sys
from typing import Any, Dict, List, Optional, Tuple

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
# Sum tree for prioritized replay (efficient priority sampling)
# ---------------------------------------------------------------------------

class SumTree:
    """Sum tree data structure for O(log n) priority updates and sampling.

    Used by PrioritizedReplayBuffer to efficiently sample transitions
    proportional to their priority.
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        # Tree size = 2 * capacity - 1 (leaves + internal nodes)
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float64)
        self.max_priority = 1.0

    def update(self, idx: int, priority: float):
        """Update the priority of leaf at index `idx`."""
        tree_idx = idx + self.capacity - 1
        diff = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        # Propagate up the tree
        while tree_idx > 0:
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += diff
        self.max_priority = max(self.max_priority, priority)

    def get_total(self) -> float:
        """Return the total priority sum."""
        return float(self.tree[0])

    def sample(self, value: float) -> Tuple[int, float]:
        """Sample a leaf index by cumulative priority value.

        Returns (leaf_index, priority).
        """
        idx = 0
        while idx < self.capacity - 1:
            left = 2 * idx + 1
            right = left + 1
            if value <= self.tree[left]:
                idx = left
            else:
                value -= self.tree[left]
                idx = right
        leaf_idx = idx - self.capacity + 1
        return leaf_idx, float(self.tree[idx])


# ---------------------------------------------------------------------------
# Prioritized experience replay buffer
# ---------------------------------------------------------------------------

class PrioritizedReplayBuffer:
    """Prioritized experience replay buffer.

    Samples transitions proportional to their TD-error priority, which
    improves sample efficiency over uniform replay.

    References:
        Schaul et al., "Prioritized Experience Replay", ICLR 2016
    """

    def __init__(
        self,
        capacity: int = 10000,
        alpha: float = 0.6,      # priority exponent (0 = uniform, 1 = full priority)
        beta: float = 0.4,       # importance sampling exponent (annealed to 1.0)
        beta_annealing: float = 0.001,
        epsilon: float = 1e-6,
    ):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_annealing = beta_annealing
        self.epsilon = epsilon
        self.tree = SumTree(capacity)
        self.buffer: List[dict] = []
        self.position = 0

    def push(
        self,
        target: Any,
        canvas: Any,
        action: Any,
        reward: float,
        next_target: Any,
        next_canvas: Any,
        done: bool,
        hidden_h: Any = None,
        hidden_c: Any = None,
    ):
        """Add a transition to the buffer with max priority."""
        transition = {
            'target': target,
            'canvas': canvas,
            'action': action,
            'reward': reward,
            'next_target': next_target,
            'next_canvas': next_canvas,
            'done': done,
            'hidden_h': hidden_h,
            'hidden_c': hidden_c,
        }
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.position] = transition

        # Use max priority for new transitions
        self.tree.update(self.position, self.tree.max_priority)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int) -> Tuple[List[dict], List[int], np.ndarray]:
        """Sample a batch of transitions by priority.

        Returns (transitions, indices, weights) where weights are
        importance-sampling weights for bias correction.
        """
        if len(self.buffer) == 0:
            return [], [], np.array([])

        total = self.tree.get_total()
        segment = total / batch_size

        transitions = []
        indices = []
        priorities = []

        for i in range(batch_size):
            value = random.uniform(segment * i, segment * (i + 1))
            idx, priority = self.tree.sample(value)
            idx = min(idx, len(self.buffer) - 1)
            transitions.append(self.buffer[idx])
            indices.append(idx)
            priorities.append(priority)

        priorities = np.array(priorities)
        # Compute importance sampling weights
        probs = priorities / (total + self.epsilon)
        weights = (len(self.buffer) * probs) ** (-self.beta)
        weights = weights / weights.max()

        # Anneal beta towards 1.0
        self.beta = min(1.0, self.beta + self.beta_annealing)

        return transitions, indices, weights

    def update_priorities(self, indices: List[int], td_errors: np.ndarray):
        """Update priorities based on TD-errors."""
        for idx, td in zip(indices, td_errors):
            priority = (abs(td) + self.epsilon) ** self.alpha
            self.tree.update(idx, priority)

    def __len__(self):
        return len(self.buffer)


# ---------------------------------------------------------------------------
# Multi-dimensional reward function
# ---------------------------------------------------------------------------

class MultiDimensionalReward:
    """Combines multiple perceptual metrics into a single reward.

    Reward components:
      - SSIM (structural similarity)
      - LPIPS (perceptual distance, simplified)
      - Color distribution similarity
      - Edge preservation

    The final reward is a weighted sum of these components.
    """

    def __init__(
        self,
        w_ssim: float = 0.3,
        w_lpips: float = 0.3,
        w_color: float = 0.2,
        w_edge: float = 0.2,
        device: str = 'cpu',
    ):
        self.w_ssim = w_ssim
        self.w_lpips = w_lpips
        self.w_color = w_color
        self.w_edge = w_edge
        self.device = device

    def _to_numpy(self, img: Any) -> np.ndarray:
        """Convert tensor to numpy (H, W, 3) in [0, 1]."""
        if _HAS_TORCH and isinstance(img, torch.Tensor):
            img = img.detach().cpu().numpy()
        img = np.asarray(img)
        if img.ndim == 4:
            img = img[0]
        if img.ndim == 3 and img.shape[0] in (1, 3):
            img = np.transpose(img, (1, 2, 0))
        if img.dtype == np.uint8:
            img = img.astype(np.float32) / 255.0
        return img

    def _ssim(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """Simplified SSIM."""
        try:
            from skimage.metrics import structural_similarity
            return float(structural_similarity(img1, img2, channel_axis=2, data_range=1.0))
        except ImportError:
            # Fallback: mean-based similarity
            return float(1.0 - np.mean(np.abs(img1 - img2)))

    def _lpips_simplified(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """Simplified LPIPS using feature distance."""
        # Use Gaussian blur as a simple feature extractor
        try:
            import cv2
            g1 = cv2.GaussianBlur(img1, (5, 5), 1.0)
            g2 = cv2.GaussianBlur(img2, (5, 5), 1.0)
            dist = np.mean((g1 - g2) ** 2)
            return float(1.0 / (1.0 + dist))
        except ImportError:
            return float(1.0 - np.mean(np.abs(img1 - img2)))

    def _color_similarity(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """Color histogram similarity."""
        try:
            import cv2
            h1 = cv2.calcHist([img1], [0, 1, 2], None, [8, 8, 8], [0, 1, 0, 1, 0, 1])
            h2 = cv2.calcHist([img2], [0, 1, 2], None, [8, 8, 8], [0, 1, 0, 1, 0, 1])
            h1 = h1 / (h1.sum() + 1e-8)
            h2 = h2 / (h2.sum() + 1e-8)
            return float(cv2.compareHist(h1, h2, cv2.HISTCMP_CORREL))
        except ImportError:
            return float(1.0 - np.mean(np.abs(img1.mean(axis=(0,1)) - img2.mean(axis=(0,1)))))

    def _edge_preservation(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """Edge preservation score."""
        try:
            import cv2
            e1 = cv2.Canny((img1 * 255).astype(np.uint8), 50, 150)
            e2 = cv2.Canny((img2 * 255).astype(np.uint8), 50, 150)
            intersection = np.logical_and(e1 > 0, e2 > 0).sum()
            union = np.logical_or(e1 > 0, e2 > 0).sum()
            return float(intersection / (union + 1e-8))
        except ImportError:
            return 1.0

    def compute(self, target: Any, canvas: Any) -> Dict[str, float]:
        """Compute multi-dimensional reward.

        Returns dict with individual component scores and the combined reward.
        """
        t = self._to_numpy(target)
        c = self._to_numpy(canvas)

        ssim = self._ssim(t, c)
        lpips = self._lpips_simplified(t, c)
        color = self._color_similarity(t, c)
        edge = self._edge_preservation(t, c)

        reward = (self.w_ssim * ssim +
                  self.w_lpips * lpips +
                  self.w_color * color +
                  self.w_edge * edge)

        return {
            'reward': float(reward),
            'ssim': float(ssim),
            'lpips': float(lpips),
            'color': float(color),
            'edge': float(edge),
        }


# ---------------------------------------------------------------------------
# Training stability utilities
# ---------------------------------------------------------------------------

class TrainingStability:
    """Training stability utilities: gradient clipping, LR scheduling, early stopping."""

    def __init__(
        self,
        grad_clip: float = 1.0,
        lr_schedule: str = 'cosine',  # 'constant', 'cosine', 'linear'
        initial_lr: float = 1e-4,
        min_lr: float = 1e-6,
        warmup_steps: int = 100,
        early_stop_patience: int = 50,
        early_stop_min_delta: float = 1e-4,
    ):
        self.grad_clip = grad_clip
        self.lr_schedule = lr_schedule
        self.initial_lr = initial_lr
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        self.early_stop_patience = early_stop_patience
        self.early_stop_min_delta = early_stop_min_delta
        self.step_count = 0
        self.best_loss = float('inf')
        self.patience_counter = 0
        self.should_stop = False

    def clip_gradients(self, model: nn.Module) -> float:
        """Clip gradients and return the gradient norm."""
        if not _HAS_TORCH:
            return 0.0
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), self.grad_clip)
        return float(grad_norm)

    def get_lr(self, total_steps: int) -> float:
        """Compute learning rate based on schedule."""
        self.step_count += 1
        if self.step_count < self.warmup_steps:
            # Linear warmup
            return self.initial_lr * (self.step_count / self.warmup_steps)

        progress = (self.step_count - self.warmup_steps) / max(1, total_steps - self.warmup_steps)
        progress = min(1.0, max(0.0, progress))

        if self.lr_schedule == 'cosine':
            lr = self.min_lr + 0.5 * (self.initial_lr - self.min_lr) * (1 + math.cos(math.pi * progress))
        elif self.lr_schedule == 'linear':
            lr = self.initial_lr - (self.initial_lr - self.min_lr) * progress
        else:
            lr = self.initial_lr
        return lr

    def update_lr(self, optimizer, total_steps: int):
        """Update optimizer learning rate based on schedule."""
        lr = self.get_lr(total_steps)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        return lr

    def check_early_stop(self, loss: float) -> bool:
        """Check if training should stop early."""
        if loss < self.best_loss - self.early_stop_min_delta:
            self.best_loss = loss
            self.patience_counter = 0
        else:
            self.patience_counter += 1
            if self.patience_counter >= self.early_stop_patience:
                self.should_stop = True
        return self.should_stop

    def reset(self):
        """Reset state for a new training run."""
        self.step_count = 0
        self.best_loss = float('inf')
        self.patience_counter = 0
        self.should_stop = False


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------

def build_prioritized_replay_buffer(capacity: int = 10000, **kwargs) -> PrioritizedReplayBuffer:
    """Create a PrioritizedReplayBuffer."""
    return PrioritizedReplayBuffer(capacity=capacity, **kwargs)


def build_multi_dimensional_reward(**kwargs) -> MultiDimensionalReward:
    """Create a MultiDimensionalReward."""
    return MultiDimensionalReward(**kwargs)


def build_training_stability(**kwargs) -> TrainingStability:
    """Create a TrainingStability."""
    return TrainingStability(**kwargs)
