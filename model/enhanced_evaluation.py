#!/usr/bin/env python3
"""
Enhanced evaluation system with comprehensive perceptual metrics.

This module provides:
  - FSIM (Feature Similarity Index) for phase-aware quality assessment
  - Color distribution similarity (histogram correlation)
  - Edge preservation metric
  - Painting feature metrics (stroke count, painting time, human-likeness)
  - Style consistency evaluation
  - A/B testing framework for user studies

All metrics work on numpy arrays in [0, 1] range.
"""
from __future__ import annotations

import math
import os
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False
    torch = None


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _to_numpy(img: Union[np.ndarray, Any]) -> np.ndarray:
    """Convert input to numpy array in [0, 1] range, shape (H, W, 3)."""
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


# ---------------------------------------------------------------------------
# FSIM (Feature Similarity Index)
# ---------------------------------------------------------------------------

def fsim(img1: np.ndarray, img2: np.ndarray) -> float:
    """Compute FSIM (Feature Similarity Index).

    FSIM uses phase congruency and gradient magnitude to measure
    image quality. Higher is better (1.0 = identical).

    This is a simplified implementation using gradient magnitude only
    (full phase congruency requires complex wavelet transforms).

    Args:
        img1, img2: images in [0, 1] range, shape (H, W, 3)

    Returns:
        FSIM score in [0, 1]
    """
    img1 = _to_numpy(img1)
    img2 = _to_numpy(img2)

    # Convert to grayscale
    g1 = np.dot(img1[..., :3], [0.299, 0.587, 0.114])
    g2 = np.dot(img2[..., :3], [0.299, 0.587, 0.114])

    # Compute gradient magnitude using Scharr operator
    def gradient_magnitude(gray):
        gx = np.array([[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]], dtype=np.float32)
        gy = gx.T
        gx_img = _convolve2d(gray, gx)
        gy_img = _convolve2d(gray, gy)
        return np.sqrt(gx_img ** 2 + gy_img ** 2)

    gm1 = gradient_magnitude(g1)
    gm2 = gradient_magnitude(g2)

    # Compute similarity
    T1 = 0.85 * gm1.max() if gm1.max() > 0 else 1e-6
    T2 = 0.85 * gm2.max() if gm2.max() > 0 else 1e-6

    S = (2 * gm1 * gm2 + T2) / (gm1 ** 2 + gm2 ** 2 + T2)

    # Weight by gradient magnitude
    weight = np.maximum(gm1, gm2)
    weight_sum = weight.sum()

    if weight_sum < 1e-6:
        return 1.0

    fsim_score = float((S * weight).sum() / weight_sum)
    return max(0.0, min(1.0, fsim_score))


def _convolve2d(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Simple 2D convolution with zero padding."""
    kh, kw = kernel.shape
    ph, pw = kh // 2, kw // 2
    padded = np.pad(img, ((ph, ph), (pw, pw)), mode='edge')
    result = np.zeros_like(img, dtype=np.float32)
    for i in range(kh):
        for j in range(kw):
            result += kernel[i, j] * padded[i:i + img.shape[0], j:j + img.shape[1]]
    return result


# ---------------------------------------------------------------------------
# Color distribution similarity
# ---------------------------------------------------------------------------

def color_distribution_similarity(img1: np.ndarray, img2: np.ndarray,
                                  bins: int = 32) -> float:
    """Compute color distribution similarity using histogram correlation.

    Args:
        img1, img2: images in [0, 1] range, shape (H, W, 3)
        bins: number of histogram bins per channel

    Returns:
        Similarity score in [0, 1] (1.0 = identical distributions)
    """
    img1 = _to_numpy(img1)
    img2 = _to_numpy(img2)

    correlations = []
    for c in range(3):
        hist1 = np.histogram(img1[..., c], bins=bins, range=(0, 1))[0].astype(np.float32)
        hist2 = np.histogram(img2[..., c], bins=bins, range=(0, 1))[0].astype(np.float32)

        # Normalize
        hist1 /= (hist1.sum() + 1e-6)
        hist2 /= (hist2.sum() + 1e-6)

        # Correlation
        m1 = hist1.mean()
        m2 = hist2.mean()
        num = ((hist1 - m1) * (hist2 - m2)).sum()
        den = math.sqrt(((hist1 - m1) ** 2).sum() * ((hist2 - m2) ** 2).sum())
        if den < 1e-6:
            correlations.append(1.0)
        else:
            correlations.append(float(num / den))

    # Average correlation across channels, map [-1, 1] to [0, 1]
    avg_corr = sum(correlations) / 3
    return max(0.0, min(1.0, (avg_corr + 1) / 2))


# ---------------------------------------------------------------------------
# Edge preservation metric
# ---------------------------------------------------------------------------

def edge_preservation(img1: np.ndarray, img2: np.ndarray) -> float:
    """Compute edge preservation ratio.

    Measures how well edges in img1 are preserved in img2.
    Uses Sobel edge detection and compares edge maps.

    Args:
        img1: target image (reference edges)
        img2: painted image (edges to compare)

    Returns:
        Preservation score in [0, 1] (1.0 = perfect edge preservation)
    """
    img1 = _to_numpy(img1)
    img2 = _to_numpy(img2)

    # Convert to grayscale
    g1 = np.dot(img1[..., :3], [0.299, 0.587, 0.114])
    g2 = np.dot(img2[..., :3], [0.299, 0.587, 0.114])

    # Sobel edges
    def sobel_edges(gray):
        gx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
        gy = gx.T
        gx_img = _convolve2d(gray, gx)
        gy_img = _convolve2d(gray, gy)
        return np.sqrt(gx_img ** 2 + gy_img ** 2)

    e1 = sobel_edges(g1)
    e2 = sobel_edges(g2)

    # Normalize
    e1 = e1 / (e1.max() + 1e-6)
    e2 = e2 / (e2.max() + 1e-6)

    # Threshold to get binary edge maps
    thresh = 0.2
    b1 = (e1 > thresh).astype(np.float32)
    b2 = (e2 > thresh).astype(np.float32)

    # IoU (Intersection over Union)
    intersection = (b1 * b2).sum()
    union = b1.sum() + b2.sum() - intersection

    if union < 1e-6:
        return 1.0  # no edges in either image

    return float(intersection / union)


# ---------------------------------------------------------------------------
# Painting feature metrics
# ---------------------------------------------------------------------------

def painting_features(strokes: List[Dict], painting_time: float = 0.0) -> Dict[str, float]:
    """Compute painting feature metrics from stroke data.

    Args:
        strokes: list of stroke dicts
        painting_time: total painting time in seconds

    Returns:
        Dict with feature metrics
    """
    if not strokes:
        return {
            'stroke_count': 0,
            'painting_time': painting_time,
            'avg_stroke_length': 0,
            'avg_radius': 0,
            'color_diversity': 0,
            'human_likeness': 0,
        }

    # Stroke lengths
    lengths = []
    for s in strokes:
        dx = s.get('x_end', 0) - s.get('x_start', 0)
        dy = s.get('y_end', 0) - s.get('y_start', 0)
        lengths.append(math.sqrt(dx ** 2 + dy ** 2))

    # Radii
    radii = [s.get('brush_radius', 0) for s in strokes]

    # Color diversity (unique colors)
    colors = set()
    for s in strokes:
        r = round(s.get('color_r', 0), 2)
        g = round(s.get('color_g', 0), 2)
        b = round(s.get('color_b', 0), 2)
        colors.add((r, g, b))

    # Human-likeness heuristic: based on stroke length variation and
    # color diversity. Human paintings tend to have varied stroke lengths
    # and moderate color diversity.
    len_var = np.std(lengths) / (np.mean(lengths) + 1e-6) if lengths else 0
    color_div = len(colors) / max(len(strokes), 1)
    # Human-likeness: high when length variation is moderate (0.3-0.8)
    # and color diversity is moderate (0.1-0.5)
    len_score = 1.0 - abs(len_var - 0.5) * 2 if len_var < 1.0 else 0
    color_score = 1.0 - abs(color_div - 0.3) * 3 if color_div < 0.6 else 0.5
    human_likeness = max(0, min(1, (len_score + color_score) / 2))

    return {
        'stroke_count': len(strokes),
        'painting_time': painting_time,
        'avg_stroke_length': float(np.mean(lengths)),
        'avg_radius': float(np.mean(radii)),
        'color_diversity': len(colors),
        'human_likeness': float(human_likeness),
    }


# ---------------------------------------------------------------------------
# Style consistency evaluation
# ---------------------------------------------------------------------------

def style_consistency(strokes: List[Dict], style_name: str = 'default') -> float:
    """Evaluate how consistent the painting is with a given style.

    Args:
        strokes: list of stroke dicts
        style_name: target style name

    Returns:
        Consistency score in [0, 1]
    """
    if not strokes:
        return 0.0

    radii = [s.get('brush_radius', 0) for s in strokes]
    alphas = [s.get('color_a', 0) for s in strokes]

    # Style expectations
    style_expectations = {
        'default': {'radius_var': 0.5, 'alpha_mean': 0.7},
        'oil': {'radius_var': 0.6, 'alpha_mean': 0.8},
        'watercolor': {'radius_var': 0.4, 'alpha_mean': 0.5},
        'sketch': {'radius_var': 0.3, 'alpha_mean': 0.9},
        'anime': {'radius_var': 0.5, 'alpha_mean': 0.85},
    }

    exp = style_expectations.get(style_name, style_expectations['default'])

    # Compute consistency
    radius_var = np.std(radii) / (np.mean(radii) + 1e-6) if radii else 0
    alpha_mean = np.mean(alphas) if alphas else 0

    # Score based on how close to expectations
    var_score = 1.0 - abs(radius_var - exp['radius_var'])
    alpha_score = 1.0 - abs(alpha_mean - exp['alpha_mean'])

    return max(0.0, min(1.0, (var_score + alpha_score) / 2))


# ---------------------------------------------------------------------------
# Comprehensive evaluation
# ---------------------------------------------------------------------------

def compute_enhanced_metrics(target: np.ndarray, painted: np.ndarray,
                             strokes: Optional[List[Dict]] = None,
                             painting_time: float = 0.0,
                             style: str = 'default') -> Dict[str, float]:
    """Compute all available metrics.

    Args:
        target: target image (H, W, 3) in [0, 1]
        painted: painted image (H, W, 3) in [0, 1]
        strokes: optional list of stroke dicts
        painting_time: total painting time in seconds
        style: style name for consistency evaluation

    Returns:
        Dict with all metrics
    """
    from evaluation import mse, psnr, ssim, lpips_simplified

    target = _to_numpy(target)
    painted = _to_numpy(painted)

    metrics = {
        'mse': mse(target, painted),
        'psnr': psnr(target, painted),
        'ssim': ssim(target, painted),
        'lpips': lpips_simplified(target, painted),
        'fsim': fsim(target, painted),
        'color_similarity': color_distribution_similarity(target, painted),
        'edge_preservation': edge_preservation(target, painted),
    }

    if strokes is not None:
        features = painting_features(strokes, painting_time)
        metrics.update(features)
        metrics['style_consistency'] = style_consistency(strokes, style)

    # Overall quality score (weighted average)
    weights = {
        'ssim': 0.25,
        'fsim': 0.20,
        'color_similarity': 0.15,
        'edge_preservation': 0.15,
        'lpips': 0.25,  # inverted (lower is better)
    }
    overall = 0
    for k, w in weights.items():
        if k == 'lpips':
            overall += w * (1 - metrics[k])  # invert LPIPS
        else:
            overall += w * metrics[k]
    metrics['overall_quality'] = float(overall)

    return metrics


# ---------------------------------------------------------------------------
# A/B testing framework
# ---------------------------------------------------------------------------

class ABTestFramework:
    """A/B testing framework for comparing painting methods.

    Allows comparing two methods by collecting user ratings on
    randomly presented pairs.
    """

    def __init__(self, method_a_name: str = 'A', method_b_name: str = 'B'):
        self.method_a_name = method_a_name
        self.method_b_name = method_b_name
        self.results: List[Dict] = []

    def add_rating(self, user_id: str, image_id: str,
                   rating_a: float, rating_b: float,
                   preferred: str = 'none'):
        """Add a user rating for a pair.

        Args:
            user_id: unique user identifier
            image_id: image being rated
            rating_a: rating for method A (0-1)
            rating_b: rating for method B (0-1)
            preferred: 'a', 'b', or 'none'
        """
        self.results.append({
            'user_id': user_id,
            'image_id': image_id,
            'rating_a': rating_a,
            'rating_b': rating_b,
            'preferred': preferred,
        })

    def compute_statistics(self) -> Dict:
        """Compute A/B test statistics.

        Returns:
            Dict with mean ratings, win rates, and sample size
        """
        if not self.results:
            return {'sample_size': 0}

        ratings_a = [r['rating_a'] for r in self.results]
        ratings_b = [r['rating_b'] for r in self.results]

        wins_a = sum(1 for r in self.results if r['preferred'] == 'a')
        wins_b = sum(1 for r in self.results if r['preferred'] == 'b')

        return {
            'sample_size': len(self.results),
            'mean_rating_a': float(np.mean(ratings_a)),
            'mean_rating_b': float(np.mean(ratings_b)),
            'std_rating_a': float(np.std(ratings_a)),
            'std_rating_b': float(np.std(ratings_b)),
            'win_rate_a': wins_a / len(self.results),
            'win_rate_b': wins_b / len(self.results),
            'tie_rate': (len(self.results) - wins_a - wins_b) / len(self.results),
        }

    def print_report(self):
        """Print a formatted A/B test report."""
        stats = self.compute_statistics()
        if stats['sample_size'] == 0:
            print("No data collected yet.")
            return

        print(f"\n{'='*50}")
        print(f"A/B Test Report ({stats['sample_size']} ratings)")
        print(f"{'='*50}")
        print(f"Method A ({self.method_a_name}):")
        print(f"  Mean rating: {stats['mean_rating_a']:.3f} ± {stats['std_rating_a']:.3f}")
        print(f"  Win rate:    {stats['win_rate_a']:.1%}")
        print(f"Method B ({self.method_b_name}):")
        print(f"  Mean rating: {stats['mean_rating_b']:.3f} ± {stats['std_rating_b']:.3f}")
        print(f"  Win rate:    {stats['win_rate_b']:.1%}")
        print(f"Tie rate: {stats['tie_rate']:.1%}")
        diff = stats['mean_rating_b'] - stats['mean_rating_a']
        winner = self.method_b_name if diff > 0 else self.method_a_name
        print(f"Winner: {winner} (diff={abs(diff):.3f})")
        print(f"{'='*50}")


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_ab_test(method_a: str = 'A', method_b: str = 'B') -> ABTestFramework:
    """Create an A/B test framework."""
    return ABTestFramework(method_a, method_b)
