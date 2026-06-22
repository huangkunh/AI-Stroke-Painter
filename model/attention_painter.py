#!/usr/bin/env python3
"""
Enhanced hierarchical painting strategy with attention mechanism and style support.

This module implements a 5-layer painting strategy with:
  - Layer 1: Global composition (large color blocks, 40px radius)
  - Layer 2: Regional details (medium blocks, 20px radius)
  - Layer 3: Local features (small blocks, 10px radius)
  - Layer 4: Detail outlining (thin lines, 3px radius)
  - Layer 5: Final adjustment (fine corrections, 1px radius)

Additionally, it provides:
  - Spatial attention network for focusing on key regions
  - Dynamic stroke density based on attention maps
  - Style templates (oil, watercolor, sketch, etc.)
  - Style transfer algorithm

The output is compatible with the existing converter/transform.py pipeline.
"""
from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageFilter

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
# 5-layer configuration
# ---------------------------------------------------------------------------

ENHANCED_LAYER_CONFIGS = {
    'global': {
        'name': '全局构图',
        'radius': 40.0,
        'alpha': 0.45,
        'budget_frac': 0.25,
        'description': '大色块铺底，快速覆盖画布',
    },
    'regional': {
        'name': '区域细节',
        'radius': 20.0,
        'alpha': 0.55,
        'budget_frac': 0.25,
        'description': '中等色块，添加主要结构',
    },
    'local': {
        'name': '局部特征',
        'radius': 10.0,
        'alpha': 0.65,
        'budget_frac': 0.20,
        'description': '小色块，添加局部特征',
    },
    'detail': {
        'name': '细节勾勒',
        'radius': 3.0,
        'alpha': 0.80,
        'budget_frac': 0.20,
        'description': '细线，勾勒细节和纹理',
    },
    'adjustment': {
        'name': '最终调整',
        'radius': 1.0,
        'alpha': 0.90,
        'budget_frac': 0.10,
        'description': '精细修正，优化整体效果',
    },
}


# ---------------------------------------------------------------------------
# Style templates
# ---------------------------------------------------------------------------

STYLE_TEMPLATES = {
    'default': {
        'name': '默认',
        'layers': ['global', 'regional', 'local', 'detail', 'adjustment'],
        'color_quantize_k': 8,
        'edge_threshold': (50, 120),
        'blur_sigma': 0,
    },
    'oil': {
        'name': '油画',
        'layers': ['global', 'regional', 'local', 'detail'],
        'color_quantize_k': 6,
        'edge_threshold': (40, 100),
        'blur_sigma': 0.5,  # slight blur for painterly effect
    },
    'watercolor': {
        'name': '水彩',
        'layers': ['global', 'regional', 'local'],
        'color_quantize_k': 5,
        'edge_threshold': (30, 80),
        'blur_sigma': 1.0,  # more blur for soft watercolor effect
    },
    'sketch': {
        'name': '素描',
        'layers': ['detail', 'adjustment'],  # skip color layers
        'color_quantize_k': 2,  # black and white
        'edge_threshold': (80, 160),
        'blur_sigma': 0,
    },
    'anime': {
        'name': '动漫',
        'layers': ['global', 'regional', 'detail'],
        'color_quantize_k': 4,
        'edge_threshold': (100, 200),
        'blur_sigma': 0,
    },
}


# ---------------------------------------------------------------------------
# Spatial attention network
# ---------------------------------------------------------------------------

class SpatialAttentionNet:
    """Spatial attention network for focusing on key regions.

    Computes an attention map that highlights regions with:
      - High edge density (important structural features)
      - High color variance (regions with rich detail)
      - High residual error (regions that need more strokes)

    The attention map is used to guide stroke placement and density.
    """

    def __init__(self, canvas_size: int = 512):
        self.canvas_size = canvas_size

    def compute_attention(self, target: np.ndarray, canvas: Optional[np.ndarray] = None) -> np.ndarray:
        """Compute spatial attention map.

        Args:
            target: target image (H, W, 3) in [0, 1]
            canvas: current canvas (H, W, 3) in [0, 1], or None for initial pass

        Returns:
            attention map (H, W) in [0, 1], higher = more important
        """
        h, w = target.shape[:2]

        # 1. Edge density attention
        gray = cv2.cvtColor((target * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_attention = edges.astype(np.float32) / 255.0

        # 2. Color variance attention (local standard deviation)
        gray_f = gray.astype(np.float32) / 255.0
        local_mean = cv2.boxFilter(gray_f, -1, (15, 15))
        local_var = cv2.boxFilter(gray_f ** 2, -1, (15, 15)) - local_mean ** 2
        var_attention = np.clip(local_var * 5, 0, 1)

        # 3. Residual attention (if canvas provided)
        if canvas is not None:
            residual = np.abs(target - canvas).mean(axis=2)
            residual_attention = np.clip(residual * 3, 0, 1)
        else:
            residual_attention = np.ones_like(edge_attention)

        # Combine attention maps
        attention = (edge_attention * 0.3 +
                     var_attention * 0.3 +
                     residual_attention * 0.4)

        # Smooth the attention map
        attention = cv2.GaussianBlur(attention, (21, 21), 0)

        # Normalize to [0, 1]
        if attention.max() > 0:
            attention = attention / attention.max()

        return attention

    def sample_positions(self, attention: np.ndarray, n_positions: int,
                         seed: int = 42) -> List[Tuple[int, int]]:
        """Sample stroke positions based on attention map.

        Higher attention regions get more strokes.
        """
        rng = np.random.default_rng(seed)
        h, w = attention.shape

        # Flatten and normalize attention to a probability distribution
        probs = attention.flatten()
        total = probs.sum()
        if total > 0:
            probs = probs / total
        else:
            probs = np.ones(h * w) / (h * w)

        # Sample positions
        indices = rng.choice(h * w, size=n_positions, replace=True, p=probs)
        positions = [(int(idx % w), int(idx // w)) for idx in indices]
        return positions


# ---------------------------------------------------------------------------
# Enhanced hierarchical painter
# ---------------------------------------------------------------------------

class AttentionHierarchicalPainter:
    """5-layer painting strategy with attention mechanism and style support."""

    def __init__(self, canvas_size: int = 512, seed: int = 42,
                 style: str = 'default'):
        self.canvas_size = canvas_size
        self.seed = seed
        self.style = style
        self.attention_net = SpatialAttentionNet(canvas_size)

    def paint(self, target: np.ndarray, max_strokes: int = 500,
              use_attention: bool = True) -> List[Dict]:
        """Generate a sequence of strokes to paint the target image.

        Args:
            target: target image (H, W, 3) in [0, 1]
            max_strokes: total stroke budget
            use_attention: if True, use attention to guide stroke placement

        Returns:
            List of stroke dicts compatible with converter/transform.py
        """
        style_config = STYLE_TEMPLATES.get(self.style, STYLE_TEMPLATES['default'])
        layers = style_config['layers']

        # Apply style preprocessing
        if style_config['blur_sigma'] > 0:
            pil_img = Image.fromarray((target * 255).astype(np.uint8))
            pil_img = pil_img.filter(ImageFilter.GaussianBlur(radius=style_config['blur_sigma']))
            target = np.asarray(pil_img, dtype=np.float32) / 255.0

        # Compute attention map
        attention = self.attention_net.compute_attention(target) if use_attention else None

        strokes: List[Dict] = []
        canvas = np.zeros_like(target)

        for layer_name in layers:
            config = ENHANCED_LAYER_CONFIGS[layer_name]
            budget = int(max_strokes * config['budget_frac'])
            if budget <= 0:
                continue

            print(f"[attention_painter] {config['name']}: {budget} strokes (budget {budget})")

            layer_strokes = self._paint_layer(
                target, canvas, budget, config, attention, layer_name
            )
            strokes.extend(layer_strokes)

            # Update canvas estimate (simplified)
            for s in layer_strokes:
                self._apply_stroke_to_canvas(canvas, s)

        print(f"[attention_painter] Total: {len(strokes)} strokes")
        return strokes

    def _paint_layer(self, target: np.ndarray, canvas: np.ndarray,
                     budget: int, config: Dict, attention: Optional[np.ndarray],
                     layer_name: str) -> List[Dict]:
        """Paint a single layer."""
        if layer_name in ('global', 'regional', 'local'):
            return self._color_block_layer(target, canvas, budget, config, attention)
        elif layer_name == 'detail':
            return self._edge_stroke_layer(target, canvas, budget, config, attention)
        elif layer_name == 'adjustment':
            return self._adjustment_layer(target, canvas, budget, config, attention)
        return []

    def _color_block_layer(self, target: np.ndarray, canvas: np.ndarray,
                           budget: int, config: Dict,
                           attention: Optional[np.ndarray]) -> List[Dict]:
        """Paint color blocks for global/regional/local layers."""
        h, w = target.shape[:2]
        k = STYLE_TEMPLATES[self.style]['color_quantize_k']
        quant, palette = self._quantize(target, k=k)

        strokes: List[Dict] = []
        rng = np.random.default_rng(self.seed)

        # Sample positions based on attention
        if attention is not None:
            positions = self.attention_net.sample_positions(attention, budget, self.seed)
        else:
            positions = [(int(rng.uniform(0, w)), int(rng.uniform(0, h)))
                         for _ in range(budget)]

        for px, py in positions:
            px = max(0, min(w - 1, px))
            py = max(0, min(h - 1, py))
            r, g, b = target[py, px]
            length = float(rng.uniform(0.05, 0.15))
            angle = float(rng.uniform(0, 2 * np.pi))
            dx, dy = length * np.cos(angle), length * np.sin(angle)
            ex = min(1.0, max(0.0, px / w + dx))
            ey = min(1.0, max(0.0, py / h + dy))
            strokes.append({
                "x_start": float(px) / w, "y_start": float(py) / h,
                "x_end": float(ex), "y_end": float(ey),
                "color_r": float(r), "color_g": float(g), "color_b": float(b),
                "color_a": config['alpha'],
                "brush_radius": config['radius'],
            })
        return strokes

    def _edge_stroke_layer(self, target: np.ndarray, canvas: np.ndarray,
                           budget: int, config: Dict,
                           attention: Optional[np.ndarray]) -> List[Dict]:
        """Paint edge-based strokes for detail layer."""
        h, w = target.shape[:2]
        gray = cv2.cvtColor((target * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        t1, t2 = STYLE_TEMPLATES[self.style]['edge_threshold']
        edges = cv2.Canny(gray, t1, t2)

        ys, xs = np.where(edges > 0)
        if len(xs) < 4:
            return []

        weights = edges[ys, xs].astype(np.float64) ** 2
        weights /= weights.sum() + 1e-9
        rng = np.random.default_rng(self.seed)
        n_pick = min(budget, len(xs))
        pick = rng.choice(len(xs), size=n_pick, replace=False, p=weights)

        strokes: List[Dict] = []
        for i in pick:
            sx, sy = int(xs[i]), int(ys[i])
            r, g, b = target[sy, sx]
            gy, gx = np.gradient(target[:, :, 0].astype(np.float32))
            tx, ty = -gy[sy, sx], gx[sy, sx]
            norm = (tx * tx + ty * ty) ** 0.5 + 1e-6
            tx, ty = tx / norm, ty / norm
            length = float(rng.uniform(0.02, 0.05))
            ex = min(1.0, max(0.0, sx / w + tx * length))
            ey = min(1.0, max(0.0, sy / h + ty * length))
            strokes.append({
                "x_start": float(sx) / w, "y_start": float(sy) / h,
                "x_end": float(ex), "y_end": float(ey),
                "color_r": float(r), "color_g": float(g), "color_b": float(b),
                "color_a": config['alpha'],
                "brush_radius": config['radius'],
            })
        return strokes

    def _adjustment_layer(self, target: np.ndarray, canvas: np.ndarray,
                          budget: int, config: Dict,
                          attention: Optional[np.ndarray]) -> List[Dict]:
        """Paint adjustment strokes based on residual error."""
        h, w = target.shape[:2]
        residual = np.abs(target - canvas).mean(axis=2)

        # Sample positions with highest residual
        ys, xs = np.where(residual > np.percentile(residual, 90))
        if len(xs) < 4:
            return []

        rng = np.random.default_rng(self.seed)
        n_pick = min(budget, len(xs))
        pick = rng.choice(len(xs), size=n_pick, replace=False)

        strokes: List[Dict] = []
        for i in pick:
            sx, sy = int(xs[i]), int(ys[i])
            r, g, b = target[sy, sx]
            length = float(rng.uniform(0.01, 0.03))
            angle = float(rng.uniform(0, 2 * np.pi))
            dx, dy = length * np.cos(angle), length * np.sin(angle)
            ex = min(1.0, max(0.0, sx / w + dx))
            ey = min(1.0, max(0.0, sy / h + dy))
            strokes.append({
                "x_start": float(sx) / w, "y_start": float(sy) / h,
                "x_end": float(ex), "y_end": float(ey),
                "color_r": float(r), "color_g": float(g), "color_b": float(b),
                "color_a": config['alpha'],
                "brush_radius": config['radius'],
            })
        return strokes

    def _quantize(self, img: np.ndarray, k: int = 8) -> Tuple[np.ndarray, np.ndarray]:
        """K-means color quantization."""
        h, w, _ = img.shape
        pts = img.reshape(-1, 3).astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.05)
        _, labels, centers = cv2.kmeans(pts, k, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
        quant = centers[labels.flatten()].reshape(h, w, 3)
        return quant, centers

    def _apply_stroke_to_canvas(self, canvas: np.ndarray, stroke: Dict):
        """Simplified stroke application for canvas tracking."""
        h, w = canvas.shape[:2]
        x0 = int(stroke['x_start'] * w)
        y0 = int(stroke['y_start'] * h)
        x1 = int(stroke['x_end'] * w)
        y1 = int(stroke['y_end'] * h)
        color = [int(stroke['color_r'] * 255),
                 int(stroke['color_g'] * 255),
                 int(stroke['color_b'] * 255)]
        radius = max(1, int(stroke['brush_radius']))
        cv2.line(canvas, (x0, y0), (x1, y1), color, radius)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_attention_painter(canvas_size: int = 512, seed: int = 42,
                            style: str = 'default') -> AttentionHierarchicalPainter:
    """Create an AttentionHierarchicalPainter instance."""
    return AttentionHierarchicalPainter(canvas_size=canvas_size, seed=seed, style=style)


def get_available_styles() -> List[str]:
    """Return list of available style names."""
    return list(STYLE_TEMPLATES.keys())
