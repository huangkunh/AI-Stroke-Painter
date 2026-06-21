#!/usr/bin/env python3
"""
Hierarchical painting strategy for stroke-based image rendering.

This module implements a multi-layer painting approach that mimics how a
human artist paints:
  1. Coarse layer: large color blocks to quickly cover the canvas
  2. Medium layer: medium strokes to add main structure and shapes
  3. Fine layer: thin strokes for details and texture
  4. Adjustment layer: final corrections to optimize overall appearance

Each layer uses different stroke parameters (radius, alpha, pressure) and
targets different frequency bands of the target image. The hierarchical
approach produces more natural-looking paintings than single-pass methods.

The output is compatible with the existing converter/transform.py pipeline:
each stroke is a dict with keys (x_start, y_start, x_end, y_end,
color_r, color_g, color_b, color_a, brush_radius).
"""
from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False
    torch = None


# ---------------------------------------------------------------------------
# Layer configurations
# ---------------------------------------------------------------------------

LAYER_CONFIGS = {
    'coarse': {
        'name': '粗略层',
        'radius': 16.0,
        'alpha': 0.5,
        'pressure_start': 0.3,
        'pressure_end': 0.7,
        'stroke_budget_frac': 0.30,   # 30% of total strokes
        'blur_kernel': 15,             # low-frequency target
        'description': '大色块铺底，快速覆盖画布',
    },
    'medium': {
        'name': '中等层',
        'radius': 8.0,
        'alpha': 0.7,
        'pressure_start': 0.4,
        'pressure_end': 0.8,
        'stroke_budget_frac': 0.35,   # 35% of total strokes
        'blur_kernel': 7,              # mid-frequency target
        'description': '中等细节，添加主要结构',
    },
    'fine': {
        'name': '精细层',
        'radius': 3.0,
        'alpha': 0.85,
        'pressure_start': 0.5,
        'pressure_end': 0.9,
        'stroke_budget_frac': 0.25,   # 25% of total strokes
        'blur_kernel': 3,              # high-frequency target
        'description': '细节勾勒，添加纹理和细节',
    },
    'adjustment': {
        'name': '调整层',
        'radius': 2.0,
        'alpha': 0.9,
        'pressure_start': 0.6,
        'pressure_end': 1.0,
        'stroke_budget_frac': 0.10,   # 10% of total strokes
        'blur_kernel': 0,              # full resolution
        'description': '最终调整，优化整体效果',
    },
}


# ---------------------------------------------------------------------------
# Hierarchical Painter
# ---------------------------------------------------------------------------

class HierarchicalPainter:
    """Multi-layer painting strategy.

    This is a heuristic painter (no neural network required) that produces
    human-like stroke sequences by painting in layers from coarse to fine.
    It can optionally use a trained DDPG agent for the fine/adjustment
    layers when PyTorch and trained weights are available.

    The output format is fully compatible with converter/transform.py.
    """

    def __init__(self, canvas_size: int = 512, seed: int = 42):
        self.canvas_size = canvas_size
        self.rng = np.random.default_rng(seed)

    def paint(self, image: np.ndarray, max_strokes: int = 500,
              use_neural: bool = False) -> List[Dict[str, float]]:
        """Generate a hierarchical stroke sequence for the target image.

        Args:
            image: H x W x 3 float32 in [0, 1]
            max_strokes: total stroke budget across all layers
            use_neural: if True, use DDPG agent for fine/adjustment layers
                        (requires PyTorch + trained weights)

        Returns:
            List of stroke action dicts, compatible with transform.py
        """
        h, w = image.shape[:2]
        all_strokes: List[Dict[str, float]] = []

        for layer_name in ['coarse', 'medium', 'fine', 'adjustment']:
            config = LAYER_CONFIGS[layer_name]
            budget = max(1, int(max_strokes * config['stroke_budget_frac']))

            # Generate the frequency-targeted version of the image
            if config['blur_kernel'] > 0:
                target = self._blur_image(image, config['blur_kernel'])
            else:
                target = image

            # Generate strokes for this layer
            if layer_name in ('fine', 'adjustment') and use_neural and _HAS_TORCH:
                strokes = self._neural_layer(
                    target, budget, config, layer_name
                )
            else:
                strokes = self._heuristic_layer(
                    target, budget, config, layer_name
                )

            all_strokes.extend(strokes)
            print(f"[hierarchical] {config['name']}: {len(strokes)} strokes "
                  f"(budget {budget})")

        return all_strokes

    def _blur_image(self, image: np.ndarray, kernel: int) -> np.ndarray:
        """Apply Gaussian blur to extract a specific frequency band."""
        if kernel <= 1:
            return image
        img_uint8 = (image * 255).astype(np.uint8)
        blurred = cv2.GaussianBlur(img_uint8, (kernel, kernel), 0)
        return blurred.astype(np.float32) / 255.0

    def _heuristic_layer(self, target: np.ndarray, budget: int,
                         config: Dict, layer_name: str) -> List[Dict]:
        """Generate strokes for a layer using heuristic sampling.

        Strategy:
          - Coarse: sample from large color regions (k-means quantization)
          - Medium: sample along mid-frequency edges
          - Fine: sample along high-frequency edges
          - Adjustment: sample from largest residual errors
        """
        h, w = target.shape[:2]
        strokes: List[Dict] = []

        if layer_name == 'coarse':
            # Color-block: k-means quantize, sample from each region
            strokes = self._color_block_strokes(
                target, budget, config['radius'], config['alpha']
            )
        elif layer_name == 'medium':
            # Mid-frequency edges
            gray = cv2.cvtColor((target * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 50, 120)
            strokes = self._edge_strokes(
                target, edges, budget, config['radius'], config['alpha']
            )
        elif layer_name == 'fine':
            # High-frequency edges
            gray = cv2.cvtColor((target * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 100, 200)
            strokes = self._edge_strokes(
                target, edges, budget, config['radius'], config['alpha']
            )
        elif layer_name == 'adjustment':
            # Sample from residual (difference between target and a blurred version)
            blurred = self._blur_image(target, 5)
            residual = np.abs(target - blurred).mean(axis=2)
            strokes = self._residual_strokes(
                target, residual, budget, config['radius'], config['alpha']
            )

        return strokes

    def _color_block_strokes(self, target: np.ndarray, budget: int,
                             radius: float, alpha: float) -> List[Dict]:
        """Sample large color-block strokes from quantized regions."""
        h, w = target.shape[:2]
        # K-means quantization
        pts = target.reshape(-1, 3).astype(np.float32)
        k = min(6, budget)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.05)
        _, labels, centers = cv2.kmeans(pts, k, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
        quant = centers[labels.flatten()].reshape(h, w, 3)

        strokes: List[Dict] = []
        per_region = max(1, budget // k)
        for ci in range(k):
            mask = (labels.flatten() == ci).reshape(h, w).astype(np.uint8) * 255
            if mask.sum() < 100:
                continue
            # Clean up
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
            # Distance transform for sampling
            dist = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
            probs = dist.flatten().astype(np.float64) ** 2
            total = probs.sum()
            if total < 1e-9:
                continue
            probs /= total
            n_nonzero = int((probs > 0).sum())
            n_pick = min(per_region, n_nonzero)
            if n_pick <= 0:
                continue
            idxs = self.rng.choice(len(probs), size=n_pick, replace=False, p=probs)
            color = centers[ci]
            for idx in idxs:
                sy, sx = divmod(int(idx), w)
                length = float(self.rng.uniform(0.05, 0.12))
                angle = float(self.rng.uniform(0, 2 * np.pi))
                dx, dy = length * np.cos(angle), length * np.sin(angle)
                strokes.append({
                    'x_start': float(sx / w),
                    'y_start': float(sy / h),
                    'x_end': float(sx / w + dx),
                    'y_end': float(sy / h + dy),
                    'color_r': float(color[0]),
                    'color_g': float(color[1]),
                    'color_b': float(color[2]),
                    'color_a': float(alpha),
                    'brush_radius': float(radius),
                })
        return strokes

    def _edge_strokes(self, target: np.ndarray, edges: np.ndarray,
                      budget: int, radius: float, alpha: float) -> List[Dict]:
        """Sample strokes along edges."""
        h, w = target.shape[:2]
        ys, xs = np.where(edges > 0)
        if len(xs) < 4:
            return []
        weights = edges[ys, xs].astype(np.float64) ** 2
        total = weights.sum()
        if total < 1e-9:
            weights = np.ones(len(weights)) / len(weights)
        else:
            weights /= total
        n_pick = min(budget, len(xs))
        pick = self.rng.choice(len(xs), size=n_pick, replace=False, p=weights)
        strokes: List[Dict] = []
        for i in pick:
            sx, sy = int(xs[i]), int(ys[i])
            r, g, b = target[sy, sx]
            # Walk along edge tangent
            gray = cv2.cvtColor((target * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
            gy, gx = np.gradient(gray.astype(np.float32))
            tx, ty = -gy[sy, sx], gx[sy, sx]
            norm = (tx * tx + ty * ty) ** 0.5 + 1e-6
            tx, ty = tx / norm, ty / norm
            length = float(self.rng.uniform(0.02, 0.05))
            strokes.append({
                'x_start': float(sx / w),
                'y_start': float(sy / h),
                'x_end': float(sx / w + tx * length),
                'y_end': float(sy / h + ty * length),
                'color_r': float(r),
                'color_g': float(g),
                'color_b': float(b),
                'color_a': float(alpha),
                'brush_radius': float(radius),
            })
        return strokes

    def _residual_strokes(self, target: np.ndarray, residual: np.ndarray,
                          budget: int, radius: float, alpha: float) -> List[Dict]:
        """Sample strokes from high-residual regions."""
        h, w = target.shape[:2]
        probs = residual.flatten().astype(np.float64) ** 2
        total = probs.sum()
        if total < 1e-9:
            return []
        probs /= total
        n_nonzero = int((probs > 0).sum())
        n_pick = min(budget, n_nonzero)
        if n_pick <= 0:
            return []
        idxs = self.rng.choice(len(probs), size=n_pick, replace=False, p=probs)
        strokes: List[Dict] = []
        for idx in idxs:
            sy, sx = divmod(int(idx), w)
            r, g, b = target[sy, sx]
            length = float(self.rng.uniform(0.01, 0.03))
            angle = float(self.rng.uniform(0, 2 * np.pi))
            dx, dy = length * np.cos(angle), length * np.sin(angle)
            strokes.append({
                'x_start': float(sx / w),
                'y_start': float(sy / h),
                'x_end': float(sx / w + dx),
                'y_end': float(sy / h + dy),
                'color_r': float(r),
                'color_g': float(g),
                'color_b': float(b),
                'color_a': float(alpha),
                'brush_radius': float(radius),
            })
        return strokes

    def _neural_layer(self, target: np.ndarray, budget: int,
                      config: Dict, layer_name: str) -> List[Dict]:
        """Use a trained DDPG agent to generate strokes for this layer.

        Falls back to heuristic if the agent or weights are unavailable.
        """
        try:
            from ddpg_agent import build_ddpg_agent
            from differentiable_renderer import build_differentiable_renderer

            # Use small canvas for neural inference (speed)
            neural_size = 64
            target_small = cv2.resize(
                (target * 255).astype(np.uint8),
                (neural_size, neural_size),
                interpolation=cv2.INTER_AREA
            ).astype(np.float32) / 255.0
            target_tensor = torch.from_numpy(target_small).permute(2, 0, 1).unsqueeze(0)

            agent = build_ddpg_agent(canvas_size=neural_size, device='cpu')
            renderer = build_differentiable_renderer(
                canvas_size=neural_size, num_samples=8, max_radius=config['radius']
            )

            # Try to load weights
            weights_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                'model', 'pretrained', 'hierarchical_agent.pth'
            )
            if os.path.isfile(weights_path):
                agent.load(weights_path)
            else:
                # No weights -> fall back to heuristic
                return self._heuristic_layer(target, budget, config, layer_name)

            canvas = torch.zeros_like(target_tensor)
            h, c = agent.actor.init_hidden(1, torch.device('cpu'))
            strokes: List[Dict] = []
            from differentiable_renderer import decode_stroke_params

            for _ in range(budget):
                action, (h, c) = agent.select_action(
                    target_tensor, canvas, (h, c), noise=False
                )
                canvas = renderer(canvas, action)
                params = decode_stroke_params(action[0], max_radius=config['radius'])
                strokes.append({
                    'x_start': params['x_start'],
                    'y_start': params['y_start'],
                    'x_end': params['x_end'],
                    'y_end': params['y_end'],
                    'color_r': params['color_r'],
                    'color_g': params['color_g'],
                    'color_b': params['color_b'],
                    'color_a': params['color_a'],
                    'brush_radius': params['brush_radius'],
                })
            return strokes

        except Exception as e:
            print(f"[hierarchical] neural layer failed ({e}), using heuristic")
            return self._heuristic_layer(target, budget, config, layer_name)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_hierarchical_painter(canvas_size: int = 512,
                               seed: int = 42) -> HierarchicalPainter:
    """Create a HierarchicalPainter instance."""
    return HierarchicalPainter(canvas_size=canvas_size, seed=seed)


def run_hierarchical_inference(image_path: str, max_strokes: int = 500,
                               use_neural: bool = False,
                               out_path: str = 'raw_strokes.json') -> List[Dict]:
    """Convenience function: image path -> stroke JSON.

    This produces output compatible with converter/transform.py.
    """
    img = Image.open(image_path).convert('RGB')
    img = img.resize((512, 512), Image.LANCZOS)
    arr = np.asarray(img, dtype=np.float32) / 255.0

    painter = build_hierarchical_painter(canvas_size=512)
    strokes = painter.paint(arr, max_strokes=max_strokes, use_neural=use_neural)

    import json
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(strokes, f, ensure_ascii=False)
    print(f"[hierarchical] wrote {len(strokes)} strokes -> {out_path}")
    return strokes
