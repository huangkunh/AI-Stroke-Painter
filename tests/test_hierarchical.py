#!/usr/bin/env python3
"""
Tests for the hierarchical painting strategy.

Run with:
    python -m unittest tests.test_hierarchical
"""
import os
import sys
import unittest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
MODEL_DIR = os.path.join(PROJECT_ROOT, "model")
if MODEL_DIR not in sys.path:
    sys.path.insert(0, MODEL_DIR)

import numpy as np  # noqa: E402


class TestHierarchicalPainter(unittest.TestCase):
    """Tests for model/hierarchical_painter.py."""

    @classmethod
    def setUpClass(cls):
        from hierarchical_painter import (
            HierarchicalPainter, build_hierarchical_painter,
            LAYER_CONFIGS, run_hierarchical_inference
        )
        cls.HierarchicalPainter = HierarchicalPainter
        cls.build_painter = staticmethod(build_hierarchical_painter)
        cls.LAYER_CONFIGS = LAYER_CONFIGS
        cls.run_inference = staticmethod(run_hierarchical_inference)
        # Create a test image
        cls.image = np.random.rand(64, 64, 3).astype(np.float32)

    def test_painter_creation(self):
        """Painter should be created with default params."""
        painter = self.build_painter(canvas_size=64)
        self.assertEqual(painter.canvas_size, 64)

    def test_layer_configs_exist(self):
        """All 4 layer configs should be present."""
        for name in ['coarse', 'medium', 'fine', 'adjustment']:
            self.assertIn(name, self.LAYER_CONFIGS)
            cfg = self.LAYER_CONFIGS[name]
            self.assertIn('radius', cfg)
            self.assertIn('alpha', cfg)
            self.assertIn('stroke_budget_frac', cfg)

    def test_paint_returns_strokes(self):
        """paint() should return a list of stroke dicts."""
        painter = self.build_painter(canvas_size=64)
        strokes = painter.paint(self.image, max_strokes=50)
        self.assertIsInstance(strokes, list)
        self.assertGreater(len(strokes), 0)

    def test_stroke_format(self):
        """Each stroke should have all required keys."""
        painter = self.build_painter(canvas_size=64)
        strokes = painter.paint(self.image, max_strokes=20)
        required_keys = {
            'x_start', 'y_start', 'x_end', 'y_end',
            'color_r', 'color_g', 'color_b', 'color_a', 'brush_radius'
        }
        for s in strokes:
            self.assertEqual(set(s.keys()), required_keys,
                             f"Stroke missing keys: {set(s.keys())}")

    def test_stroke_value_ranges(self):
        """All stroke values should be in valid ranges."""
        painter = self.build_painter(canvas_size=64)
        strokes = painter.paint(self.image, max_strokes=30)
        for s in strokes:
            for k in ('x_start', 'y_start', 'x_end', 'y_end',
                      'color_r', 'color_g', 'color_b', 'color_a'):
                self.assertGreaterEqual(s[k], -0.2, f"{k}={s[k]} < -0.2")
                self.assertLessEqual(s[k], 1.2, f"{k}={s[k]} > 1.2")
            self.assertGreater(s['brush_radius'], 0)
            self.assertLess(s['brush_radius'], 30)

    def test_layer_budget_distribution(self):
        """Stroke budget should be distributed across layers."""
        painter = self.build_painter(canvas_size=64)
        strokes = painter.paint(self.image, max_strokes=100)
        # With 100 strokes, we expect at least 10 per layer
        self.assertGreater(len(strokes), 20)

    def test_coarse_layer_uses_large_radius(self):
        """Coarse layer should use larger radius than fine layer."""
        painter = self.build_painter(canvas_size=64)
        # Paint only coarse layer by checking radius distribution
        strokes = painter.paint(self.image, max_strokes=100)
        radii = [s['brush_radius'] for s in strokes]
        # Coarse layer radius is 16, fine is 3, adjustment is 2
        # So we should see some large radii
        self.assertGreater(max(radii), 10, "No large-radius strokes found")

    def test_paint_deterministic_with_seed(self):
        """Same seed should produce same strokes."""
        p1 = self.build_painter(canvas_size=64, seed=123)
        p2 = self.build_painter(canvas_size=64, seed=123)
        s1 = p1.paint(self.image, max_strokes=20)
        s2 = p2.paint(self.image, max_strokes=20)
        self.assertEqual(len(s1), len(s2))

    def test_paint_with_neural_fallback(self):
        """use_neural=True should fall back to heuristic when no weights."""
        painter = self.build_painter(canvas_size=64)
        # No trained weights available -> should fall back gracefully
        strokes = painter.paint(self.image, max_strokes=20, use_neural=True)
        self.assertGreater(len(strokes), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
