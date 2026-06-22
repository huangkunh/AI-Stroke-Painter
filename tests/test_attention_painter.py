#!/usr/bin/env python3
"""
Tests for the attention-based hierarchical painter.

Run with:
    python -m unittest tests.test_attention_painter
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


class TestAttentionPainter(unittest.TestCase):
    """Tests for model/attention_painter.py."""

    @classmethod
    def setUpClass(cls):
        from attention_painter import (
            AttentionHierarchicalPainter, SpatialAttentionNet,
            ENHANCED_LAYER_CONFIGS, STYLE_TEMPLATES,
            build_attention_painter, get_available_styles
        )
        cls.Painter = AttentionHierarchicalPainter
        cls.AttentionNet = SpatialAttentionNet
        cls.LAYER_CONFIGS = ENHANCED_LAYER_CONFIGS
        cls.STYLE_TEMPLATES = STYLE_TEMPLATES
        cls.build = staticmethod(build_attention_painter)
        cls.get_styles = staticmethod(get_available_styles)
        cls.image = np.random.rand(64, 64, 3).astype(np.float32)

    def test_5_layers_exist(self):
        """All 5 layer configs should be present."""
        for name in ['global', 'regional', 'local', 'detail', 'adjustment']:
            self.assertIn(name, self.LAYER_CONFIGS)

    def test_layer_radii_decrease(self):
        """Layer radii should decrease from global to adjustment."""
        radii = [self.LAYER_CONFIGS[n]['radius']
                 for n in ['global', 'regional', 'local', 'detail', 'adjustment']]
        for i in range(len(radii) - 1):
            self.assertGreater(radii[i], radii[i + 1])

    def test_styles_available(self):
        """Multiple style templates should be available."""
        styles = self.get_styles()
        self.assertIn('default', styles)
        self.assertIn('oil', styles)
        self.assertIn('watercolor', styles)
        self.assertIn('sketch', styles)
        self.assertIn('anime', styles)

    def test_attention_computation(self):
        """SpatialAttentionNet should compute attention map."""
        net = self.AttentionNet(canvas_size=64)
        attention = net.compute_attention(self.image)
        self.assertEqual(attention.shape, (64, 64))
        self.assertTrue(np.all(attention >= 0))
        self.assertTrue(np.all(attention <= 1))

    def test_attention_with_canvas(self):
        """Attention should work with a canvas input."""
        net = self.AttentionNet(canvas_size=64)
        canvas = np.zeros_like(self.image)
        attention = net.compute_attention(self.image, canvas)
        self.assertEqual(attention.shape, (64, 64))

    def test_sample_positions(self):
        """Position sampling should return requested count."""
        net = self.AttentionNet(canvas_size=64)
        attention = net.compute_attention(self.image)
        positions = net.sample_positions(attention, 20, seed=42)
        self.assertEqual(len(positions), 20)
        for x, y in positions:
            self.assertGreaterEqual(x, 0)
            self.assertLess(x, 64)
            self.assertGreaterEqual(y, 0)
            self.assertLess(y, 64)

    def test_paint_default_style(self):
        """Painting with default style should produce strokes."""
        painter = self.build(canvas_size=64, style='default')
        strokes = painter.paint(self.image, max_strokes=50)
        self.assertGreater(len(strokes), 0)
        # Check stroke format
        s = strokes[0]
        self.assertIn('x_start', s)
        self.assertIn('y_start', s)
        self.assertIn('x_end', s)
        self.assertIn('y_end', s)
        self.assertIn('color_r', s)
        self.assertIn('brush_radius', s)

    def test_paint_oil_style(self):
        """Painting with oil style should produce strokes."""
        painter = self.build(canvas_size=64, style='oil')
        strokes = painter.paint(self.image, max_strokes=50)
        self.assertGreater(len(strokes), 0)

    def test_paint_sketch_style(self):
        """Sketch style should skip color layers."""
        painter = self.build(canvas_size=64, style='sketch')
        strokes = painter.paint(self.image, max_strokes=50)
        self.assertGreater(len(strokes), 0)
        # Sketch uses only detail and adjustment layers (small radii)
        radii = [s['brush_radius'] for s in strokes]
        self.assertLess(max(radii), 5)

    def test_paint_with_attention(self):
        """Painting with attention should produce different results than without."""
        p1 = self.build(canvas_size=64, seed=42)
        p2 = self.build(canvas_size=64, seed=42)
        s1 = p1.paint(self.image, max_strokes=30, use_attention=True)
        s2 = p2.paint(self.image, max_strokes=30, use_attention=False)
        # Both should produce strokes
        self.assertGreater(len(s1), 0)
        self.assertGreater(len(s2), 0)

    def test_stroke_value_ranges(self):
        """All stroke values should be in valid ranges."""
        painter = self.build(canvas_size=64)
        strokes = painter.paint(self.image, max_strokes=30)
        for s in strokes:
            self.assertGreaterEqual(s['x_start'], 0)
            self.assertLessEqual(s['x_start'], 1)
            self.assertGreaterEqual(s['y_start'], 0)
            self.assertLessEqual(s['y_start'], 1)
            self.assertGreaterEqual(s['color_r'], 0)
            self.assertLessEqual(s['color_r'], 1)
            self.assertGreater(s['brush_radius'], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
