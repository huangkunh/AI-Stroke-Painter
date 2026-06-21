#!/usr/bin/env python3
"""
Tests for the differentiable neural renderer.

Run with:
    python -m unittest tests.test_renderer
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


class TestDifferentiableRenderer(unittest.TestCase):
    """Tests for model/differentiable_renderer.py."""

    @classmethod
    def setUpClass(cls):
        try:
            import torch
            cls.torch = torch
            from differentiable_renderer import (
                DifferentiableRenderer, build_differentiable_renderer,
                decode_stroke_params, DEFAULT_ACTION_DIM, DEFAULT_CANVAS_SIZE
            )
            cls.DifferentiableRenderer = DifferentiableRenderer
            cls.build_renderer = staticmethod(build_differentiable_renderer)
            cls.decode_stroke_params = staticmethod(decode_stroke_params)
            cls.DEFAULT_ACTION_DIM = DEFAULT_ACTION_DIM
        except ImportError as e:
            raise unittest.SkipTest(f"PyTorch not available: {e}")

    def test_renderer_creation(self):
        """Renderer should be created with default params."""
        renderer = self.build_renderer(canvas_size=32)
        self.assertEqual(renderer.canvas_size, 32)
        self.assertEqual(renderer.action_dim, self.DEFAULT_ACTION_DIM)

    def test_forward_output_shape(self):
        """Forward pass should produce correct output shape."""
        renderer = self.build_renderer(canvas_size=32, num_samples=8)
        canvas = self.torch.zeros(2, 3, 32, 32)
        action = self.torch.randn(2, self.DEFAULT_ACTION_DIM)
        out = renderer(canvas, action)
        self.assertEqual(out.shape, (2, 3, 32, 32))

    def test_forward_output_range(self):
        """Output should be in [0, 1] range."""
        renderer = self.build_renderer(canvas_size=32, num_samples=8)
        canvas = self.torch.zeros(1, 3, 32, 32)
        action = self.torch.tanh(self.torch.randn(1, self.DEFAULT_ACTION_DIM))  # clamp to [-1, 1]
        out = renderer(canvas, action)
        self.assertTrue(out.min() >= -0.01, f"min {out.min()} < 0")
        self.assertTrue(out.max() <= 1.01, f"max {out.max()} > 1")

    def test_gradient_flow(self):
        """Gradients should flow through the renderer to the action."""
        renderer = self.build_renderer(canvas_size=32, num_samples=8)
        canvas = self.torch.zeros(1, 3, 32, 32)
        action = self.torch.randn(1, self.DEFAULT_ACTION_DIM, requires_grad=True)
        out = renderer(canvas, action)
        loss = out.sum()
        loss.backward()
        self.assertIsNotNone(action.grad, "No gradient on action")
        self.assertFalse(self.torch.all(action.grad == 0), "Gradient is all zeros")

    def test_render_sequence(self):
        """render_sequence should return final canvas and intermediates."""
        renderer = self.build_renderer(canvas_size=32, num_samples=8)
        canvas = self.torch.zeros(1, 3, 32, 32)
        actions = self.torch.randn(1, 5, self.DEFAULT_ACTION_DIM)  # 5 strokes
        final, intermediates = renderer.render_sequence(canvas, actions)
        self.assertEqual(final.shape, (1, 3, 32, 32))
        self.assertEqual(len(intermediates), 5)

    def test_decode_stroke_params(self):
        """decode_stroke_params should return a dict with all expected keys."""
        action = self.torch.zeros(self.DEFAULT_ACTION_DIM)
        params = self.decode_stroke_params(action)
        expected_keys = {
            "x_start", "y_start", "x_end", "y_end",
            "color_r", "color_g", "color_b", "color_a",
            "brush_radius", "pressure_start", "pressure_end",
            "control_x", "control_y", "stroke_len", "softness"
        }
        self.assertEqual(set(params.keys()), expected_keys)

    def test_batch_rendering(self):
        """Renderer should handle batch sizes > 1."""
        renderer = self.build_renderer(canvas_size=32, num_samples=8)
        canvas = self.torch.zeros(4, 3, 32, 32)
        action = self.torch.randn(4, self.DEFAULT_ACTION_DIM)
        out = renderer(canvas, action)
        self.assertEqual(out.shape, (4, 3, 32, 32))


if __name__ == "__main__":
    unittest.main(verbosity=2)
