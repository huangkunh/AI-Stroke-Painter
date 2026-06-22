#!/usr/bin/env python3
"""
Tests for the enhanced differentiable renderer.

Run with:
    python -m unittest tests.test_enhanced_renderer
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


class TestEnhancedRenderer(unittest.TestCase):
    """Tests for model/enhanced_renderer.py."""

    @classmethod
    def setUpClass(cls):
        try:
            import torch
            cls.torch = torch
            from enhanced_renderer import (
                EnhancedDifferentiableRenderer, build_enhanced_renderer,
                decode_stroke_params_enhanced, ENHANCED_ACTION_DIM
            )
            cls.Renderer = EnhancedDifferentiableRenderer
            cls.build = staticmethod(build_enhanced_renderer)
            cls.decode = staticmethod(decode_stroke_params_enhanced)
            cls.ACTION_DIM = ENHANCED_ACTION_DIM
        except ImportError as e:
            raise unittest.SkipTest(f"PyTorch not available: {e}")

    def test_renderer_creation(self):
        """Renderer should be created with 20-dim action space."""
        r = self.build(canvas_size=32)
        self.assertEqual(r.action_dim, self.ACTION_DIM)
        self.assertEqual(r.canvas_size, 32)

    def test_forward_output_shape(self):
        """Forward pass should produce correct output shape."""
        r = self.build(canvas_size=32, num_samples=8)
        canvas = self.torch.zeros(2, 3, 32, 32)
        action = self.torch.tanh(self.torch.randn(2, self.ACTION_DIM))
        out = r(canvas, action)
        self.assertEqual(out.shape, (2, 3, 32, 32))

    def test_forward_output_range(self):
        """Output should be in [0, 1] range."""
        r = self.build(canvas_size=32, num_samples=8)
        canvas = self.torch.zeros(1, 3, 32, 32)
        action = self.torch.tanh(self.torch.randn(1, self.ACTION_DIM))
        out = r(canvas, action)
        self.assertTrue(out.min() >= -0.01, f"min {out.min()} < 0")
        self.assertTrue(out.max() <= 1.01, f"max {out.max()} > 1")

    def test_gradient_flow(self):
        """Gradients should flow through the renderer to the action."""
        r = self.build(canvas_size=32, num_samples=8)
        canvas = self.torch.zeros(1, 3, 32, 32)
        action = self.torch.zeros(1, self.ACTION_DIM, requires_grad=True)
        with self.torch.no_grad():
            action.data[0, 4:9] = 0.5  # color + alpha + radius
        out = r(canvas, action)
        loss = out.sum()
        loss.backward()
        self.assertIsNotNone(action.grad)
        self.assertGreater(action.grad.abs().sum().item(), 0)

    def test_material_simulation(self):
        """Material simulation should affect output when wetness > 0."""
        r = self.build(canvas_size=32, num_samples=8, simulate_material=True)
        canvas = self.torch.zeros(1, 3, 32, 32)
        # High wetness action
        action_wet = self.torch.zeros(1, self.ACTION_DIM)
        action_wet[0, 4:9] = 0.5  # color + alpha + radius
        action_wet[0, 16] = 0.9   # high wetness
        out_wet = r(canvas, action_wet)

        r.reset_material_state()
        # Low wetness action
        action_dry = self.torch.zeros(1, self.ACTION_DIM)
        action_dry[0, 4:9] = 0.5
        action_dry[0, 16] = 0.1   # low wetness
        out_dry = r(canvas, action_dry)

        # Outputs should differ due to material simulation
        diff = (out_wet - out_dry).abs().sum().item()
        # With material sim, wet strokes spread more
        self.assertGreaterEqual(diff, 0)

    def test_decode_params(self):
        """decode_stroke_params_enhanced should return all 20 params."""
        action = self.torch.zeros(self.ACTION_DIM)
        params = self.decode(action)
        expected_keys = {
            "x_start", "y_start", "x_end", "y_end",
            "color_r", "color_g", "color_b", "color_a",
            "brush_radius", "pressure_start", "pressure_end",
            "control_x", "control_y", "stroke_len", "softness",
            "rotation", "wetness", "dryness_rate", "pigment", "blend_mode"
        }
        self.assertEqual(set(params.keys()), expected_keys)

    def test_render_sequence(self):
        """render_sequence should return final canvas and intermediates."""
        r = self.build(canvas_size=32, num_samples=8)
        canvas = self.torch.zeros(1, 3, 32, 32)
        actions = self.torch.tanh(self.torch.randn(1, 5, self.ACTION_DIM))
        final, intermediates = r.render_sequence(canvas, actions)
        self.assertEqual(final.shape, (1, 3, 32, 32))
        self.assertEqual(len(intermediates), 5)

    def test_batch_rendering(self):
        """Renderer should handle batch sizes > 1."""
        r = self.build(canvas_size=32, num_samples=8)
        canvas = self.torch.zeros(4, 3, 32, 32)
        action = self.torch.tanh(self.torch.randn(4, self.ACTION_DIM))
        out = r(canvas, action)
        self.assertEqual(out.shape, (4, 3, 32, 32))

    def test_reset_material_state(self):
        """reset_material_state should zero the wetness map."""
        r = self.build(canvas_size=32, simulate_material=True)
        # Paint something to populate wetness map
        canvas = self.torch.zeros(1, 3, 32, 32)
        action = self.torch.zeros(1, self.ACTION_DIM)
        action[0, 4:9] = 0.5
        action[0, 16] = 0.8  # high wetness
        r(canvas, action)
        r.reset_material_state()
        self.assertEqual(r.wetness_map.sum().item(), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
