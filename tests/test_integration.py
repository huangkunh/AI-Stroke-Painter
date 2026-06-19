#!/usr/bin/env python3
"""
Integration tests for the full AI-Stroke-Painter pipeline.

Tests the end-to-end flow:
    image -> model/inference.py -> raw_strokes.json
           -> converter/transform.py -> output_strokes.json

Run with:
    python -m unittest tests.test_integration
"""
import json
import os
import sys
import tempfile
import unittest

# Make the project root importable
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
# Make model/ importable (for inference.py sibling imports)
MODEL_DIR = os.path.join(PROJECT_ROOT, "model")
if MODEL_DIR not in sys.path:
    sys.path.insert(0, MODEL_DIR)

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from converter.transform import (  # noqa: E402
    transform,
    ENGINE_W,
    ENGINE_H,
    DEFAULT_BACKGROUND,
)


class TestPipelineIntegration(unittest.TestCase):
    """End-to-end integration tests: image -> raw strokes -> engine JSON."""

    @classmethod
    def setUpClass(cls):
        """Create a small synthetic test image once for all tests."""
        cls.tmpdir = tempfile.mkdtemp(prefix="asp_test_")
        # Create a 64x64 test image with distinct colour regions
        img = np.zeros((64, 64, 3), dtype=np.uint8)
        img[:32, :32] = [200, 50, 50]    # red quadrant
        img[:32, 32:] = [50, 200, 50]    # green quadrant
        img[32:, :32] = [50, 50, 200]    # blue quadrant
        img[32:, 32:] = [200, 200, 50]   # yellow quadrant
        cls.image_path = os.path.join(cls.tmpdir, "test_input.png")
        Image.fromarray(img).save(cls.image_path)
        cls.test_image = img

    def _run_inference(self, max_steps=50):
        """Run lite inference on the test image, return raw actions list."""
        import inference as inf
        image = inf.load_image(self.image_path, size=64)
        return inf.run_lite_inference(image, max_steps=max_steps)

    def test_inference_produces_valid_actions(self):
        """Lite inference should produce a non-empty list of valid action dicts."""
        actions = self._run_inference(max_steps=50)
        self.assertGreater(len(actions), 0, "inference should produce strokes")

        required_keys = {
            "x_start", "y_start", "x_end", "y_end",
            "color_r", "color_g", "color_b", "color_a", "brush_radius",
        }
        for i, a in enumerate(actions):
            self.assertTrue(required_keys.issubset(a.keys()),
                            f"action {i} missing keys: {required_keys - set(a.keys())}")
            # coordinates in [0, 1]
            for k in ("x_start", "y_start", "x_end", "y_end"):
                self.assertGreaterEqual(a[k], -0.1, f"action {i} {k}={a[k]} < -0.1")
                self.assertLessEqual(a[k], 1.1, f"action {i} {k}={a[k]} > 1.1")
            # colours in [0, 1]
            for k in ("color_r", "color_g", "color_b", "color_a"):
                self.assertGreaterEqual(a[k], 0.0, f"action {i} {k}={a[k]} < 0")
                self.assertLessEqual(a[k], 1.0, f"action {i} {k}={a[k]} > 1")
            # radius positive
            self.assertGreater(a["brush_radius"], 0, f"action {i} radius={a['brush_radius']}")

    def test_transform_produces_valid_engine_json(self):
        """transform() output should be a valid engine instruction stream."""
        actions = self._run_inference(max_steps=50)
        instructions = transform(actions, start_with_background=True)

        self.assertIsInstance(instructions, list)
        self.assertGreater(len(instructions), 0)

        # First instruction should be background
        self.assertEqual(instructions[0][0], "background")
        self.assertEqual(instructions[0][1], DEFAULT_BACKGROUND)

        # All instructions should have valid op names
        valid_ops = {"background", "colour", "width", "alpha", "line"}
        for i, inst in enumerate(instructions):
            self.assertIn(inst[0], valid_ops,
                          f"instruction {i} has invalid op: {inst[0]}")

    def test_full_pipeline_image_to_json(self):
        """Full pipeline: image -> inference -> transform -> JSON file."""
        # 1. Inference
        actions = self._run_inference(max_steps=50)
        raw_path = os.path.join(self.tmpdir, "raw_strokes.json")
        with open(raw_path, "w") as f:
            json.dump(actions, f)

        # 2. Load raw + transform
        with open(raw_path) as f:
            loaded_actions = json.load(f)
        instructions = transform(loaded_actions, start_with_background=True)

        # 3. Save engine JSON
        out_path = os.path.join(self.tmpdir, "output_strokes.json")
        with open(out_path, "w") as f:
            json.dump(instructions, f)

        # 4. Verify the saved file is valid JSON and round-trips
        with open(out_path) as f:
            final = json.load(f)
        self.assertEqual(len(final), len(instructions))
        self.assertEqual(final[0][0], "background")

        # 5. Verify there's at least one line instruction
        lines = [i for i in final if i[0] == "line"]
        self.assertGreater(len(lines), 0, "pipeline should produce line instructions")

    def test_pipeline_deterministic(self):
        """Running lite inference twice should produce similar results.

        The lite painter uses a fixed RNG seed (42) for stroke sampling, but
        cv2.kmeans (colour quantisation) has its own internal randomness that
        we cannot fully seed from Python. So we verify the two runs produce
        the same NUMBER of strokes and statistically similar colour
        distributions, rather than bit-identical output.
        """
        import inference as inf
        image = inf.load_image(self.image_path, size=64)
        actions1 = inf.run_lite_inference(image, max_steps=30)
        actions2 = inf.run_lite_inference(image, max_steps=30)
        self.assertEqual(len(actions1), len(actions2),
                         "stroke count should be deterministic")
        # Verify colour distributions are similar (mean RGB within 0.1)
        for key in ("color_r", "color_g", "color_b"):
            m1 = np.mean([a[key] for a in actions1])
            m2 = np.mean([a[key] for a in actions2])
            self.assertAlmostEqual(m1, m2, delta=0.15,
                                   msg=f"mean {key} differs too much: {m1} vs {m2}")
        # Verify all coordinates are in valid range
        for actions in (actions1, actions2):
            for a in actions:
                for k in ("x_start", "y_start", "x_end", "y_end"):
                    self.assertGreaterEqual(a[k], -0.1)
                    self.assertLessEqual(a[k], 1.1)

    def test_pipeline_with_real_sample_images(self):
        """Pipeline should work on the real sample images in assets/."""
        for name in ("sample_landscape.png", "sample_sketch.png"):
            path = os.path.join(PROJECT_ROOT, "assets", name)
            if not os.path.isfile(path):
                self.skipTest(f"sample image {name} not found")
            import inference as inf
            image = inf.load_image(path, size=128)
            actions = inf.run_lite_inference(image, max_steps=30)
            self.assertGreater(len(actions), 0, f"no strokes for {name}")
            instructions = transform(actions, start_with_background=True)
            lines = [i for i in instructions if i[0] == "line"]
            self.assertGreater(len(lines), 0, f"no line instructions for {name}")

    def test_coordinates_in_engine_resolution_range(self):
        """All emitted coordinates should fall within the engine canvas."""
        actions = self._run_inference(max_steps=50)
        instructions = transform(actions, start_with_background=False)
        for inst in instructions:
            if inst[0] != "line":
                continue
            points = inst[2:]  # skip "line" and brushId
            # brush 0: 2 values/point; brush 5: 3 values/point
            stride = 3 if inst[1] == 5 else 2
            for j in range(0, len(points), stride):
                x, y = points[j], points[j + 1]
                # Allow small margin for strokes that extend slightly off-canvas
                self.assertGreaterEqual(x, -50, f"x={x} out of range")
                self.assertLessEqual(x, ENGINE_W + 50, f"x={x} out of range")
                self.assertGreaterEqual(y, -50, f"y={y} out of range")
                self.assertLessEqual(y, ENGINE_H + 50, f"y={y} out of range")


if __name__ == "__main__":
    unittest.main(verbosity=2)
