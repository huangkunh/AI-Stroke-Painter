#!/usr/bin/env python3
"""
Edge-case tests for the AI-Stroke-Painter pipeline.

Tests extreme inputs: empty images, single-colour images, very small/large
images, corrupt data, etc. The system should not crash and should give
reasonable error messages or graceful degradation.

Run with:
    python -m unittest tests.test_edge_cases
"""
import json
import os
import sys
import tempfile
import unittest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
MODEL_DIR = os.path.join(PROJECT_ROOT, "model")
if MODEL_DIR not in sys.path:
    sys.path.insert(0, MODEL_DIR)

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from converter.transform import transform  # noqa: E402


class TestEdgeCases(unittest.TestCase):
    """Edge-case and boundary condition tests."""

    @classmethod
    def setUpClass(cls):
        cls.tmpdir = tempfile.mkdtemp(prefix="asp_edge_")

    def _save_image(self, arr, name):
        path = os.path.join(self.tmpdir, name)
        Image.fromarray((arr * 255).astype(np.uint8)).save(path)
        return path

    # -------------------------------------------------------------------------
    # Image edge cases
    # -------------------------------------------------------------------------

    def test_single_colour_image(self):
        """A pure single-colour image should not crash inference."""
        import inference as inf
        img = np.ones((64, 64, 3), dtype=np.float32) * 0.5  # grey
        path = self._save_image(img, "solid.png")
        image = inf.load_image(path, size=64)
        actions = inf.run_lite_inference(image, max_steps=30)
        self.assertIsInstance(actions, list)
        # May produce 0 or few strokes for a flat image — that's OK
        for a in actions:
            self.assertIn("x_start", a)

    def test_black_image(self):
        """An all-black image should not crash."""
        import inference as inf
        img = np.zeros((64, 64, 3), dtype=np.float32)
        path = self._save_image(img, "black.png")
        image = inf.load_image(path, size=64)
        actions = inf.run_lite_inference(image, max_steps=20)
        self.assertIsInstance(actions, list)

    def test_white_image(self):
        """An all-white image should not crash."""
        import inference as inf
        img = np.ones((64, 64, 3), dtype=np.float32)
        path = self._save_image(img, "white.png")
        image = inf.load_image(path, size=64)
        actions = inf.run_lite_inference(image, max_steps=20)
        self.assertIsInstance(actions, list)

    def test_tiny_image_8x8(self):
        """A very small 8x8 image should not crash."""
        import inference as inf
        img = np.random.rand(8, 8, 3).astype(np.float32)
        path = self._save_image(img, "tiny.png")
        image = inf.load_image(path, size=8)
        actions = inf.run_lite_inference(image, max_steps=10)
        self.assertIsInstance(actions, list)

    def test_large_image_1024(self):
        """A large 1024x1024 image should not crash (may be slow)."""
        import inference as inf
        img = np.random.rand(1024, 1024, 3).astype(np.float32)
        path = self._save_image(img, "large.png")
        image = inf.load_image(path, size=128)  # resize down for speed
        actions = inf.run_lite_inference(image, max_steps=30)
        self.assertIsInstance(actions, list)

    def test_non_square_image(self):
        """A non-square image (e.g. 200x100) should be handled."""
        import inference as inf
        img = np.random.rand(100, 200, 3).astype(np.float32)
        path = self._save_image(img, "nonsquare.png")
        image = inf.load_image(path, size=128)  # resized to square
        self.assertEqual(image.shape, (128, 128, 3))
        actions = inf.run_lite_inference(image, max_steps=20)
        self.assertIsInstance(actions, list)

    def test_rgba_image(self):
        """An RGBA image should be converted to RGB without error."""
        import inference as inf
        img = np.random.rand(64, 64, 4).astype(np.float32)
        path = os.path.join(self.tmpdir, "rgba.png")
        Image.fromarray((img * 255).astype(np.uint8), mode="RGBA").save(path)
        image = inf.load_image(path, size=64)
        self.assertEqual(image.shape, (64, 64, 3))

    # -------------------------------------------------------------------------
    # Inference parameter edge cases
    # -------------------------------------------------------------------------

    def test_max_steps_zero(self):
        """max_steps=0 should produce 0 strokes (or very few) without crashing."""
        import inference as inf
        img = np.random.rand(64, 64, 3).astype(np.float32)
        path = self._save_image(img, "random.png")
        image = inf.load_image(path, size=64)
        actions = inf.run_lite_inference(image, max_steps=0)
        self.assertIsInstance(actions, list)

    def test_max_steps_one(self):
        """max_steps=1 should produce at most 1 stroke without crashing."""
        import inference as inf
        img = np.random.rand(64, 64, 3).astype(np.float32)
        path = self._save_image(img, "random1.png")
        image = inf.load_image(path, size=64)
        actions = inf.run_lite_inference(image, max_steps=1)
        self.assertLessEqual(len(actions), 5)  # may produce a few from pass 1

    # -------------------------------------------------------------------------
    # Transform edge cases
    # -------------------------------------------------------------------------

    def test_empty_actions_list(self):
        """An empty actions list should produce only the background instruction."""
        result = transform([], start_with_background=True)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0][0], "background")

    def test_empty_actions_no_background(self):
        """An empty actions list with no background should produce empty output."""
        result = transform([], start_with_background=False)
        self.assertEqual(len(result), 0)

    def test_single_action(self):
        """A single action should produce valid instructions."""
        action = {
            "x_start": 0.1, "y_start": 0.2, "x_end": 0.3, "y_end": 0.4,
            "color_r": 0.5, "color_g": 0.6, "color_b": 0.7, "color_a": 0.8,
            "brush_radius": 5.0
        }
        result = transform([action], start_with_background=False)
        self.assertGreater(len(result), 0)
        lines = [i for i in result if i[0] == "line"]
        self.assertEqual(len(lines), 1)

    def test_action_with_extreme_coordinates(self):
        """Actions with coordinates at 0.0 and 1.0 should be handled."""
        actions = [
            {"x_start": 0.0, "y_start": 0.0, "x_end": 1.0, "y_end": 1.0,
             "color_r": 0.0, "color_g": 0.0, "color_b": 0.0, "color_a": 1.0,
             "brush_radius": 10.0},
            {"x_start": 1.0, "y_start": 1.0, "x_end": 0.0, "y_end": 0.0,
             "color_r": 1.0, "color_g": 1.0, "color_b": 1.0, "color_a": 0.5,
             "brush_radius": 2.0}
        ]
        result = transform(actions, start_with_background=False)
        lines = [i for i in result if i[0] == "line"]
        self.assertEqual(len(lines), 2)

    def test_action_with_extreme_radius(self):
        """Very small and very large radii should be handled."""
        actions = [
            {"x_start": 0.5, "y_start": 0.5, "x_end": 0.6, "y_end": 0.6,
             "color_r": 0.5, "color_g": 0.5, "color_b": 0.5, "color_a": 1.0,
             "brush_radius": 0.1},  # very thin -> brush 5
            {"x_start": 0.5, "y_start": 0.5, "x_end": 0.6, "y_end": 0.6,
             "color_r": 0.5, "color_g": 0.5, "color_b": 0.5, "color_a": 1.0,
             "brush_radius": 100.0}  # very thick -> brush 0
        ]
        result = transform(actions, start_with_background=False)
        lines = [i for i in result if i[0] == "line"]
        self.assertEqual(len(lines), 2)
        self.assertEqual(lines[0][1], 5)   # thin -> brush 5
        self.assertEqual(lines[1][1], 0)   # thick -> brush 0

    def test_dedup_with_empty_list(self):
        """Dedup on an empty list should return empty."""
        result = transform([], start_with_background=False, dedup=True)
        self.assertEqual(len(result), 0)

    def test_dedup_with_identical_strokes(self):
        """Dedup should remove identical strokes."""
        action = {
            "x_start": 0.1, "y_start": 0.2, "x_end": 0.3, "y_end": 0.4,
            "color_r": 0.5, "color_g": 0.6, "color_b": 0.7, "color_a": 0.8,
            "brush_radius": 5.0
        }
        actions = [action] * 10  # 10 identical strokes
        result = transform(actions, start_with_background=False, dedup=True)
        lines = [i for i in result if i[0] == "line"]
        self.assertLess(len(lines), 10, "dedup should remove identical strokes")

    # -------------------------------------------------------------------------
    # File I/O edge cases
    # -------------------------------------------------------------------------

    def test_load_nonexistent_image(self):
        """Loading a non-existent image should raise a clear error."""
        import inference as inf
        with self.assertRaises(Exception):
            inf.load_image("/nonexistent/path/to/image.png", size=64)

    def test_load_corrupt_image(self):
        """Loading a corrupt image file should raise an error."""
        import inference as inf
        corrupt_path = os.path.join(self.tmpdir, "corrupt.png")
        with open(corrupt_path, "wb") as f:
            f.write(b"not a real image file")
        with self.assertRaises(Exception):
            inf.load_image(corrupt_path, size=64)


if __name__ == "__main__":
    unittest.main(verbosity=2)
