#!/usr/bin/env python3
"""
Tests for the serverless API functions (api/infer.py, api/transform.py,
api/pipeline.py).

Tests the CLI mode of each API function, which exercises the same _handle()
logic as the serverless runtimes.

Run with:
    python -m unittest tests.test_api
"""
import base64
import json
import os
import sys
import tempfile
import unittest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
API_DIR = os.path.join(PROJECT_ROOT, "api")
if API_DIR not in sys.path:
    sys.path.insert(0, API_DIR)
MODEL_DIR = os.path.join(PROJECT_ROOT, "model")
if MODEL_DIR not in sys.path:
    sys.path.insert(0, MODEL_DIR)

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402


class TestInferAPI(unittest.TestCase):
    """Tests for api/infer.py _handle() logic."""

    @classmethod
    def setUpClass(cls):
        cls.tmpdir = tempfile.mkdtemp(prefix="asp_api_")
        # Create a small test image
        img = np.random.rand(64, 64, 3).astype(np.float32)
        cls.image_path = os.path.join(cls.tmpdir, "test.png")
        Image.fromarray((img * 255).astype(np.uint8)).save(cls.image_path)
        with open(cls.image_path, "rb") as f:
            cls.image_b64 = base64.b64encode(f.read()).decode("ascii")

    def test_infer_valid_request(self):
        """A valid request should return 200 with strokes."""
        import infer
        body = {"image": self.image_b64, "mode": "lite", "max_steps": 20, "size": 64}
        status, resp = infer._handle(body, {}, "127.0.0.1")
        self.assertEqual(status, 200)
        self.assertIn("strokes", resp)
        self.assertIn("count", resp)
        self.assertIn("mode", resp)
        self.assertIn("elapsed_ms", resp)
        self.assertGreater(resp["count"], 0)

    def test_infer_missing_image(self):
        """Request without image field should return 400."""
        import infer
        status, resp = infer._handle({}, {}, "127.0.0.1")
        self.assertEqual(status, 400)
        self.assertIn("error", resp)

    def test_infer_invalid_mode(self):
        """Invalid mode should return 400."""
        import infer
        body = {"image": self.image_b64, "mode": "invalid"}
        status, resp = infer._handle(body, {}, "127.0.0.1")
        self.assertEqual(status, 400)
        self.assertIn("error", resp)

    def test_infer_max_steps_out_of_range(self):
        """max_steps > 2000 should return 400."""
        import infer
        body = {"image": self.image_b64, "max_steps": 5000}
        status, resp = infer._handle(body, {}, "127.0.0.1")
        self.assertEqual(status, 400)

    def test_infer_size_out_of_range(self):
        """size > 1024 should return 400."""
        import infer
        body = {"image": self.image_b64, "size": 2048}
        status, resp = infer._handle(body, {}, "127.0.0.1")
        self.assertEqual(status, 400)

    def test_infer_api_key_required(self):
        """When ASP_API_KEY is set, requests without key should return 401."""
        import infer
        os.environ["ASP_API_KEY"] = "test-secret-key"
        try:
            status, resp = infer._handle({"image": self.image_b64}, {}, "127.0.0.1")
            self.assertEqual(status, 401)
            self.assertIn("error", resp)

            # With correct key, should succeed
            status2, resp2 = infer._handle(
                {"image": self.image_b64, "max_steps": 10, "size": 64},
                {"x-api-key": "test-secret-key"},
                "127.0.0.1"
            )
            self.assertEqual(status2, 200)
        finally:
            del os.environ["ASP_API_KEY"]


class TestTransformAPI(unittest.TestCase):
    """Tests for api/transform.py _handle() logic."""

    def test_transform_valid_request(self):
        """A valid request should return 200 with instructions."""
        import transform as transform_api
        strokes = [
            {"x_start": 0.1, "y_start": 0.2, "x_end": 0.3, "y_end": 0.4,
             "color_r": 0.5, "color_g": 0.6, "color_b": 0.7, "color_a": 0.8,
             "brush_radius": 5.0}
        ]
        body = {"strokes": strokes, "background": "#f8ecdb"}
        status, resp = transform_api._handle(body, {}, "127.0.0.1")
        self.assertEqual(status, 200)
        self.assertIn("instructions", resp)
        self.assertIn("count", resp)
        self.assertGreater(resp["count"], 0)

    def test_transform_missing_strokes(self):
        """Request without strokes should return 400."""
        import transform as transform_api
        status, resp = transform_api._handle({}, {}, "127.0.0.1")
        self.assertEqual(status, 400)
        self.assertIn("error", resp)

    def test_transform_invalid_strokes_type(self):
        """Non-array strokes should return 400."""
        import transform as transform_api
        status, resp = transform_api._handle({"strokes": "not an array"}, {}, "127.0.0.1")
        self.assertEqual(status, 400)

    def test_transform_with_dedup(self):
        """dedup=True should work without error."""
        import transform as transform_api
        strokes = [
            {"x_start": 0.1, "y_start": 0.2, "x_end": 0.3, "y_end": 0.4,
             "color_r": 0.5, "color_g": 0.6, "color_b": 0.7, "color_a": 0.8,
             "brush_radius": 5.0}
        ] * 5  # 5 identical strokes
        body = {"strokes": strokes, "dedup": True}
        status, resp = transform_api._handle(body, {}, "127.0.0.1")
        self.assertEqual(status, 200)
        self.assertGreater(resp["count"], 0)


class TestPipelineAPI(unittest.TestCase):
    """Tests for api/pipeline.py _handle() logic."""

    @classmethod
    def setUpClass(cls):
        cls.tmpdir = tempfile.mkdtemp(prefix="asp_pipe_")
        img = np.random.rand(64, 64, 3).astype(np.float32)
        cls.image_path = os.path.join(cls.tmpdir, "test.png")
        Image.fromarray((img * 255).astype(np.uint8)).save(cls.image_path)
        with open(cls.image_path, "rb") as f:
            cls.image_b64 = base64.b64encode(f.read()).decode("ascii")

    def test_pipeline_valid_request(self):
        """A valid pipeline request should return 200 with instructions."""
        import pipeline
        body = {"image": self.image_b64, "mode": "lite",
                "max_steps": 20, "size": 64, "background": "#f8ecdb"}
        status, resp = pipeline._handle(body, {}, "127.0.0.1")
        self.assertEqual(status, 200)
        self.assertIn("instructions", resp)
        self.assertIn("stroke_count", resp)
        self.assertIn("instruction_count", resp)
        self.assertIn("mode", resp)
        self.assertIn("elapsed_ms", resp)
        self.assertGreater(resp["stroke_count"], 0)
        self.assertGreater(resp["instruction_count"], 0)

    def test_pipeline_missing_image(self):
        """Request without image should return 400."""
        import pipeline
        status, resp = pipeline._handle({}, {}, "127.0.0.1")
        self.assertEqual(status, 400)

    def test_pipeline_with_dedup(self):
        """Pipeline with dedup=True should work."""
        import pipeline
        body = {"image": self.image_b64, "mode": "lite",
                "max_steps": 20, "size": 64, "dedup": True}
        status, resp = pipeline._handle(body, {}, "127.0.0.1")
        self.assertEqual(status, 200)
        self.assertIn("instructions", resp)


if __name__ == "__main__":
    unittest.main(verbosity=2)
