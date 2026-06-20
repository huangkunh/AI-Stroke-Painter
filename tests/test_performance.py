#!/usr/bin/env python3
"""
Performance tests for the AI-Stroke-Painter pipeline.

Measures inference + transform time across different image sizes and stroke
counts. Establishes a performance baseline so regressions can be detected.

Run with:
    python -m unittest tests.test_performance
"""
import json
import os
import sys
import time
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


class TestPerformanceBaseline(unittest.TestCase):
    """Performance baselines for inference and transform.

    These are SMOKE tests, not strict benchmarks. They verify the pipeline
    completes within a generous time budget so regressions (e.g. accidental
    O(n²) algorithms) are caught. Times will vary by hardware.
    """

    @classmethod
    def setUpClass(cls):
        cls.tmpdir = tempfile.mkdtemp(prefix="asp_perf_")
        # Create test images of different sizes
        cls.images = {}
        for size in (128, 256, 512):
            img = np.random.rand(size, size, 3).astype(np.float32)
            path = os.path.join(cls.tmpdir, f"random_{size}.png")
            Image.fromarray((img * 255).astype(np.uint8)).save(path)
            cls.images[size] = path

    def _run_inference(self, image_path, max_steps, size=256):
        import inference as inf
        image = inf.load_image(image_path, size=size)
        t0 = time.time()
        actions = inf.run_lite_inference(image, max_steps=max_steps)
        t1 = time.time()
        return actions, t1 - t0

    def _run_transform(self, actions, dedup=False):
        t0 = time.time()
        instructions = transform(actions, start_with_background=True, dedup=dedup)
        t1 = time.time()
        return instructions, t1 - t0

    def test_small_image_fast_inference(self):
        """128x128 image with 50 steps should complete in < 3 seconds."""
        actions, elapsed = self._run_inference(self.images[128], max_steps=50, size=128)
        self.assertGreater(len(actions), 0)
        self.assertLess(elapsed, 3.0, f"Small inference took {elapsed:.2f}s (>3s)")
        print(f"\n  128px/50steps: {elapsed:.3f}s, {len(actions)} strokes")

    def test_medium_image_inference(self):
        """256x256 image with 200 steps should complete in < 8 seconds."""
        actions, elapsed = self._run_inference(self.images[256], max_steps=200, size=256)
        self.assertGreater(len(actions), 0)
        self.assertLess(elapsed, 8.0, f"Medium inference took {elapsed:.2f}s (>8s)")
        print(f"\n  256px/200steps: {elapsed:.3f}s, {len(actions)} strokes")

    def test_large_image_inference(self):
        """512x512 image with 400 steps should complete in < 20 seconds."""
        actions, elapsed = self._run_inference(self.images[512], max_steps=400, size=512)
        self.assertGreater(len(actions), 0)
        self.assertLess(elapsed, 20.0, f"Large inference took {elapsed:.2f}s (>20s)")
        print(f"\n  512px/400steps: {elapsed:.3f}s, {len(actions)} strokes")

    def test_transform_performance(self):
        """Transform of 400 strokes should complete in < 0.5 seconds."""
        actions, _ = self._run_inference(self.images[256], max_steps=400, size=256)
        instructions, elapsed = self._run_transform(actions)
        self.assertGreater(len(instructions), 0)
        self.assertLess(elapsed, 0.5, f"Transform took {elapsed:.2f}s (>0.5s)")
        print(f"\n  transform 400 strokes: {elapsed:.3f}s, {len(instructions)} instructions")

    def test_transform_with_dedup_performance(self):
        """Transform with dedup of 400 strokes should complete in < 1 second."""
        actions, _ = self._run_inference(self.images[256], max_steps=400, size=256)
        instructions, elapsed = self._run_transform(actions, dedup=True)
        self.assertGreater(len(instructions), 0)
        self.assertLess(elapsed, 1.0, f"Transform+dedup took {elapsed:.2f}s (>1s)")
        print(f"\n  transform+dedup 400 strokes: {elapsed:.3f}s, {len(instructions)} instructions")

    def test_full_pipeline_end_to_end(self):
        """Full pipeline (128px, 100 steps) should complete in < 5 seconds."""
        actions, t_inf = self._run_inference(self.images[128], max_steps=100, size=128)
        instructions, t_tr = self._run_transform(actions)
        total = t_inf + t_tr
        self.assertLess(total, 5.0, f"Full pipeline took {total:.2f}s (>5s)")
        print(f"\n  full pipeline 128px/100steps: {total:.3f}s "
              f"(infer={t_inf:.3f}s, transform={t_tr:.3f}s)")


if __name__ == "__main__":
    unittest.main(verbosity=2)
