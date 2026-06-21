#!/usr/bin/env python3
"""
Tests for the evaluation metrics and visualization tools.

Run with:
    python -m unittest tests.test_evaluation
"""
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


class TestMetrics(unittest.TestCase):
    """Tests for image quality metrics."""

    @classmethod
    def setUpClass(cls):
        from evaluation import (
            mse, psnr, ssim, lpips_simplified, compute_all_metrics
        )
        cls.mse = staticmethod(mse)
        cls.psnr = staticmethod(psnr)
        cls.ssim = staticmethod(ssim)
        cls.lpips = staticmethod(lpips_simplified)
        cls.compute_all = staticmethod(compute_all_metrics)
        # Create test images
        cls.img1 = np.random.rand(32, 32, 3).astype(np.float32)
        cls.img2 = cls.img1.copy()  # identical
        cls.img3 = np.random.rand(32, 32, 3).astype(np.float32)  # different

    def test_mse_identical(self):
        """MSE of identical images should be 0."""
        self.assertEqual(self.mse(self.img1, self.img2), 0.0)

    def test_mse_different(self):
        """MSE of different images should be > 0."""
        self.assertGreater(self.mse(self.img1, self.img3), 0.0)

    def test_psnr_identical(self):
        """PSNR of identical images should be infinity."""
        self.assertEqual(self.psnr(self.img1, self.img2), float('inf'))

    def test_psnr_different(self):
        """PSNR of different images should be finite."""
        val = self.psnr(self.img1, self.img3)
        self.assertGreater(val, 0)
        self.assertLess(val, float('inf'))

    def test_ssim_identical(self):
        """SSIM of identical images should be 1.0."""
        val = self.ssim(self.img1, self.img2)
        self.assertAlmostEqual(val, 1.0, places=1)

    def test_ssim_different(self):
        """SSIM of different images should be < 1.0."""
        val = self.ssim(self.img1, self.img3)
        self.assertLess(val, 1.0)

    def test_lpips_identical(self):
        """LPIPS of identical images should be 0."""
        self.assertAlmostEqual(self.lpips(self.img1, self.img2), 0.0, places=4)

    def test_lpips_different(self):
        """LPIPS of different images should be > 0."""
        self.assertGreater(self.lpips(self.img1, self.img3), 0.0)

    def test_compute_all_metrics(self):
        """compute_all_metrics should return all 4 metrics."""
        m = self.compute_all(self.img1, self.img3)
        self.assertIn('mse', m)
        self.assertIn('psnr', m)
        self.assertIn('ssim', m)
        self.assertIn('lpips', m)


class TestVisualization(unittest.TestCase):
    """Tests for visualization tools."""

    def test_training_visualizer(self):
        """TrainingVisualizer should record and save history."""
        from evaluation import TrainingVisualizer
        tmpdir = tempfile.mkdtemp(prefix="asp_viz_")
        viz = TrainingVisualizer(save_dir=tmpdir)
        viz.record(reward=1.0, critic_loss=0.5, actor_loss=0.3)
        viz.record(reward=1.5, critic_loss=0.4, actor_loss=0.2)
        self.assertEqual(len(viz.history['rewards']), 2)
        viz.save_history()
        self.assertTrue(os.path.exists(os.path.join(tmpdir, 'history.json')))

    def test_painting_visualizer(self):
        """PaintingVisualizer should record steps."""
        from evaluation import PaintingVisualizer
        tmpdir = tempfile.mkdtemp(prefix="asp_paint_")
        viz = PaintingVisualizer(save_dir=tmpdir)
        img = np.random.rand(32, 32, 3).astype(np.float32)
        viz.record(0, img)
        viz.record(1, img)
        self.assertEqual(len(viz.steps), 2)


class TestComparison(unittest.TestCase):
    """Tests for comparison utilities."""

    def test_compare_methods(self):
        """compare_painting_methods should return metrics for each method."""
        from evaluation import compare_painting_methods, print_comparison_table
        target = np.random.rand(32, 32, 3).astype(np.float32)
        results = {
            'method_a': np.random.rand(32, 32, 3).astype(np.float32),
            'method_b': np.random.rand(32, 32, 3).astype(np.float32),
        }
        metrics = compare_painting_methods(target, results)
        self.assertIn('method_a', metrics)
        self.assertIn('method_b', metrics)
        self.assertIn('ssim', metrics['method_a'])
        # print_comparison_table should not crash
        print_comparison_table(metrics)


if __name__ == "__main__":
    unittest.main(verbosity=2)
