#!/usr/bin/env python3
"""
Tests for the enhanced evaluation system.

Run with:
    python -m unittest tests.test_enhanced_evaluation
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


class TestEnhancedMetrics(unittest.TestCase):
    """Tests for enhanced perceptual metrics."""

    @classmethod
    def setUpClass(cls):
        from enhanced_evaluation import (
            fsim, color_distribution_similarity, edge_preservation,
            painting_features, style_consistency, compute_enhanced_metrics
        )
        cls.fsim = staticmethod(fsim)
        cls.color_sim = staticmethod(color_distribution_similarity)
        cls.edge_pres = staticmethod(edge_preservation)
        cls.painting_features = staticmethod(painting_features)
        cls.style_consistency = staticmethod(style_consistency)
        cls.compute_all = staticmethod(compute_enhanced_metrics)
        cls.img1 = np.random.rand(32, 32, 3).astype(np.float32)
        cls.img2 = cls.img1.copy()
        cls.img3 = np.random.rand(32, 32, 3).astype(np.float32)

    def test_fsim_identical(self):
        """FSIM of identical images should be ~1.0."""
        score = self.fsim(self.img1, self.img2)
        self.assertGreater(score, 0.95)

    def test_fsim_different(self):
        """FSIM of different images should be < 1.0."""
        score = self.fsim(self.img1, self.img3)
        self.assertLess(score, 1.0)

    def test_color_similarity_identical(self):
        """Color similarity of identical images should be ~1.0."""
        score = self.color_sim(self.img1, self.img2)
        self.assertGreater(score, 0.9)

    def test_color_similarity_different(self):
        """Color similarity of different images should be < 1.0."""
        score = self.color_sim(self.img1, self.img3)
        self.assertLess(score, 1.0)

    def test_edge_preservation_identical(self):
        """Edge preservation of identical images should be ~1.0."""
        score = self.edge_pres(self.img1, self.img2)
        self.assertGreater(score, 0.9)

    def test_edge_preservation_different(self):
        """Edge preservation of different images should be < 1.0."""
        score = self.edge_pres(self.img1, self.img3)
        self.assertLess(score, 1.0)

    def test_painting_features_empty(self):
        """Empty strokes should return zero features."""
        features = self.painting_features([])
        self.assertEqual(features['stroke_count'], 0)

    def test_painting_features_with_strokes(self):
        """Features should be computed from strokes."""
        strokes = [
            {'x_start': 0.1, 'y_start': 0.1, 'x_end': 0.5, 'y_end': 0.5,
             'color_r': 0.5, 'color_g': 0.3, 'color_b': 0.2,
             'color_a': 0.7, 'brush_radius': 5.0},
            {'x_start': 0.3, 'y_start': 0.3, 'x_end': 0.7, 'y_end': 0.7,
             'color_r': 0.6, 'color_g': 0.4, 'color_b': 0.3,
             'color_a': 0.8, 'brush_radius': 3.0},
        ]
        features = self.painting_features(strokes, painting_time=1.5)
        self.assertEqual(features['stroke_count'], 2)
        self.assertEqual(features['painting_time'], 1.5)
        self.assertGreater(features['avg_stroke_length'], 0)
        self.assertGreater(features['color_diversity'], 0)

    def test_style_consistency(self):
        """Style consistency should return a score in [0, 1]."""
        strokes = [
            {'brush_radius': 5.0, 'color_a': 0.7},
            {'brush_radius': 6.0, 'color_a': 0.8},
        ]
        score = self.style_consistency(strokes, 'default')
        self.assertGreaterEqual(score, 0)
        self.assertLessEqual(score, 1)

    def test_compute_enhanced_metrics(self):
        """compute_enhanced_metrics should return all metrics."""
        metrics = self.compute_all(self.img1, self.img2)
        expected_keys = {
            'mse', 'psnr', 'ssim', 'lpips', 'fsim',
            'color_similarity', 'edge_preservation', 'overall_quality'
        }
        self.assertTrue(expected_keys.issubset(set(metrics.keys())))
        self.assertGreater(metrics['overall_quality'], 0.5)


class TestABTestFramework(unittest.TestCase):
    """Tests for A/B testing framework."""

    def test_creation(self):
        from enhanced_evaluation import ABTestFramework
        ab = ABTestFramework('Method A', 'Method B')
        self.assertEqual(ab.method_a_name, 'Method A')
        self.assertEqual(ab.method_b_name, 'Method B')

    def test_add_rating(self):
        from enhanced_evaluation import ABTestFramework
        ab = ABTestFramework()
        ab.add_rating('user1', 'img1', 0.8, 0.6, 'a')
        ab.add_rating('user2', 'img1', 0.7, 0.9, 'b')
        self.assertEqual(len(ab.results), 2)

    def test_statistics(self):
        from enhanced_evaluation import ABTestFramework
        ab = ABTestFramework()
        ab.add_rating('user1', 'img1', 0.8, 0.6, 'a')
        ab.add_rating('user2', 'img1', 0.7, 0.9, 'b')
        ab.add_rating('user3', 'img1', 0.5, 0.5, 'none')
        stats = ab.compute_statistics()
        self.assertEqual(stats['sample_size'], 3)
        self.assertAlmostEqual(stats['mean_rating_a'], (0.8 + 0.7 + 0.5) / 3)
        self.assertAlmostEqual(stats['win_rate_a'], 1 / 3)
        self.assertAlmostEqual(stats['win_rate_b'], 1 / 3)

    def test_empty_statistics(self):
        from enhanced_evaluation import ABTestFramework
        ab = ABTestFramework()
        stats = ab.compute_statistics()
        self.assertEqual(stats['sample_size'], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
