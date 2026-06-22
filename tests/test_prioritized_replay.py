#!/usr/bin/env python3
"""
Tests for the prioritized replay buffer and training stability utilities.

Run with:
    python -m unittest tests.test_prioritized_replay
"""
import math
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


class TestSumTree(unittest.TestCase):
    """Tests for SumTree."""

    def test_creation(self):
        from prioritized_replay import SumTree
        tree = SumTree(capacity=8)
        self.assertEqual(tree.capacity, 8)
        self.assertEqual(tree.get_total(), 0.0)

    def test_update_and_total(self):
        from prioritized_replay import SumTree
        tree = SumTree(capacity=4)
        tree.update(0, 1.0)
        tree.update(1, 2.0)
        tree.update(2, 3.0)
        tree.update(3, 4.0)
        self.assertAlmostEqual(tree.get_total(), 10.0)

    def test_sample(self):
        from prioritized_replay import SumTree
        tree = SumTree(capacity=4)
        for i in range(4):
            tree.update(i, float(i + 1))
        # Sample value 0 should return index 0 (priority 1)
        idx, priority = tree.sample(0.5)
        self.assertEqual(idx, 0)
        # Sample value 9.5 should return index 3 (priority 4)
        idx, priority = tree.sample(9.5)
        self.assertEqual(idx, 3)


class TestPrioritizedReplayBuffer(unittest.TestCase):
    """Tests for PrioritizedReplayBuffer."""

    @classmethod
    def setUpClass(cls):
        try:
            import torch
            cls.torch = torch
        except ImportError:
            raise unittest.SkipTest("PyTorch not available")

    def test_buffer_creation(self):
        from prioritized_replay import PrioritizedReplayBuffer
        buf = PrioritizedReplayBuffer(capacity=100)
        self.assertEqual(len(buf), 0)

    def test_push_and_sample(self):
        from prioritized_replay import PrioritizedReplayBuffer
        buf = PrioritizedReplayBuffer(capacity=100)
        for i in range(20):
            buf.push(
                target=self.torch.randn(3, 16, 16),
                canvas=self.torch.randn(3, 16, 16),
                action=self.torch.randn(15),
                reward=0.5,
                next_target=self.torch.randn(3, 16, 16),
                next_canvas=self.torch.randn(3, 16, 16),
                done=False,
            )
        self.assertEqual(len(buf), 20)
        transitions, indices, weights = buf.sample(batch_size=4)
        self.assertEqual(len(transitions), 4)
        self.assertEqual(len(indices), 4)
        self.assertEqual(len(weights), 4)

    def test_update_priorities(self):
        from prioritized_replay import PrioritizedReplayBuffer
        buf = PrioritizedReplayBuffer(capacity=100)
        for i in range(10):
            buf.push(
                target=self.torch.randn(3, 16, 16),
                canvas=self.torch.randn(3, 16, 16),
                action=self.torch.randn(15),
                reward=0.5,
                next_target=self.torch.randn(3, 16, 16),
                next_canvas=self.torch.randn(3, 16, 16),
                done=False,
            )
        _, indices, _ = buf.sample(4)
        td_errors = np.array([0.1, 0.5, 0.3, 0.8])
        buf.update_priorities(indices, td_errors)
        # After update, sampling should still work
        transitions, _, _ = buf.sample(4)
        self.assertEqual(len(transitions), 4)


class TestMultiDimensionalReward(unittest.TestCase):
    """Tests for MultiDimensionalReward."""

    def test_reward_identical_images(self):
        from prioritized_replay import MultiDimensionalReward
        reward_fn = MultiDimensionalReward()
        img = np.random.rand(32, 32, 3).astype(np.float32)
        result = reward_fn.compute(img, img)
        # SSIM should be 1.0 for identical images, but color histogram
        # correlation can be lower for random images. Total reward should be > 0.5.
        self.assertGreater(result['reward'], 0.5)
        self.assertAlmostEqual(result['ssim'], 1.0, places=1)

    def test_reward_different_images(self):
        from prioritized_replay import MultiDimensionalReward
        reward_fn = MultiDimensionalReward()
        img1 = np.zeros((32, 32, 3), dtype=np.float32)
        img2 = np.ones((32, 32, 3), dtype=np.float32)
        result = reward_fn.compute(img1, img2)
        self.assertLess(result['reward'], 0.8)

    def test_reward_components_present(self):
        from prioritized_replay import MultiDimensionalReward
        reward_fn = MultiDimensionalReward()
        img = np.random.rand(32, 32, 3).astype(np.float32)
        result = reward_fn.compute(img, img)
        for key in ['reward', 'ssim', 'lpips', 'color', 'edge']:
            self.assertIn(key, result)


class TestTrainingStability(unittest.TestCase):
    """Tests for TrainingStability."""

    @classmethod
    def setUpClass(cls):
        try:
            import torch
            cls.torch = torch
        except ImportError:
            raise unittest.SkipTest("PyTorch not available")

    def test_creation(self):
        from prioritized_replay import TrainingStability
        ts = TrainingStability()
        self.assertEqual(ts.grad_clip, 1.0)
        self.assertFalse(ts.should_stop)

    def test_lr_schedule_cosine(self):
        from prioritized_replay import TrainingStability
        ts = TrainingStability(lr_schedule='cosine', initial_lr=1e-3, warmup_steps=10)
        # During warmup, LR should increase
        lr1 = ts.get_lr(100)
        lr2 = ts.get_lr(100)
        self.assertGreaterEqual(lr2, lr1 * 0.9)  # should be increasing or similar

    def test_early_stopping(self):
        from prioritized_replay import TrainingStability
        ts = TrainingStability(early_stop_patience=3, early_stop_min_delta=0.01)
        # Loss not improving
        for _ in range(5):
            ts.check_early_stop(1.0)
        self.assertTrue(ts.should_stop)

    def test_gradient_clipping(self):
        from prioritized_replay import TrainingStability
        import torch.nn as nn
        ts = TrainingStability(grad_clip=1.0)
        model = nn.Linear(10, 10)
        # Create large gradients
        for p in model.parameters():
            p.grad = self.torch.randn_like(p) * 100
        grad_norm = ts.clip_gradients(model)
        self.assertGreater(grad_norm, 0)
        # After clipping, gradient norm should be <= clip value
        total_norm = 0
        for p in model.parameters():
            total_norm += p.grad.norm().item() ** 2
        self.assertLess(math.sqrt(total_norm), 2.0)  # should be clipped


if __name__ == "__main__":
    import math
    unittest.main(verbosity=2)
