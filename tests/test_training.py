#!/usr/bin/env python3
"""
Tests for the training system (dataset, reward, environment, trainer).

Run with:
    python -m unittest tests.test_training
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
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
if DATA_DIR not in sys.path:
    sys.path.insert(0, DATA_DIR)


class TestPaintDataset(unittest.TestCase):
    """Tests for PaintDataset."""

    @classmethod
    def setUpClass(cls):
        try:
            import torch
            cls.torch = torch
        except ImportError:
            raise unittest.SkipTest("PyTorch not available")
        # Create a temp directory with test images
        from PIL import Image
        import numpy as np
        cls.tmpdir = tempfile.mkdtemp(prefix="asp_dataset_")
        for i in range(5):
            arr = np.random.rand(32, 32, 3) * 255
            Image.fromarray(arr.astype('uint8')).save(
                os.path.join(cls.tmpdir, f"img_{i}.png")
            )

    def test_dataset_load(self):
        """Dataset should load images from a directory."""
        from dataset import PaintDataset
        ds = PaintDataset(self.tmpdir, canvas_size=32, augment=False)
        self.assertEqual(len(ds), 5)

    def test_dataset_getitem(self):
        """__getitem__ should return a tensor with correct shape and range."""
        from dataset import PaintDataset
        ds = PaintDataset(self.tmpdir, canvas_size=32, augment=False)
        img = ds[0]
        self.assertEqual(img.shape, (3, 32, 32))
        self.assertGreaterEqual(img.min(), 0)
        self.assertLessEqual(img.max(), 1)

    def test_dataset_augmentation(self):
        """Augmentation should not crash and should return valid tensors."""
        from dataset import PaintDataset
        ds = PaintDataset(self.tmpdir, canvas_size=32, augment=True)
        for i in range(5):
            img = ds[i]
            self.assertEqual(img.shape, (3, 32, 32))

    def test_dataset_empty_dir(self):
        """Empty directory should raise ValueError."""
        from dataset import PaintDataset
        empty_dir = tempfile.mkdtemp(prefix="asp_empty_")
        with self.assertRaises(ValueError):
            PaintDataset(empty_dir)


class TestRewardFunction(unittest.TestCase):
    """Tests for RewardFunction."""

    @classmethod
    def setUpClass(cls):
        try:
            import torch
            cls.torch = torch
        except ImportError:
            raise unittest.SkipTest("PyTorch not available")

    def test_reward_identical_images(self):
        """Reward should be high when canvas == target."""
        from dataset import RewardFunction
        rf = RewardFunction(use_lpips=False)
        img = self.torch.rand(1, 3, 16, 16)
        reward, info = rf.compute(img, img)
        # SSIM of identical images = 1, MSE = 0
        self.assertAlmostEqual(info['mse'], 0, places=4)
        self.assertAlmostEqual(info['ssim'], 1, places=2)
        self.assertGreater(reward, 0)

    def test_reward_different_images(self):
        """Reward should be lower when canvas != target."""
        from dataset import RewardFunction
        rf = RewardFunction(use_lpips=False)
        target = self.torch.ones(1, 3, 16, 16)
        canvas = self.torch.zeros(1, 3, 16, 16)
        reward, info = rf.compute(canvas, target)
        self.assertGreater(info['mse'], 0)
        self.assertLess(info['ssim'], 1)

    def test_reward_delta(self):
        """Delta reward should be positive when canvas improves."""
        from dataset import RewardFunction
        rf = RewardFunction(use_lpips=False)
        target = self.torch.ones(1, 3, 16, 16)
        prev = self.torch.zeros(1, 3, 16, 16)
        curr = self.torch.full((1, 3, 16, 16), 0.5)
        reward, info = rf.compute(curr, target, prev)
        self.assertIn('delta_mse', info)
        self.assertGreater(info['delta_mse'], 0)  # improvement


class TestPaintEnv(unittest.TestCase):
    """Tests for PaintEnv."""

    @classmethod
    def setUpClass(cls):
        try:
            import torch
            cls.torch = torch
        except ImportError:
            raise unittest.SkipTest("PyTorch not available")

    def test_env_reset(self):
        """reset() should return (target, black_canvas)."""
        from differentiable_renderer import build_differentiable_renderer
        from dataset import RewardFunction, PaintEnv
        renderer = build_differentiable_renderer(canvas_size=16, num_samples=4)
        rf = RewardFunction(use_lpips=False)
        env = PaintEnv(renderer, rf, max_strokes=5)
        target = self.torch.rand(1, 3, 16, 16)
        t, c = env.reset(target)
        self.assertTrue(self.torch.allclose(t, target))
        self.assertTrue(self.torch.all(c == 0))

    def test_env_step(self):
        """step() should return new state, reward, done, info."""
        from differentiable_renderer import build_differentiable_renderer
        from dataset import RewardFunction, PaintEnv
        renderer = build_differentiable_renderer(canvas_size=16, num_samples=4)
        rf = RewardFunction(use_lpips=False)
        env = PaintEnv(renderer, rf, max_strokes=3)
        target = self.torch.rand(1, 3, 16, 16)
        env.reset(target)
        action = self.torch.randn(1, 15)
        state, reward, done, info = env.step(action)
        self.assertEqual(state[0].shape, (1, 3, 16, 16))
        self.assertEqual(state[1].shape, (1, 3, 16, 16))
        self.assertIsInstance(reward, float)
        self.assertFalse(done)
        self.assertIn('step', info)

    def test_env_done(self):
        """Env should be done after max_strokes steps."""
        from differentiable_renderer import build_differentiable_renderer
        from dataset import RewardFunction, PaintEnv
        renderer = build_differentiable_renderer(canvas_size=16, num_samples=4)
        rf = RewardFunction(use_lpips=False)
        env = PaintEnv(renderer, rf, max_strokes=2)
        env.reset(self.torch.rand(1, 3, 16, 16))
        action = self.torch.randn(1, 15)
        env.step(action)
        _, _, done, _ = env.step(action)
        self.assertTrue(done)


class TestTrainer(unittest.TestCase):
    """Tests for the Trainer class."""

    @classmethod
    def setUpClass(cls):
        try:
            import torch
            cls.torch = torch
        except ImportError:
            raise unittest.SkipTest("PyTorch not available")
        from PIL import Image
        import numpy as np
        cls.tmpdir = tempfile.mkdtemp(prefix="asp_trainer_")
        for i in range(3):
            arr = np.random.rand(16, 16, 3) * 255
            Image.fromarray(arr.astype('uint8')).save(
                os.path.join(cls.tmpdir, f"img_{i}.png")
            )

    def test_trainer_creation(self):
        """Trainer should be created without error."""
        from differentiable_renderer import build_differentiable_renderer
        from ddpg_agent import build_ddpg_agent
        from dataset import PaintDataset, Trainer
        renderer = build_differentiable_renderer(canvas_size=16, num_samples=4)
        agent = build_ddpg_agent(canvas_size=16, device='cpu', batch_size=4)
        ds = PaintDataset(self.tmpdir, canvas_size=16, augment=False)
        trainer = Trainer(agent, renderer, ds, device='cpu', max_strokes=3)
        self.assertEqual(trainer.max_strokes, 3)

    def test_trainer_short_run(self):
        """Trainer should run 1 episode without crashing."""
        from differentiable_renderer import build_differentiable_renderer
        from ddpg_agent import build_ddpg_agent
        from dataset import PaintDataset, Trainer
        renderer = build_differentiable_renderer(canvas_size=16, num_samples=4)
        agent = build_ddpg_agent(canvas_size=16, device='cpu', batch_size=4)
        ds = PaintDataset(self.tmpdir, canvas_size=16, augment=False)
        trainer = Trainer(agent, renderer, ds, device='cpu', max_strokes=3,
                          save_dir=tempfile.mkdtemp(prefix="asp_ckpt_"))
        history = trainer.train(num_episodes=1)
        self.assertIn('rewards', history)
        self.assertEqual(len(history['rewards']), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
