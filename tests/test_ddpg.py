#!/usr/bin/env python3
"""
Tests for the DDPG agent.

Run with:
    python -m unittest tests.test_ddpg
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


class TestDDPGAgent(unittest.TestCase):
    """Tests for model/ddpg_agent.py."""

    @classmethod
    def setUpClass(cls):
        try:
            import torch
            cls.torch = torch
            from ddpg_agent import (
                DDPGAgent, Actor, Critic, ReplayBuffer, OUNoise,
                build_ddpg_agent, DEFAULT_ACTION_DIM
            )
            cls.DDPGAgent = DDPGAgent
            cls.Actor = Actor
            cls.Critic = Critic
            cls.ReplayBuffer = ReplayBuffer
            cls.OUNoise = OUNoise
            cls.build_agent = staticmethod(build_ddpg_agent)
            cls.DEFAULT_ACTION_DIM = DEFAULT_ACTION_DIM
        except ImportError as e:
            raise unittest.SkipTest(f"PyTorch not available: {e}")

    def test_actor_forward(self):
        """Actor should output actions in [-1, 1] with correct shape."""
        actor = self.Actor(canvas_size=32)
        target = self.torch.randn(2, 3, 32, 32)
        canvas = self.torch.randn(2, 3, 32, 32)
        h, c = actor.init_hidden(2, self.torch.device('cpu'))
        action, (h2, c2) = actor(target, canvas, (h, c))
        self.assertEqual(action.shape, (2, self.DEFAULT_ACTION_DIM))
        self.assertTrue((action >= -1.0).all() and (action <= 1.0).all())

    def test_critic_forward(self):
        """Critic should output scalar Q-values."""
        critic = self.Critic(canvas_size=32)
        target = self.torch.randn(2, 3, 32, 32)
        canvas = self.torch.randn(2, 3, 32, 32)
        action = self.torch.randn(2, self.DEFAULT_ACTION_DIM)
        h, c = critic.init_hidden(2, self.torch.device('cpu'))
        q, _ = critic(target, canvas, action, (h, c))
        self.assertEqual(q.shape, (2, 1))

    def test_replay_buffer(self):
        """ReplayBuffer should store and sample transitions."""
        buf = self.ReplayBuffer(capacity=100)
        for i in range(50):
            buf.push(
                self.torch.randn(3, 32, 32),  # target
                self.torch.randn(3, 32, 32),  # canvas
                self.torch.randn(self.DEFAULT_ACTION_DIM),  # action
                0.5,  # reward
                self.torch.randn(3, 32, 32),  # next_canvas
                False,  # done
                self.torch.randn(256),  # hidden_h
                self.torch.randn(256),  # hidden_c
            )
        self.assertEqual(len(buf), 50)
        batch = buf.sample(16, self.torch.device('cpu'))
        self.assertEqual(len(batch), 8)  # 8-tuple
        self.assertEqual(batch[0].shape[0], 16)  # batch size

    def test_ou_noise(self):
        """OUNoise should produce correlated noise."""
        noise = self.OUNoise(action_dim=15)
        n1 = noise.sample()
        n2 = noise.sample()
        self.assertEqual(n1.shape, (15,))
        # Consecutive samples should be correlated (not identical)
        self.assertFalse((n1 == n2).all())

    def test_agent_creation(self):
        """DDPGAgent should be created with all components."""
        agent = self.build_agent(canvas_size=32, device='cpu')
        self.assertIsNotNone(agent.actor)
        self.assertIsNotNone(agent.critic)
        self.assertIsNotNone(agent.actor_target)
        self.assertIsNotNone(agent.critic_target)
        self.assertIsNotNone(agent.replay_buffer)
        self.assertIsNotNone(agent.noise)

    def test_select_action(self):
        """select_action should return action in [-1, 1]."""
        agent = self.build_agent(canvas_size=32, device='cpu')
        target = self.torch.randn(1, 3, 32, 32)
        canvas = self.torch.randn(1, 3, 32, 32)
        h, c = agent.init_hidden(1)
        action, _ = agent.select_action(target, canvas, (h, c), noise=False)
        self.assertEqual(action.shape, (1, self.DEFAULT_ACTION_DIM))
        self.assertTrue((action >= -1.0).all() and (action <= 1.0).all())

    def test_select_action_with_noise(self):
        """select_action with noise should still be in [-1, 1]."""
        agent = self.build_agent(canvas_size=32, device='cpu')
        target = self.torch.randn(1, 3, 32, 32)
        canvas = self.torch.randn(1, 3, 32, 32)
        h, c = agent.init_hidden(1)
        action, _ = agent.select_action(target, canvas, (h, c), noise=True, noise_scale=0.5)
        self.assertTrue((action >= -1.0).all() and (action <= 1.0).all())

    def test_update_returns_zero_when_buffer_small(self):
        """update() should return (0, 0) when buffer is too small."""
        agent = self.build_agent(canvas_size=32, device='cpu', batch_size=16)
        cl, al = agent.update()
        self.assertEqual(cl, 0.0)
        self.assertEqual(al, 0.0)

    def test_update_with_data(self):
        """update() should return non-zero losses when buffer has enough data."""
        agent = self.build_agent(canvas_size=32, device='cpu', batch_size=4)
        # Fill buffer with random transitions
        for _ in range(10):
            agent.replay_buffer.push(
                self.torch.randn(3, 32, 32),
                self.torch.randn(3, 32, 32),
                self.torch.randn(self.DEFAULT_ACTION_DIM),
                0.5,
                self.torch.randn(3, 32, 32),
                False,
                self.torch.randn(256),
                self.torch.randn(256),
            )
        cl, al = agent.update(batch_size=4)
        self.assertGreater(cl, 0)  # critic loss should be positive

    def test_save_load(self):
        """save/load should preserve agent state."""
        import tempfile
        agent = self.build_agent(canvas_size=32, device='cpu')
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
            path = f.name
        try:
            agent.save(path)
            agent2 = self.build_agent(canvas_size=32, device='cpu')
            agent2.load(path)
            # Check that weights match
            for p1, p2 in zip(agent.actor.parameters(), agent2.actor.parameters()):
                self.assertTrue(self.torch.allclose(p1, p2))
        finally:
            os.unlink(path)


if __name__ == "__main__":
    unittest.main(verbosity=2)
