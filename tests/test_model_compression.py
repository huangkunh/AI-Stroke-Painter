#!/usr/bin/env python3
"""
Tests for model compression utilities.

Run with:
    python -m unittest tests.test_model_compression
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


class TestModelCompression(unittest.TestCase):
    """Tests for model/model_compression.py."""

    @classmethod
    def setUpClass(cls):
        try:
            import torch
            import torch.nn as nn
            cls.torch = torch
            cls.nn = nn
        except ImportError:
            raise unittest.SkipTest("PyTorch not available")

        from model_compression import (
            quantize_model_dynamic, prune_model_magnitude,
            prune_model_structured, benchmark_model, measure_model_size,
            optimize_for_inference, get_memory_usage, clear_cache,
            build_compression_pipeline, apply_compression_pipeline
        )
        cls.quantize_dynamic = staticmethod(quantize_model_dynamic)
        cls.prune_magnitude = staticmethod(prune_model_magnitude)
        cls.prune_structured = staticmethod(prune_model_structured)
        cls.benchmark = staticmethod(benchmark_model)
        cls.measure_size = staticmethod(measure_model_size)
        cls.optimize_inference = staticmethod(optimize_for_inference)
        cls.get_memory = staticmethod(get_memory_usage)
        cls.clear_cache = staticmethod(clear_cache)
        cls.build_pipeline = staticmethod(build_compression_pipeline)
        cls.apply_pipeline = staticmethod(apply_compression_pipeline)

        # Create a simple test model
        class TestModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(64, 128)
                self.fc2 = nn.Linear(128, 64)

            def forward(self, x):
                x = self.fc1(x)
                x = self.fc2(x)
                return x

        cls.TestModel = TestModel

    def test_quantize_dynamic(self):
        """Dynamic quantization should produce a working model."""
        model = self.TestModel()
        quantized = self.quantize_dynamic(model)
        x = self.torch.randn(1, 64)
        out = quantized(x)
        self.assertEqual(out.shape, (1, 64))

    def test_prune_magnitude(self):
        """Magnitude pruning should zero out some weights."""
        model = self.TestModel()
        original_nonzero = (model.fc1.weight != 0).sum().item()
        pruned = self.prune_magnitude(model, amount=0.3)
        pruned_nonzero = (pruned.fc1.weight != 0).sum().item()
        self.assertLess(pruned_nonzero, original_nonzero)

    def test_prune_structured(self):
        """Structured pruning should work without error."""
        model = self.TestModel()
        pruned = self.prune_structured(model, amount=0.2)
        x = self.torch.randn(1, 64)
        out = pruned(x)
        self.assertEqual(out.shape, (1, 64))

    def test_benchmark_model(self):
        """Benchmark should return timing statistics."""
        model = self.TestModel()
        input_fn = lambda: self.torch.randn(1, 64)
        stats = self.benchmark(model, input_fn, num_runs=10, warmup=2)
        self.assertIn('mean_ms', stats)
        self.assertIn('std_ms', stats)
        self.assertGreater(stats['mean_ms'], 0)
        self.assertEqual(stats['num_runs'], 10)

    def test_measure_model_size(self):
        """Model size measurement should return parameter count."""
        model = self.TestModel()
        stats = self.measure_size(model)
        self.assertIn('total_params', stats)
        self.assertIn('param_size_mb', stats)
        # fc1: 64*128 + 128 = 8320, fc2: 128*64 + 64 = 8256
        self.assertEqual(stats['total_params'], 8320 + 8256)

    def test_optimize_for_inference(self):
        """Inference optimization should set eval mode and disable gradients."""
        model = self.TestModel()
        model.train()
        optimized = self.optimize_inference(model)
        self.assertFalse(optimized.training)
        for param in optimized.parameters():
            self.assertFalse(param.requires_grad)

    def test_get_memory_usage(self):
        """Memory usage should return a dict with allocated_mb."""
        mem = self.get_memory()
        self.assertIn('allocated_mb', mem)
        self.assertGreaterEqual(mem['allocated_mb'], 0)

    def test_clear_cache(self):
        """Clear cache should not raise."""
        self.clear_cache()

    def test_compression_pipeline(self):
        """Compression pipeline should apply multiple steps."""
        model = self.TestModel()
        config = self.build_pipeline(quantize=True, prune_amount=0.2)
        compressed = self.apply_pipeline(model, config)
        self.assertGreater(len(config['steps']), 0)
        x = self.torch.randn(1, 64)
        out = compressed(x)
        self.assertEqual(out.shape, (1, 64))


if __name__ == "__main__":
    unittest.main(verbosity=2)
