#!/usr/bin/env python3
"""
Model compression and performance optimization utilities.

This module provides:
  - Model quantization (dynamic and static)
  - Model pruning (magnitude-based structured pruning)
  - Inference speed optimization
  - Memory usage optimization
  - Performance benchmarking

These utilities help reduce model size and improve inference speed for
deployment on resource-constrained environments (mobile, edge, serverless).
"""
from __future__ import annotations

import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False
    torch = None
    nn = None


# ---------------------------------------------------------------------------
# Model quantization
# ---------------------------------------------------------------------------

def quantize_model_dynamic(model: nn.Module, dtype: Any = None) -> nn.Module:
    """Apply dynamic quantization to a model.

    Dynamic quantization converts float32 weights to int8 (or float16)
    at runtime, reducing memory usage and improving CPU inference speed.
    It works best for Linear and LSTM layers.

    Args:
        model: the PyTorch model to quantize
        dtype: torch.qint8 (default) or torch.float16

    Returns:
        Quantized model
    """
    if not _HAS_TORCH:
        raise RuntimeError("PyTorch is required for quantization")

    if dtype is None:
        dtype = torch.qint8

    quantized = torch.quantization.quantize_dynamic(
        model, {nn.Linear, nn.LSTM, nn.LSTMCell}, dtype=dtype
    )
    return quantized


def quantize_model_static(model: nn.Module, calibration_data: List[torch.Tensor],
                          backend: str = 'fbgemm') -> nn.Module:
    """Apply static quantization with calibration data.

    Static quantization requires representative input data to determine
    optimal quantization parameters. It provides better speedup than
    dynamic quantization but requires calibration.

    Args:
        model: the PyTorch model to quantize
        calibration_data: list of example input tensors
        backend: 'fbgemm' (x86) or 'qnnpack' (ARM)

    Returns:
        Quantized model
    """
    if not _HAS_TORCH:
        raise RuntimeError("PyTorch is required for quantization")

    # Set backend
    torch.backends.quantized.engine = backend

    # Prepare model for quantization
    model.eval()
    model.qconfig = torch.quantization.get_default_qconfig(backend)
    prepared = torch.quantization.prepare(model)

    # Calibrate with example data
    with torch.no_grad():
        for data in calibration_data:
            prepared(data)

    # Convert to quantized model
    quantized = torch.quantization.convert(prepared)
    return quantized


# ---------------------------------------------------------------------------
# Model pruning
# ---------------------------------------------------------------------------

def prune_model_magnitude(model: nn.Module, amount: float = 0.3) -> nn.Module:
    """Apply magnitude-based pruning to Linear layers.

    Prunes the smallest weights (by absolute value) in each Linear layer,
    setting them to zero. This reduces effective model size without
    changing the architecture.

    Args:
        model: the PyTorch model to prune
        amount: fraction of weights to prune (0.0 to 1.0)

    Returns:
        Pruned model (modified in-place)
    """
    if not _HAS_TORCH:
        raise RuntimeError("PyTorch is required for pruning")

    try:
        from torch.nn.utils import prune
    except ImportError:
        raise RuntimeError("torch.nn.utils.prune not available")

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            prune.l1_unstructured(module, name='weight', amount=amount)
            # Make pruning permanent
            prune.remove(module, 'weight')

    return model


def prune_model_structured(model: nn.Module, amount: float = 0.3) -> nn.Module:
    """Apply structured pruning to remove entire neurons/channels.

    Structured pruning removes entire rows/columns of weight matrices,
    which provides actual speedup (unlike unstructured pruning).

    Args:
        model: the PyTorch model to prune
        amount: fraction of structures to prune

    Returns:
        Pruned model
    """
    if not _HAS_TORCH:
        raise RuntimeError("PyTorch is required for pruning")

    try:
        from torch.nn.utils import prune
    except ImportError:
        raise RuntimeError("torch.nn.utils.prune not available")

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # Prune entire output neurons
            prune.ln_structured(module, name='weight', amount=amount, n=2, dim=0)
            prune.remove(module, 'weight')

    return model


# ---------------------------------------------------------------------------
# Performance benchmarking
# ---------------------------------------------------------------------------

def benchmark_model(model: nn.Module, input_fn: Any, num_runs: int = 100,
                    warmup: int = 10) -> Dict[str, float]:
    """Benchmark model inference speed.

    Args:
        model: the PyTorch model to benchmark
        input_fn: callable that returns input tensors
        num_runs: number of timed runs
        warmup: number of warmup runs (not timed)

    Returns:
        Dict with timing statistics (ms)
    """
    if not _HAS_TORCH:
        raise RuntimeError("PyTorch is required for benchmarking")

    model.eval()

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            inputs = input_fn()
            if isinstance(inputs, (list, tuple)):
                model(*inputs)
            else:
                model(inputs)

    # Timed runs
    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            inputs = input_fn()
            t0 = time.perf_counter()
            if isinstance(inputs, (list, tuple)):
                model(*inputs)
            else:
                model(inputs)
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)  # ms

    times = np.array(times)
    return {
        'mean_ms': float(times.mean()),
        'std_ms': float(times.std()),
        'min_ms': float(times.min()),
        'max_ms': float(times.max()),
        'p50_ms': float(np.percentile(times, 50)),
        'p95_ms': float(np.percentile(times, 95)),
        'num_runs': num_runs,
    }


def measure_model_size(model: nn.Module) -> Dict[str, Any]:
    """Measure model size and parameter count.

    Returns:
        Dict with size statistics
    """
    if not _HAS_TORCH:
        raise RuntimeError("PyTorch is required")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Estimate size in MB (float32 = 4 bytes)
    param_size = total_params * 4 / (1024 * 1024)

    # Buffer size
    buffer_size = sum(b.numel() for b in model.buffers()) * 4 / (1024 * 1024)

    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'param_size_mb': round(param_size, 3),
        'buffer_size_mb': round(buffer_size, 3),
        'total_size_mb': round(param_size + buffer_size, 3),
    }


# ---------------------------------------------------------------------------
# Inference optimization
# ---------------------------------------------------------------------------

def optimize_for_inference(model: nn.Module) -> nn.Module:
    """Optimize a model for inference.

    Applies several optimizations:
      - Sets model to eval mode
      - Disables gradient computation
      - Fuses operations where possible (via JIT)

    Returns:
        Optimized model
    """
    if not _HAS_TORCH:
        raise RuntimeError("PyTorch is required")

    model.eval()

    # Disable gradients
    for param in model.parameters():
        param.requires_grad = False

    return model


def jit_compile_model(model: nn.Module, example_inputs: Any) -> Any:
    """JIT compile a model for faster inference.

    Args:
        model: the PyTorch model
        example_inputs: example inputs for tracing

    Returns:
        JIT-compiled model (torch.jit.ScriptModule)
    """
    if not _HAS_TORCH:
        raise RuntimeError("PyTorch is required for JIT compilation")

    model.eval()
    with torch.no_grad():
        if isinstance(example_inputs, (list, tuple)):
            traced = torch.jit.trace(model, example_inputs)
        else:
            traced = torch.jit.trace(model, example_inputs)
    return traced


# ---------------------------------------------------------------------------
# Memory optimization
# ---------------------------------------------------------------------------

def get_memory_usage() -> Dict[str, float]:
    """Get current memory usage.

    Returns:
        Dict with memory statistics in MB
    """
    if not _HAS_TORCH:
        return {'allocated_mb': 0, 'cached_mb': 0}

    if torch.cuda.is_available():
        return {
            'allocated_mb': torch.cuda.memory_allocated() / (1024 * 1024),
            'cached_mb': torch.cuda.memory_reserved() / (1024 * 1024),
        }
    else:
        # CPU memory (approximate via garbage collector)
        import gc
        import psutil
        process = psutil.Process(os.getpid())
        mem = process.memory_info().rss / (1024 * 1024)
        return {'allocated_mb': mem, 'cached_mb': 0}


def clear_cache():
    """Clear PyTorch cache to free memory."""
    if _HAS_TORCH and torch.cuda.is_available():
        torch.cuda.empty_cache()
    import gc
    gc.collect()


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_compression_pipeline(quantize: bool = True, prune_amount: float = 0.0,
                               jit: bool = False) -> Dict[str, Any]:
    """Build a compression pipeline configuration.

    Returns:
        Dict with pipeline configuration
    """
    return {
        'quantize': quantize,
        'prune_amount': prune_amount,
        'jit_compile': jit,
        'steps': []
    }


def apply_compression_pipeline(model: nn.Module, config: Dict[str, Any],
                               calibration_data: Optional[List] = None) -> nn.Module:
    """Apply a compression pipeline to a model.

    Args:
        model: the PyTorch model
        config: pipeline configuration from build_compression_pipeline
        calibration_data: optional calibration data for static quantization

    Returns:
        Compressed model
    """
    if not _HAS_TORCH:
        raise RuntimeError("PyTorch is required")

    # Step 1: Pruning
    if config.get('prune_amount', 0) > 0:
        model = prune_model_magnitude(model, amount=config['prune_amount'])
        config['steps'].append(f"pruned({config['prune_amount']})")

    # Step 2: Quantization
    if config.get('quantize', False):
        if calibration_data:
            model = quantize_model_static(model, calibration_data)
            config['steps'].append("quantized(static)")
        else:
            model = quantize_model_dynamic(model)
            config['steps'].append("quantized(dynamic)")

    # Step 3: JIT compilation
    if config.get('jit_compile', False) and calibration_data:
        model = jit_compile_model(model, calibration_data[0])
        config['steps'].append("jit_compiled")

    return model
