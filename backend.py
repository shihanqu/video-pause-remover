"""Array-compute backend selection: MLX (Apple Metal, or its CUDA build),
CuPy (NVIDIA CUDA), or NumPy (CPU fallback).

All backends implement process_chunk() with the same numerics, so analysis
results - and the on-disk cache - are backend-independent and shared.
Selection order: any GPU (MLX device gpu, then CuPy with a CUDA device),
then MLX CPU, then NumPy. Override with PAUSE_REMOVER_BACKEND=mlx|cupy|numpy.
"""

from __future__ import annotations

import os

import numpy as np


def _estimate_gate(d_host: np.ndarray, floor: float) -> float:
    """Per-pixel noise floor from a sample of |frame deltas| (MAD-based)."""
    sample = d_host.ravel()[::13]
    sigma = 1.4826 * float(np.median(np.abs(sample)))
    return max(floor, 4.0 * sigma)


class MLXBackend:
    name = "mlx"

    def __init__(self):
        import mlx.core as mx
        self.mx = mx
        self.device = str(mx.default_device())  # "Device(gpu, 0)" on Metal and CUDA builds

    def process_chunk(self, chunk: np.ndarray, gate: float | None, gate_floor: float,
                      ty: int, tx: int, tile: int):
        mx = self.mx
        x = mx.array(chunk).astype(mx.float32) / 255.0
        x = x - x.mean(axis=(1, 2), keepdims=True)
        d = mx.abs(x[1:] - x[:-1])
        if gate is None:
            gate = _estimate_gate(np.array(d), gate_floor)
        dt = d.reshape(d.shape[0], ty, tile, tx, tile)
        t_mean = dt.mean(axis=(2, 4)).astype(mx.float16)
        t_frac = (dt > gate).astype(mx.float32).mean(axis=(2, 4)).astype(mx.float16)
        mx.eval(t_mean, t_frac)
        return np.array(t_mean), np.array(t_frac), gate


class XPBackend:
    """NumPy and CuPy share this code path - CuPy is NumPy-API compatible,
    so the CUDA kernel is literally the tested NumPy kernel on device arrays."""

    def __init__(self, xp, name: str, device: str, to_host):
        self.xp, self.name, self.device, self._host = xp, name, device, to_host

    def process_chunk(self, chunk: np.ndarray, gate: float | None, gate_floor: float,
                      ty: int, tx: int, tile: int):
        xp = self.xp
        x = xp.asarray(chunk).astype(xp.float32) / 255.0
        x = x - x.mean(axis=(1, 2), keepdims=True)
        d = xp.abs(x[1:] - x[:-1])
        if gate is None:
            gate = _estimate_gate(self._host(d), gate_floor)
        dt = d.reshape(d.shape[0], ty, tile, tx, tile)
        t_mean = dt.mean(axis=(2, 4)).astype(xp.float16)
        t_frac = (dt > gate).astype(xp.float32).mean(axis=(2, 4)).astype(xp.float16)
        return self._host(t_mean), self._host(t_frac), gate


def _try_mlx() -> MLXBackend | None:
    try:
        return MLXBackend()
    except Exception:
        return None


def _try_cupy() -> XPBackend | None:
    try:
        import cupy as cp
        if cp.cuda.runtime.getDeviceCount() < 1:
            return None
        props = cp.cuda.runtime.getDeviceProperties(0)
        gpu = props["name"]
        gpu = gpu.decode() if isinstance(gpu, bytes) else str(gpu)
        return XPBackend(cp, "cupy", f"cuda ({gpu})", cp.asnumpy)
    except Exception:
        return None


def _numpy() -> XPBackend:
    return XPBackend(np, "numpy", "cpu", np.asarray)


def pick():
    forced = (os.environ.get("PAUSE_REMOVER_BACKEND") or "").lower()
    if forced:
        b = {"mlx": _try_mlx, "cupy": _try_cupy, "numpy": _numpy, "cpu": _numpy}.get(forced, lambda: None)()
        if b is None:
            raise RuntimeError(f"PAUSE_REMOVER_BACKEND={forced} is not available on this machine")
        return b
    b = _try_mlx()
    if b and "gpu" in b.device.lower():
        return b
    c = _try_cupy()
    if c:
        return c
    return b or _numpy()
