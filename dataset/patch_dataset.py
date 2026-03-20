"""
dataset.py

Data-loading utilities for HDF5 ultrasound datasets and GPU-accelerated Time-of-Flight Correction (ToFC) via CuPy.

HDF5 layout expected:
/data               – raw RF data,  scale_factor attribute required
/DAS                – DAS B-mode GT, scale_factor attribute required
/FDMAS              – FDMAS B-mode GT, scale_factor attribute required
/Capon              – Capon B-mode GT, scale_factor attribute required
attrs['fs']         – sampling frequency (optional, falls back to config.FS)
"""

from __future__ import annotations

import numpy as np
import h5py
import cupy as cp


# ToFC - GPU based (CuPy)

def tofc_mapping(rf_data: np.ndarray, fs: float, probe_geometry: np.ndarray, c: float, ) -> np.ndarray:
    """Time-of-Flight Correction (ToFC) using CuPy (GPU).

    Parameters
    ----------
    rf_data : (num_samples, num_channels) float array
        Normalised RF channel data.
    fs : float
        Sampling frequency [Hz].
    probe_geometry : (num_channels,) float array
        Element lateral positions [m].
    c : float
        Speed of sound [m/s].

    Returns
    -------
    np.ndarray of shape (num_samples, num_channels, num_channels)
        ToFC output on CPU.
    """
    rf_np = np.asarray(rf_data)
    num_samples, num_channels = rf_np.shape
    x_np = np.asarray(probe_geometry, dtype=np.float64).reshape(-1)

    rf = cp.asarray(rf_np, dtype=cp.float64, order="C")
    x  = cp.asarray(x_np, dtype=cp.float64)
    t  = cp.arange(num_samples, dtype=cp.float64) / float(fs)
    z  = 0.5 * c * t

    xg, zg   = cp.meshgrid(x, z, indexing="xy")
    x_flat   = xg.reshape(-1)
    z_flat   = zg.reshape(-1)

    dx = x_flat[:, None] - x[None, :]
    rx = cp.sqrt(dx * dx + z_flat[:, None] ** 2)
    td = (z_flat[:, None] + rx) / c

    idx_right = cp.searchsorted(t, td, side="right")
    idx_left  = cp.clip(idx_right - 1, 0, num_samples - 2)
    idx_right = idx_left + 1

    t_left   = t[idx_left]
    t_right  = t[idx_right]
    w_interp = (td - t_left) / cp.maximum(t_right - t_left, 1e-6)

    col_idx  = cp.arange(num_channels, dtype=cp.int64)[None, :]
    rf_left  = rf[idx_left,  col_idx]
    rf_right = rf[idx_right, col_idx]
    y = (1.0 - w_interp) * rf_left + w_interp * rf_right

    y[(td < t[0]) | (td > t[-1])] = 0.0
    return cp.asnumpy(y.reshape(num_samples, num_channels, num_channels))


# ── Normalisation helpers ──────────────────────────────────────────────────────

def compute_global_stats(rf_data: np.ndarray) -> tuple[float, float]:
    """Return (mean, std) of flattened RF data.  std is clipped to 1e-8."""
    flat = rf_data.flatten()
    mean = float(np.mean(flat))
    std  = float(np.std(flat))
    std  = max(std, 1e-8)
    return mean, std


def normalize_rf(rf_data: np.ndarray, mean: float, std: float) -> np.ndarray:
    """z-score normalisation with a small epsilon guard."""
    return (rf_data - mean) / (std + 1e-8)


# h5 file loader

def load_h5_sample(
    file_path: str,
    fallback_fs: float,
) -> dict:
    """Load a single HDF5 data file.

    Returns a dict with keys:
        idata      – (num_samples, num_channels) normalised RF
        gt_das     – ground-truth DAS B-mode
        gt_fdmas   – ground-truth FDMAS B-mode
        gt_capon   – ground-truth Capon B-mode
        fs         – actual sampling frequency
    """
    with h5py.File(file_path, "r") as f:
        inp    = np.array(f["data"],  dtype="float32") / f["data"].attrs["scale_factor"]
        idata  = np.transpose(inp, (1, 0))             # → (samples, channels)

        gt_das   = np.array(f["DAS"],   dtype="float32") / f["DAS"].attrs["scale_factor"]
        gt_fdmas = np.array(f["FDMAS"], dtype="float32") / f["FDMAS"].attrs["scale_factor"]
        gt_capon = np.array(f["Capon"], dtype="float32") / f["Capon"].attrs["scale_factor"]

        fs_attr = f.attrs.get("fs")
        fs      = float(fs_attr) if fs_attr is not None else fallback_fs

    return dict(idata=idata, gt_das=gt_das, gt_fdmas=gt_fdmas, gt_capon=gt_capon, fs=fs)


def prepare_tofc_tensor(
    idata: np.ndarray,
    fs: float,
    probe_geometry: np.ndarray,
    c: float,
) -> np.ndarray:
    """Normalise RF data and apply ToFC.  Returns (C, H, W) float32 array."""
    mean, std = compute_global_stats(idata)
    norm_rf   = normalize_rf(idata, mean, std)
    tofc      = tofc_mapping(norm_rf, fs=fs, probe_geometry=probe_geometry, c=c)
    return np.nan_to_num(tofc.transpose(2, 0, 1)).astype(np.float32)
