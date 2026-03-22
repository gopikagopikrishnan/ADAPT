"""
data/dataset.py

PyTorch Dataset for ADAPT.  Loads ADAPT-format HDF5 files produced by preprocess/generate_bmode_dataset.py and returns depth patches ready for task-specific training.

Each sample:
  input  : (N_ELEMENTS, PATCH_ROWS, N_ELEMENTS) float32 — ToFC tensor
  output : (PATCH_ROWS, N_ELEMENTS)              float32 — GT b-mode in [0, 1]

ToFC is precomputed per file on first access and cached in memory (safe for
num_workers=0; set TOFC_CACHE_SIZE to limit RAM use with large datasets).

HDF5 key contract (see configs/config.py and preprocess/generate_bmode_dataset.py):
  bmode_das / bmode_fdmas / bmode_capon  — int16, .attrs['scale_factor']
  raw_data                               — float32 [1, N_ELEMENTS, n_z]
  root attrs: fs (float)
"""

from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from pathlib import Path
from collections import OrderedDict
from typing import Literal

import numpy as np
import h5py
import cupy as cp
import torch
from torch.utils.data import Dataset

from configs.config import (
    FS, C, N_ELEMENTS, PATCH_ROWS, DB_MIN, DB_MAX, HDF5_KEYS
)

TOFC_CACHE_SIZE = 50   # max number of files whose ToFC tensors are kept in RAM


# GPU Time-of-Flight Correction (CuPy)

def tofc_mapping(
    rf_data:       np.ndarray,   # (n_z, N_ELEMENTS) normalised RF
    fs:            float,
    probe_geometry: np.ndarray,  # (N_ELEMENTS,) lateral positions [m]
    c:             float,
) -> np.ndarray:
    """Time-of-Flight Correction via GPU-vectorised bilinear interpolation.

    Returns (n_z, N_ELEMENTS, N_ELEMENTS) on CPU, float64.
    """
    num_samples, num_channels = rf_data.shape
    x_np = np.asarray(probe_geometry, dtype=np.float64).ravel()

    rf = cp.asarray(rf_data, dtype=cp.float64, order="C")
    x  = cp.asarray(x_np,   dtype=cp.float64)
    t  = cp.arange(num_samples, dtype=cp.float64) / float(fs)
    z  = 0.5 * c * t

    xg, zg  = cp.meshgrid(x, z, indexing="xy")
    x_flat  = xg.ravel()
    z_flat  = zg.ravel()

    dx       = x_flat[:, None] - x[None, :]
    rx       = cp.sqrt(dx * dx + z_flat[:, None] ** 2)
    td       = (z_flat[:, None] + rx) / c

    ir       = cp.searchsorted(t, td, side="right")
    il       = cp.clip(ir - 1, 0, num_samples - 2)
    ir       = il + 1

    t_l      = t[il];  t_r = t[ir]
    w        = (td - t_l) / cp.maximum(t_r - t_l, 1e-6)

    ch_idx   = cp.arange(num_channels, dtype=cp.int64)[None, :]
    y        = (1.0 - w) * rf[il, ch_idx] + w * rf[ir, ch_idx]
    y[(td < t[0]) | (td > t[-1])] = 0.0

    return cp.asnumpy(y.reshape(num_samples, num_channels, num_channels))


# Normalisation helpers

def zscore_normalize(arr: np.ndarray) -> np.ndarray:
    mean = float(np.mean(arr))
    std  = max(float(np.std(arr)), 1e-8)
    return (arr - mean) / (std + 1e-8)


def bmode_to_unit(bmode_db: np.ndarray) -> np.ndarray:
    """Clip to [DB_MIN, DB_MAX] dB and map to [0, 1]."""
    clipped = np.clip(bmode_db, DB_MIN, DB_MAX)
    return (clipped - DB_MIN) / (DB_MAX - DB_MIN)


# Per-file I/O

def _load_rf_and_gt(
    fpath:  Path,
    task:   str,
    fallback_fs: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return (rf [n_z, N_ELEMENTS], gt_bmode_unit [n_z, N_ELEMENTS], fs)."""
    with h5py.File(fpath, "r") as f:
        # Raw RF: stored as [1, N_ELEMENTS, n_z] → reshape to [n_z, N_ELEMENTS]
        raw   = np.array(f[HDF5_KEYS["raw"]], dtype=np.float32)  # [1, 128, n_z]
        rf    = raw[0].T                                          # [n_z, 128]

        # GT b-mode
        key   = HDF5_KEYS[task]
        sf    = float(f[key].attrs["scale_factor"])
        gt_i16 = np.array(f[key], dtype=np.float32)              # [n_z, n_x]
        gt_db  = gt_i16 * sf                                      # recover dB

        fs_val = float(f.attrs["fs"]) if "fs" in f.attrs else fallback_fs

    gt_unit = bmode_to_unit(gt_db)   # [0, 1]
    return rf, gt_unit, fs_val


def _compute_tofc(
    rf:             np.ndarray,   # [n_z, N_ELEMENTS]
    fs:             float,
    probe_geometry: np.ndarray,
    c:              float,
) -> np.ndarray:
    """z-score normalise RF then apply ToFC.

    Returns (N_ELEMENTS, n_z, N_ELEMENTS) float32 — channel-first, ready as
    model input after depth patching on axis-1.
    """
    norm_rf = zscore_normalize(rf)
    tofc    = tofc_mapping(norm_rf, fs=fs,
                           probe_geometry=probe_geometry, c=c)  # (n_z, 128, 128)
    tofc    = np.nan_to_num(tofc.transpose(1, 0, 2), nan=0.0)   # (128, n_z, 128)
    return tofc.astype(np.float32)


# Dataset

class ADAPTDataset(Dataset):
    """Patch-wise dataset for task-specific ADAPT training.

    Parameters
    
    folder_path :
        Directory containing ADAPT-format ``data_{i}.h5`` files.
    task :
        One of ``'das'``, ``'fdmas'``, ``'capon'``.
    probe_geometry :
        1-D array of element lateral positions [m], shape (N_ELEMENTS,).
    fs :
        Sampling frequency fallback [Hz] if not stored in file.
    c :
        Speed of sound [m/s].
    patch_rows :
        Number of axial samples per patch (default 128).
    """

    VALID_TASKS: tuple[str, ...] = ("das", "fdmas", "capon")

    def __init__(
        self,
        folder_path:    str,
        task:           Literal["das", "fdmas", "capon"],
        probe_geometry: np.ndarray,
        fs:             float = FS,
        c:              float = C,
        patch_rows:     int   = PATCH_ROWS,
    ) -> None:
        if task not in self.VALID_TASKS:
            raise ValueError(f"task must be one of {self.VALID_TASKS}, got '{task}'")

        self.task           = task
        self.probe_geometry = np.asarray(probe_geometry, dtype=np.float64).ravel()
        self.fs             = fs
        self.c              = c
        self.patch_rows     = patch_rows

        self.files = sorted(Path(folder_path).glob("data_*.h5"))
        if not self.files:
            raise FileNotFoundError(f"No data_*.h5 files found in {folder_path}")

        # Build flat patch index: list of (file_idx, patch_start_row)
        self._index: list[tuple[int, int]] = []
        for file_idx, fpath in enumerate(self.files):
            with h5py.File(fpath, "r") as f:
                n_z = f[HDF5_KEYS[task]].shape[0]   # depth dimension
            n_patches = n_z // patch_rows            # discard incomplete tail
            for p in range(n_patches):
                self._index.append((file_idx, p * patch_rows))

        # LRU ToFC cache: {file_idx: (tofc [128, n_z, 128], gt [n_z, 128])}
        self._cache: OrderedDict[int, tuple[np.ndarray, np.ndarray]] = OrderedDict()

    # Internal caching

    def _get_file_tensors(self, file_idx: int) -> tuple[np.ndarray, np.ndarray]:
        """Return (tofc, gt_unit) for the given file, computing on first hit."""
        if file_idx in self._cache:
            self._cache.move_to_end(file_idx)
            return self._cache[file_idx]

        # Evict oldest entry if cache is full
        if len(self._cache) >= TOFC_CACHE_SIZE:
            self._cache.popitem(last=False)

        fpath = self.files[file_idx]
        rf, gt_unit, fs_file = _load_rf_and_gt(fpath, self.task, self.fs)
        tofc  = _compute_tofc(rf, fs_file, self.probe_geometry, self.c)

        self._cache[file_idx] = (tofc, gt_unit)
        self._cache.move_to_end(file_idx)
        return tofc, gt_unit

    # Dataset interface

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        file_idx, row_start = self._index[idx]
        tofc, gt_unit       = self._get_file_tensors(file_idx)

        row_end      = row_start + self.patch_rows
        tofc_patch   = tofc[:, row_start:row_end, :]              # [128, patch_rows, 128]
        gt_patch     = gt_unit[row_start:row_end, :]              # [patch_rows, 128]

        return {
            "input":  torch.from_numpy(tofc_patch),               # float32
            "output": torch.from_numpy(gt_patch.astype(np.float32)),
        }
