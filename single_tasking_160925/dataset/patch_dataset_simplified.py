import torch
from torch.utils.data import Dataset
import numpy as np
import h5py
import random
from pathlib import Path
from scipy.interpolate import interp1d
import os, random, h5py, torch
from pathlib import Path
from torch.utils.data import Dataset
import cupy as cp

class FixedCustomDatasetTriBeamNet(Dataset):
    def __init__(self, folder_path: str, patch_rows: int = 128,
                 seed: int = 42, max_files: int = 100,
                 fs: int = 31250000, probe_geometry: np.ndarray = None,
                 save_stats_path: str = "global_stats.npz"):   # <--- NEW
        super().__init__()
        self.folder_path = Path(folder_path)
        self.file_paths = sorted([f for f in self.folder_path.glob("*.h5")])[:max_files]
        self.rows = patch_rows
        self.fs = fs
        self.probe_geometry = probe_geometry if probe_geometry is not None else np.linspace(-0.019105, 0.019105, 128)
        random.seed(seed)

        if not self.file_paths:
            raise ValueError(f"No HDF5 files found in {folder_path}")

        # Compute global statistics once (NEW VERSION)
        self.global_mean, self.global_std = self.compute_global_rf_stats()
        self.gt_min, self.gt_max = self.compute_global_gt_minmax()

        print(f"Global RF stats - Mean: {self.global_mean:.4f}, Std: {self.global_std:.4f}")
        print(f"Global GT min: {self.gt_min:.4f}, max: {self.gt_max:.4f}")

        # --- SAVE STATS TO DISK ---
        np.savez(save_stats_path,
                 rf_mean=self.global_mean,
                 rf_std=self.global_std,
                 gt_min=self.gt_min,
                 gt_max=self.gt_max)
        print(f"Saved global stats to {os.path.abspath(save_stats_path)}")

    def compute_global_rf_stats(self):
        """Compute global mean/std from ALL RF data by flattening everything"""
        all_vals = []
        for path in self.file_paths:
            with h5py.File(path, "r") as f:
                rf = f['raw_data'][:][0].T.astype(np.float32).flatten()
                all_vals.append(rf)
        all_vals = np.concatenate(all_vals)
        global_mean = np.mean(all_vals)
        global_std = np.std(all_vals)
        if global_std == 0:
            global_std = 1e-8
        return float(global_mean), float(global_std)

    def compute_global_gt_minmax(self):
        """Compute global min/max from all GT B-mode images"""
        gmin, gmax = float("inf"), float("-inf")
        for path in self.file_paths:
            with h5py.File(path, "r") as f:
                if 'bmode_Capon_DR' in f:
                    arr = f['bmode_Capon_DR'][:].astype(np.float32)
                elif 'bmode_Capon' in f:
                    arr = f['bmode_Capon'][:].astype(np.float32)
                else:
                    continue
                gmin = min(gmin, float(np.nanmin(arr)))
                gmax = max(gmax, float(np.nanmax(arr)))
        if not np.isfinite(gmin): gmin = 0.0
        if not np.isfinite(gmax): gmax = gmin + 1.0
        return gmin, gmax

    def normalize_rf(self, rf):
        return (rf - self.global_mean) / (self.global_std + 1e-8)

    def normalize_gt(self, gt):
        return (gt - self.gt_min) / (self.gt_max - self.gt_min + 1e-8)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        with h5py.File(path, 'r') as f:
            raw_rf = f['raw_data'][:][0].T
            rf_norm = self.normalize_rf(raw_rf)

            # GT
            if 'bmode_Capon_DR' in f:
                gt = f['bmode_Capon_DR'][:].astype(np.float32)
            elif 'bmode_Capon' in f:
                gt = f['bmode_Capon'][:].astype(np.float32)
            else:
                gt = np.zeros((128, 2176), dtype=np.float32)

            gt = np.nan_to_num(gt, nan=0.0, posinf=0.0, neginf=0.0)
            gt_norm = self.normalize_gt(gt)

            # ToFC mapping – full depth
            tofc = self.tofc_mapping(rf_norm, self.fs, self.probe_geometry, c=1540.0)
            cap = tofc.shape[0]
            tofc = tofc[:cap, :, :]
            gt_norm = gt_norm[:, :cap]

            # Random patch
            max_start = max(0, cap - self.rows)
            start = np.random.randint(0, max_start + 1) if max_start > 0 else 0
            end = min(start + self.rows, cap)
            rows = end - start

            in_patch = tofc[start:end, :, :]
            gt_patch = gt_norm[:, start:end].T

            # Pad if needed
            if rows < self.rows:
                pad = self.rows - rows
                in_patch = np.pad(in_patch, ((0, pad), (0, 0), (0, 0)), mode='constant')
                gt_patch = np.pad(gt_patch, ((0, pad), (0, 0)), mode='constant')

            in_patch = in_patch[:rows, :, :]
            gt_patch = gt_patch[:rows, :]

            input_tensor = torch.from_numpy(in_patch.transpose(2, 0, 1)).float()
            output_tensor = torch.from_numpy(gt_patch).float()

        return {"input": input_tensor, "output": output_tensor}

    def tofc_mapping(self, rf_data, fs, probe_geometry, c=1540.0):
        rf_np = np.asarray(rf_data)
        num_samples, num_channels = rf_np.shape
        x_np = np.asarray(probe_geometry, dtype=np.float32).reshape(-1)
        rf = cp.asarray(rf_np, dtype=cp.float32)
        x = cp.asarray(x_np, dtype=cp.float32)
        t = cp.arange(num_samples, dtype=cp.float32) / float(fs)
        z = 0.5 * c * t
        xg, zg = cp.meshgrid(x, z, indexing="xy")
        x_flat, z_flat = xg.reshape(-1), zg.reshape(-1)
        dx = (x_flat[:, None] - x[None, :])
        rx = cp.sqrt(dx * dx + (z_flat[:, None] ** 2))
        td = (z_flat[:, None] + rx) / c
        idx_right = cp.searchsorted(t, td, side="right")
        idx_left = cp.clip(idx_right - 1, 0, num_samples - 2)
        idx_right = idx_left + 1
        t_left, t_right = t[idx_left], t[idx_right]
        w = (td - t_left) / cp.maximum(t_right - t_left, 1e-6)
        col_idx = cp.arange(num_channels, dtype=cp.int32)[None, :]
        rf_left, rf_right = rf[idx_left, col_idx], rf[idx_right, col_idx]
        y = (1.0 - w) * rf_left + w * rf_right
        y[(td < t[0]) | (td > t[-1])] = 0.0
        return cp.asnumpy(y.reshape(num_samples, num_channels, num_channels))
