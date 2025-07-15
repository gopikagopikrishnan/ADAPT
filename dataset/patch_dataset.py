import torch
from torch.utils.data import Dataset
import numpy as np
import h5py
import random
from pathlib import Path
from scipy.interpolate import interp1d
from scipy.signal import decimate

class FixedCustomDatasetTriBeamNet(Dataset):
    def __init__(self, folder_path: str, patch_rows: int = 128,
                 seed: int = 42, max_files: int = 1000,
                 fs: int = 31250000, probe_geometry: np.ndarray = None):
        super().__init__()
        self.folder_path = Path(folder_path)
        self.file_paths = sorted([f for f in self.folder_path.glob("*.h5")])[:max_files]
        self.rows = patch_rows
        self.fs = fs
        self.probe_geometry = probe_geometry if probe_geometry is not None else np.linspace(-0.019105, 0.019105, 128)
        random.seed(seed)

        if not self.file_paths:
            raise ValueError(f"No HDF5 files found in {folder_path}")

        print(f"Dataset initialized with {len(self.file_paths)} files")
        self.global_mean = 0.0
        self.global_std = 1.0
        self.compute_global_stats()

    def compute_global_stats(self):
        all_values = []
        sample_size = min(10, len(self.file_paths))
        for path in self.file_paths[:sample_size]:
            try:
                with h5py.File(path, 'r') as f:
                    raw_rf = f['raw_data'][:][0].T
                    tofc = self.tofc_mapping(raw_rf)
                    tofc = np.nan_to_num(tofc, nan=0.0, posinf=1.0, neginf=-1.0)
                    all_values.append(tofc.flatten())
            except Exception as e:
                print(f"Error processing {path}: {e}")

        if all_values:
            all_values = np.concatenate(all_values)
            self.global_mean = np.median(all_values)
            self.global_std = np.std(all_values)
            if self.global_std == 0:
                self.global_std = 1.0
            print(f"Global stats - Mean: {self.global_mean:.4f}, Std: {self.global_std:.4f}")

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        try:
            with h5py.File(path, 'r') as f:
                raw_rf = f['raw_data'][:][0].T
                gt_bf_datas = []
                for key in ['DAS', 'FDMAS', 'Capon']:
                    try:
                        bf_data = f[f'bf_data_{key}_real'][:].astype(np.float32)
                        scale = f.attrs[f'scale_{key}_real']
                        bf_data *= scale
                        bf_data = np.nan_to_num(bf_data, nan=0.0, posinf=1.0, neginf=-1.0)
                        if bf_data.shape[1] == 4352:
                            bf_data = decimate(bf_data, q=2, axis=1)
                        gt_bf_datas.append(bf_data)
                    except Exception as e:
                        print(f"Missing {key} in {path.name}, filling with zeros. Error: {e}")
                        gt_bf_datas.append(np.zeros((128, 2176), dtype=np.float32))

                tofc = self.tofc_mapping(raw_rf)
                tofc = np.nan_to_num(tofc, nan=0.0, posinf=1.0, neginf=-1.0)

                valid_rows = tofc.shape[0]
                max_start = max(0, valid_rows - self.rows)
                start = np.random.randint(0, max_start + 1) if max_start > 0 else 0
                end = start + self.rows

                input_patch = tofc[start:end, :, :]
                gt_patches = [gt[:, start:end].T for gt in gt_bf_datas]

                if input_patch.shape[0] != self.rows:
                    pad = self.rows - input_patch.shape[0]
                    input_patch = np.pad(input_patch, ((0, pad), (0, 0), (0, 0)), mode='constant')
                    gt_patches = [np.pad(p, ((0, pad), (0, 0)), mode='constant') for p in gt_patches]

                input_patch = self.normalize_input(input_patch)
                gt_patches = [self.normalize_target(p) for p in gt_patches]

                input_tensor = torch.from_numpy(input_patch.transpose(2, 0, 1)).float()
                output_tensor = torch.from_numpy(np.stack(gt_patches, axis=0)).float()

                return {'input': input_tensor, 'output': output_tensor}

        except Exception as e:
            print(f"Error loading file {path}: {e}")
            return {'input': torch.randn(128, self.rows, 128), 'output': torch.randn(3, self.rows, 128)}

    def normalize_input(self, data):
        return (data - self.global_mean) / (self.global_std + 1e-8)

    def normalize_target(self, data):
        mean, std = np.mean(data), np.std(data)
        if std == 0: std = 1.0
        return (data - mean) / (std + 1e-8)

    def tofc_mapping(self, rf_data):
        try:
            c = 1540
            time = np.arange(rf_data.shape[0]) / self.fs
            x = np.linspace(self.probe_geometry[0], self.probe_geometry[-1], rf_data.shape[1])
            z = 0.5 * c * time
            xg, zg = np.meshgrid(x, z)
            x_flat, z_flat = xg.ravel(), zg.ravel()
            delay_map = np.zeros((rf_data.shape[0] * rf_data.shape[1], rf_data.shape[1]))
            for i in range(rf_data.shape[1]):
                rx_delay = np.sqrt((self.probe_geometry[i] - x_flat) ** 2 + z_flat ** 2)
                total_delay = (z_flat + rx_delay) / c
                interp = interp1d(time, rf_data[:, i], kind='cubic', fill_value=0, bounds_error=False)
                delay_map[:, i] = interp(total_delay)
            return delay_map.reshape(rf_data.shape[0], rf_data.shape[1], rf_data.shape[1])
        except Exception as e:
            print(f"TOFC mapping failed: {e}")
            return np.zeros((rf_data.shape[0], rf_data.shape[1], rf_data.shape[1]))
