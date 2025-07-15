import torch
import numpy as np
import h5py
from monai.networks.layers import HilbertTransform
from scipy.interpolate import interp1d
import os

class JointTOFCProcessor:
    def __init__(self, fs=31250000):
        self.fs = fs
        self.probe_geometry = np.linspace(-0.019105, 0.019105, 128)

    def time_to_space_mapping(self, rf_data):
        c = 1540
        time_vector = np.arange(rf_data.shape[0]) / self.fs
        x_axis = np.linspace(self.probe_geometry[0], self.probe_geometry[-1], rf_data.shape[1])
        z_axis = 0.5 * c * time_vector
        x_grid, z_grid = np.meshgrid(x_axis, z_axis)
        pixels = z_grid.shape[0] * z_grid.shape[1]
        tofc_data = np.zeros((pixels, rf_data.shape[1]), dtype=np.float32)

        for nrx in range(rf_data.shape[1]):
            receive_delay = np.sqrt((self.probe_geometry[nrx] - x_grid.ravel()) ** 2 + z_grid.ravel() ** 2)
            total_delay = (z_grid.ravel() / c) + (receive_delay / c)
            interp_func = interp1d(time_vector, rf_data[:, nrx], kind='cubic', fill_value=0, bounds_error=False)
            tofc_data[:, nrx] = interp_func(total_delay)

        result = tofc_data.reshape(z_grid.shape[0], z_grid.shape[1], rf_data.shape[1])
        return np.nan_to_num(result, nan=0.0, posinf=1.0, neginf=-1.0)

    def compute_global_stats(self, tofc_data):
        global_mean = np.mean(tofc_data)
        global_std = np.std(tofc_data)
        if global_std == 0:
            global_std = 1.0
        return global_mean, global_std

    def normalize_input(self, data, global_mean, global_std):
        return (data - global_mean) / (global_std + 1e-8)

    def hilbert_log_compress(self, img, epsilon=1e-6):
        analytic = HilbertTransform(axis=0)(img)
        envelope = torch.abs(analytic)
        envelope_max = torch.max(envelope)

        if envelope_max > 1e-10:
            envelope_log = 20 * torch.log10(torch.clamp(envelope / envelope_max, min=1e-8))
            envelope_min, envelope_max = torch.min(envelope_log), torch.max(envelope_log)
            if envelope_max > envelope_min:
                log_img_norm = (envelope_log - envelope_min) / (envelope_max - envelope_min)
            else:
                log_img_norm = torch.zeros_like(envelope_log)
        else:
            log_img_norm = torch.zeros_like(envelope)

        log_img_norm -= log_img_norm.max()
        return log_img_norm

    def process_file(self, file_path, patch_rows, model, device):
        with h5py.File(file_path, "r") as f:
            inp = f['raw_data'][:]
            gt_das = f['bf_data_DAS_real'][:] * f.attrs['scale_DAS_real']
            gt_fdmas = f['bf_data_FDMAS_real'][:] * f.attrs['scale_FDMAS_real']
            gt_capon = f['bf_data_Capon_real'][:] * f.attrs['scale_Capon_real']

        idata = np.transpose(inp[0], (1, 0))
        tofc = self.time_to_space_mapping(idata)
        global_mean, global_std = self.compute_global_stats(tofc)

        total_depth = tofc.shape[0]
        num_patches = (total_depth + patch_rows - 1) // patch_rows

        pred_das, pred_fdmas, pred_capon = [], [], []
        gt_das_p, gt_fdmas_p, gt_capon_p = [], [], []

        for i in range(num_patches):
            start_row = i * patch_rows
            end_row = min((i + 1) * patch_rows, total_depth)
            current_patch_rows = end_row - start_row

            input_patch = tofc[start_row:end_row]
            gt_patch_d = gt_das[start_row:end_row]
            gt_patch_f = gt_fdmas[start_row:end_row]
            gt_patch_c = gt_capon[start_row:end_row]

            if current_patch_rows < patch_rows:
                pad = patch_rows - current_patch_rows
                input_patch = np.pad(input_patch, ((0, pad), (0, 0), (0, 0)), mode='constant')
                gt_patch_d = np.pad(gt_patch_d, ((0, pad), (0, 0)), mode='constant')
                gt_patch_f = np.pad(gt_patch_f, ((0, pad), (0, 0)), mode='constant')
                gt_patch_c = np.pad(gt_patch_c, ((0, pad), (0, 0)), mode='constant')

            input_np = self.normalize_input(input_patch.transpose(2, 0, 1), global_mean, global_std)
            input_tensor = torch.from_numpy(input_np).unsqueeze(0).float().to(device)

            with torch.no_grad():
                outputs = model(input_tensor)

            pred_das.append(outputs['das'].squeeze().cpu().numpy())
            pred_fdmas.append(outputs['fdmas'].squeeze().cpu().numpy())
            pred_capon.append(outputs['capon'].squeeze().cpu().numpy())

            gt_das_p.append(gt_patch_d)
            gt_fdmas_p.append(gt_patch_f)
            gt_capon_p.append(gt_patch_c)

        return {
            'pred_das': np.vstack(pred_das),
            'pred_fdmas': np.vstack(pred_fdmas),
            'pred_capon': np.vstack(pred_capon),
            'gt_das': np.vstack(gt_das_p).T,
            'gt_fdmas': np.vstack(gt_fdmas_p).T,
            'gt_capon': np.vstack(gt_capon_p).T
        }
