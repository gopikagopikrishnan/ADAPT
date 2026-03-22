import torch
from torch.utils.data import Dataset
import numpy as np
import h5py
import random
from pathlib import Path
from scipy.interpolate import interp1d
from scipy.signal import decimate

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
