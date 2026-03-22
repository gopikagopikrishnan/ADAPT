import torch
from torch.utils.data import Dataset
import numpy as np
import h5py
import random
from pathlib import Path
from scipy.interpolate import interp1d
from scipy.signal import decimate
from concurrent.futures import ThreadPoolExecutor, as_completed

def tofc_mapping(self, rf_data):
    try:
        c = 1540
        num_samples, num_channels = rf_data.shape

        # Compute time and spatial grid
        time = np.arange(num_samples) / self.fs
        x = np.linspace(self.probe_geometry[0], self.probe_geometry[-1], num_channels)
        z = 0.5 * c * time
        xg, zg = np.meshgrid(x, z)
        x_flat, z_flat = xg.ravel(), zg.ravel()

        def interpolate_channel(i):
            rx_delay = np.sqrt((self.probe_geometry[i] - x_flat) ** 2 + z_flat ** 2)
            total_delay = (z_flat + rx_delay) / c
            interp = interp1d(time, rf_data[:, i], kind='cubic', fill_value=0.0, bounds_error=False)
            return interp(total_delay)

        # Parallelize using threads (scipy interp1d releases GIL)
        delay_map = np.zeros((num_samples * num_channels, num_channels))
        with ThreadPoolExecutor(max_workers=12) as executor:  # adjust num workers
            futures = {executor.submit(interpolate_channel, i): i for i in range(num_channels)}
            for future in as_completed(futures):
                i = futures[future]
                delay_map[:, i] = future.result()

        return delay_map.reshape(num_samples, num_channels, num_channels)
