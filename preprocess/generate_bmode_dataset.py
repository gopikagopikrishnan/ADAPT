"""
preprocess/generate_bmode_dataset.py

Reads raw RF wrist acquisitions from the Zenodo dataset (requires permission to access)
(https://zenodo.org/records/7813791) and writes processed HDF5 files containing DAS, FDMAS, and Capon B-mode images.

HDF5 output convention (canonical for ADAPT):
  Datasets
  ├── raw_data        float32  [1, N_ELEMENTS, n_samples]
  ├── bmode_das       int16    [n_z, n_x]   .attrs['scale_factor']
  ├── bmode_fdmas     int16    [n_z, n_x]   .attrs['scale_factor']
  ├── bmode_capon     int16    [n_z, n_x]   .attrs['scale_factor']
  ├── probe_geometry  float32  [2, N_ELEMENTS]
  ├── tx_delays       float32  [n_tx, N_ELEMENTS]
  ├── scan_x          float32  [n_x]
  └── scan_z          float32  [n_z]
  Root attrs
  ├── probe_name      str
  └── fs              float

  Recovery: bmode_float = bmode_int16.astype(float32) * scale_factor
  Shape convention: axis-0 = depth (z), axis-1 = lateral (x).
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import h5py
import psutil
import cupy as cp
from ultraspy.probes.factory import get_probe
from ultraspy.beamformers.das import DelayAndSum
from ultraspy.beamformers.fdmas import FilteredDelayMultiplyAndSum
from ultraspy.beamformers.capon import Capon
from ultraspy.scan import GridScan

from configs.config import FS, FC, C, N_ELEMENTS, DEPTH_M

# User config.
INPUT_DIR    = "/content/wrist_data/wrist_data"        # Zenodo download path
OUTPUT_DIR   = "/content/bmode_dataset_ADAPT"
N_FILES      = 1000
RAM_WARN_GB  = 11

def check_ram(threshold_gb: float = RAM_WARN_GB) -> None:
    used = psutil.virtual_memory().used / (1024 ** 3)
    if used > threshold_gb:
        print(f"[WARNING] RAM: {used:.2f} GB > {threshold_gb} GB threshold")


def beamform_to_bmode(
    beamformer,
    data: np.ndarray,
    probe,
    transmit_delays: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run ultraspy beamforming and return (bmode [n_z, n_x], x, z).

    The returned B-mode image is log-compressed (dB) with depth on axis-0
    and lateral on axis-1: shape = (n_z, n_x).
    """
    elements = np.arange(probe.nb_elements)
    n_tx     = transmit_delays.shape[0]

    acq = {
        "sampling_freq":   FS,
        "t0":              0,
        "prf":             1000,
        "signal_duration": None,
        "delays":          transmit_delays,
        "sound_speed":     C,
        "sequence_elements": {
            "emitted":  np.tile(elements, (n_tx, 1)),
            "received": np.tile(elements, (n_tx, 1)),
        },
    }

    beamformer.automatic_setup(acq, probe)
    beamformer.update_setup("f_number", 1.75)
    beamformer.update_option("reduction",             "sum")
    beamformer.update_option("rx_apodization",        "boxcar")
    beamformer.update_option("rx_apodization_alpha",  "0.5")
    beamformer.update_option("compound",              "True")

    n_x  = N_ELEMENTS
    n_z  = data.shape[-1]
    x    = np.linspace(probe.geometry[0, 0], probe.geometry[0, -1], n_x)
    z    = np.linspace(0, DEPTH_M, n_z)
    scan = GridScan(x, z)

    d_data     = cp.asarray(data, dtype=cp.float32)
    d_output   = beamformer.beamform(d_data, scan)
    d_envelope = beamformer.compute_envelope(d_output, scan)
    beamformer.to_b_mode(d_envelope, scan)   # in-place log-compression
    bmode_raw  = d_envelope.get()

    # Ensure depth-first convention: (n_z, n_x).
    # ultraspy GridScan(x, z) with x as lateral axis typically returns (n_x, n_z).
    if bmode_raw.shape == (n_x, n_z):
        bmode_raw = bmode_raw.T                  # → (n_z, n_x)
    # If shape is already (n_z, n_x), no transpose needed.
    assert bmode_raw.shape == (n_z, n_x), (
        f"Unexpected b-mode shape {bmode_raw.shape}; expected ({n_z}, {n_x})"
    )

    return np.real(bmode_raw), x, z


def safe_scale_and_cast(
    array: np.ndarray,
    label: str,
) -> tuple[np.ndarray, float]:
    """Sanitise NaN/Inf, scale to int16.  Returns (int16_array, scale_factor).

    Recovery: float_array = int16_array.astype(float32) * scale_factor
    """
    array = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)
    scale = float(np.max(np.abs(array)))

    if scale == 0.0 or not np.isfinite(scale):
        print(f"[WARNING] Zero/invalid scale for '{label}' — filling with zeros.")
        return np.zeros_like(array, dtype=np.int16), 1.0

    scale_factor = scale / np.iinfo(np.int16).max
    scaled       = np.nan_to_num(array / scale_factor, nan=0.0, posinf=0.0, neginf=0.0)
    return scaled.astype(np.int16), scale_factor


# Main loop

os.makedirs(OUTPUT_DIR, exist_ok=True)

for i in range(1, N_FILES + 1):
    src = os.path.join(INPUT_DIR,  f"data_{i}.h5")
    dst = os.path.join(OUTPUT_DIR, f"data_{i}.h5")

    try:
        check_ram()

        # Load RF
        with h5py.File(src, "r") as f:
            raw_data        = np.array(f["data"], dtype=np.float32)   # [128, n_z]
            raw_data        = np.moveaxis(raw_data, [0, 1], [1, 0])   # [n_z, 128]
            params          = dict(f["/data"].attrs.items())
            probe_name      = params["probe_name"]
            transmit_delays = params["transmit_delays"]
            if transmit_delays.ndim < 2:
                transmit_delays = np.expand_dims(transmit_delays, axis=0)

        probe = get_probe(probe_name)
        probe.set_central_freq(FC)

        # ultraspy expects [n_tx, n_elements, n_samples]
        data = np.expand_dims(                          # [1, 128, n_z]
            np.moveaxis(raw_data, [1, 0], [0, 1]), axis=0
        )

        # Beamform
        bmode_das,   x, z = beamform_to_bmode(DelayAndSum(),                 data, probe, transmit_delays)
        bmode_fdmas, x, z = beamform_to_bmode(FilteredDelayMultiplyAndSum(), data, probe, transmit_delays)
        bmode_capon, x, z = beamform_to_bmode(Capon(),                       data, probe, transmit_delays)

        # Scale to int16
        das_i16,   sf_das   = safe_scale_and_cast(bmode_das,   "DAS")
        fdmas_i16, sf_fdmas = safe_scale_and_cast(bmode_fdmas, "FDMAS")
        capon_i16, sf_capon = safe_scale_and_cast(bmode_capon, "Capon")

        # Write HDF5 
        with h5py.File(dst, "w") as hf:
            # Raw input
            ds = hf.create_dataset("raw_data", data=data.astype(np.float32), compression="gzip")

            # B-mode images: each carries its own scale_factor attribute
            for key, arr, sf in [
                ("bmode_das",   das_i16,   sf_das),
                ("bmode_fdmas", fdmas_i16, sf_fdmas),
                ("bmode_capon", capon_i16, sf_capon),
            ]:
                ds = hf.create_dataset(key, data=arr, compression="gzip")
                ds.attrs["scale_factor"] = sf

            # Geometry / metadata
            hf.create_dataset("probe_geometry", data=probe.geometry.astype(np.float32))
            hf.create_dataset("tx_delays",      data=transmit_delays.astype(np.float32))
            hf.create_dataset("scan_x",         data=x.astype(np.float32))
            hf.create_dataset("scan_z",         data=z.astype(np.float32))
            hf.attrs["probe_name"] = probe_name
            hf.attrs["fs"]         = float(FS)

        print(f"[INFO]  {i:>4d}/{N_FILES}  Saved: data_{i}.h5")

    except Exception as exc:
        print(f"[ERROR] data_{i}.h5 — {exc}")
