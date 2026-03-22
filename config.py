"""
configs/config.py

Single source of truth for all acquisition parameters, dataset conventions and training hyper-parameters used across ADAPT.
"""

# Acquisition
FS         = 31_250_000   # Sampling frequency  [Hz]
FC         =  7_812_500   # Centre frequency     [Hz]
C          = 1540          # Speed of sound       [m/s]
N_ELEMENTS = 128           # Number of probe elements
DEPTH_M    = 0.06          # Maximum imaging depth [m]

# HDF5 dataset key convention
#  root datasets : raw_data, bmode_das, bmode_fdmas, bmode_capon,
#                  probe_geometry, tx_delays, scan_x, scan_z
#  per-dataset   : bmode_*.attrs['scale_factor']   (float; recover = int16 * sf)
#  root attrs    : probe_name (str), fs (float)
HDF5_KEYS = {
    "raw":    "raw_data",
    "das":    "bmode_das",
    "fdmas":  "bmode_fdmas",
    "capon":  "bmode_capon",
}

# B-mode dynamic range
#  ultraspy.to_b_mode outputs are log-compressed [dB].
#  GT images are clipped to [DB_MIN, DB_MAX] and normalised to [0, 1].
DB_MIN = -60.0
DB_MAX =   0.0

# Dataset / patching
PATCH_ROWS = 128           # Axial samples per training patch
SPLIT      = 0.8           # Train / validation split ratio
SEED       = 42

# Training
LR              = 1e-3
WEIGHT_DECAY    = 1e-4
SCHEDULER_PAT   = 5
SCHEDULER_FACTOR= 0.5
GRAD_CLIP       = 1.0
