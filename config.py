"""
config.py

Central configuration for ADAPT: Adaptive Depth-Agnostic Patch-wise Tunable-multibeamformer.

Update the PATH constants to your local checkpoint / data locations before running any script.
"""

import numpy as np
import torch

# Model checkpoints 
DAS_PATH   = "checkpoints/Results_DAS/best.pt"
FDMAS_PATH = "checkpoints/Results_FDMAS/best.pt"
CAPON_PATH = "checkpoints/Results_Capon/best.pt"

# Data files
FILE_PATH_RESOLUTION = "data/simulation_resolution_distorsion.h5"
FILE_PATH_CONTRAST   = "data/simulation_contrast_speckle.h5"
FILE_PATH_EXPE       = "data/experiments_resolution_distorsion.h5"

# Fusion weights (must sum to 1)
W_DAS   = 0.33
W_FDMAS = 0.33
W_CAPON = 0.34

# Ultrasound system parameters
FS             = 31.25e6         # Sampling frequency [Hz]
C              = 1540.0              # Speed of sound [m/s]
N_ELEMENTS     = 128
PROBE_GEOMETRY = np.linspace(-0.019105, 0.019105, N_ELEMENTS).astype(np.float32)

# Patch-based inference
PATCH_ROWS = 128                     # Rows per spatial patch

# Pin locations (from MATLAB, row-col in Python order)
SELECTED_PINS = [
    (269, 64), (406, 65), (545, 65), (670, 65), (808, 65),
    (949, 65), (1091, 64), (1211, 64), (547, 15), (542, 31),
    (540, 48), (542, 81), (547, 98), (545, 114), (1093, 14),
    (1082, 31), (1088, 48), (1081, 81), (1081, 97), (1091, 114),
]

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
