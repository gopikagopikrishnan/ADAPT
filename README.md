# TriBeamNet
# :x_ray: Multi-Beamforming Model for Ultrasound Imaging

TriBeamNet is a U-Net-based deep learning model that predicts apodization weights to generate three beamformed outputs: DAS, FDMAS, and Capon, from raw RF ultrasound data.

## Features
- Raw RF wrist data → ToFC mapping → U-Net + Beamforming head prediction
- Multi-head prediction for DAS, FDMAS, Capon
- Patch-wise training
- Hilbert envelope and log compression for output images


