# ADAPT: Adaptive Depth-Agnostic Patch-wise Tunable Multibeamformer

Official implementation of **"ADAPT: Multibeamformer with Tunable Weight Fusion and Patch-wise Learning for Ultrasound Imaging"** (ISBI 2026).

## Abstract
**ADAPT** is a deep learning framework for adaptive ultrasound beamforming that achieves depth-independent learning through patch-wise processing. By utilizing **parameter-space fusion**, ADAPT enables the dynamic blending of DAS, FDMAS, and Capon beamformers without retraining, providing clinicians with tunable, task-specific diagnostic perspectives in a single forward pass.

---

## Key Features
* **Depth-Agnostic Processing:** Axial patching (128 samples/patch) ensures generalization across varying imaging depths.
* **AntiRectifier Architecture:** Preserves gradient flow for both polarities of raw RF data by concatenating $ReLU(x)$ and $ReLU(-x)$.
* **Tunable Weight Fusion:** Post-training parameter averaging allows for dynamic blending:
    $$\theta_{fused} = \alpha \cdot \theta_{DAS} + \beta \cdot \theta_{FDMAS} + \gamma \cdot \theta_{Capon}$$
* **GPU Acceleration:** Vectorized bilinear interpolation and ToFC powered by **CuPy**.

---

## Methodology & Architecture

### Training Pipeline
The framework trains three identical U-Nets independently for DAS, FDMAS, and Capon tasks. The patch-wise strategy allows the model to learn local wavefront characteristics regardless of the absolute depth.

![Training Strategy](isbi_training.png)
*Figure 1: Overview of the ADAPT training workflow and patch-wise data augmentation.*

### Inference & Fusion
Instead of output ensembling, ADAPT performs fusion in the weight space. This significantly reduces computational overhead during inference while allowing real-time tuning of image characteristics.

![Inference and Weight Fusion](isbi_inference.png)
*Figure 2: The weight fusion mechanism enabling tunable diagnostic perspectives.*

---

## 📂 Repository Structure
```text
ADAPT/
├── configs/          # Hyperparameters (fs, fc, c, patch size)
├── preprocess/       # RF to HDF5 conversion scripts
├── data/             # Dataset loaders (ToFC + patching)
├── model/            # FixedUNetBeamformer & AntiRectifier
├── train/            # Task-specific training loops
├── inference/        # Parameter-space weight fusion logic
└── requirements.txt
