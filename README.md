# ADAPT: Adaptive Depth-Agnostic Patch-wise Tunable multibeamformer

Official implementation of "ADAPT: Multibeamformer with Tunable Weight Fusion and Patch-wise Learning for Ultrasound Imaging" (ISBI 2026).

## Abstract

ADAPT is a deep learning framework for adaptive ultrasound beamforming that achieves depth-independent learning through patch-wise processing and combines multiple beamforming algorithms via parameter-space fusion. The framework enables dynamic blending of DAS, FDMAS, and MV beamformers without retraining, providing clinicians with tunable diagnostic perspectives.

## Key Features

- **Depth-Agnostic Processing**: Axial patching strategy (128 samples/patch) enables generalization across imaging depths
- **Algorithm-Specific Training**: Three identical U-Nets independently trained for DAS, FDMAS, and MV beamforming
- **Tunable Weight Fusion**: Post-training parameter averaging allows dynamic task-specific optimization
- **Computational Efficiency**: 73% faster training, 94% reduction in FLOPs and activation memory vs. full-depth processing
- **CuPy-Accelerated ToFC**: GPU-optimized time-of-flight correction with vectorized interpolation

## Architecture
```
θ_fused = α·θ_DAS + β·θ_FDMAS + γ·θ_MV  (α + β + γ = 1)
```

The framework consists of:
- Modified U-Net with hybrid activation (Antirectifier + ReLU)
- Task head with channel-wise softmax for unity-sum apodization weights
- Bias-free convolutions with batch normalization
- Kaiming He initialization for stable gradient flow

## Requirements

- Python 3.8+
- PyTorch 2.0+
- CuPy 11.0+
- NumPy, SciPy
- ultraspy (for dataset preprocessing)
- MONAI (for beamforming postprocessing to b-mode image)

## Installation
```bash
git clone https://github.com/gopikagopikrishnan-fedus/ADAPT
cd ADAPT
pip install -r requirements.txt
```

## Dataset

**Training**: 1000 wrist acquisitions from healthy volunteers
- Verasonics Vantage 128 with L11-5v probe
- Center frequency: 7.6 MHz, Sampling frequency: 31.25 MHz
- Single plane-wave transmission at 0°

**Evaluation**: PICMUS dataset
- Verasonics Vantage 256 with L11-4v probe
- Center frequency: 5.208 MHz, Sampling frequency: 20.832 MHz
- In-silico, in-vitro, and in-vivo phantoms

## Training
```python
# Training specifications - individual beamformer networks:
# - Loss: SSIM between Predcited and Ground Truth B-mode Images
# - Optimizer: AdamW (with Kaiming He weight initialization)
# - Scheduler: ReduceLROnPlateau (patience=5, factor=0.5)
# - Mixed precision training enabled
# - Train/validation split: 80/20
```

## Citation
```bibtex
@inproceedings{gopikrishnan2026adapt,
  title={ADAPT: Multibeamformer with Tunable Weight Fusion and Patch-wise Learning for Ultrasound Imaging},
  author={Gopikrishnan, Gopika and Panicker, Mahesh Raveendranatha and Liu, Timothy and Beng, Ng Aik and See, Simon Chong-Wee},
  booktitle={IEEE International Symposium on Biomedical Imaging (ISBI)},
  year={2026}
}
```

## Acknowledgments

This research is supported by the Ministry of Education, Singapore, under the Academic Research Fund Tier 1 (GMS 1052). Dataset collection was conducted at Indian Institute of Technology Palakkad.

## Ethics

Human wrist data acquired following the Helsinki Declaration of 1975 (revised 2000).

## Contact

For more information, contact 
Gopika Gopikrishnan -gopikagopikrishnan@gmail.com


