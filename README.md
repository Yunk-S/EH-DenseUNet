# Enhanced H-DenseUNet: Advanced Liver and Tumor Segmentation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Weights & Biases](https://img.shields.io/badge/Weights_&_Biases-FFCC33?logo=WeightsAndBiases&logoColor=black)](https://wandb.ai/)
[![arXiv](https://img.shields.io/badge/arXiv-1709.07330-b31b1b.svg)](https://arxiv.org/abs/1709.07330)

This project is only intended for understanding and efficiency optimization based on the H-DenseUNet method. An **enhanced implementation** of the H-DenseUNet paper, integrating state-of-the-art deep learning optimization techniques for high-precision automatic liver and tumor segmentation from non-contrast CT scans. This project implements the original 2D-3D hybrid architecture and incorporates cutting-edge techniques including attention mechanisms, progressive training strategies, and advanced data augmentation.

**Paper References**: 
- Original Paper: Li, Xiaomeng, et al. "H-DenseUNet: hybrid densely connected UNet for liver and tumor segmentation from CT volumes." *IEEE TMI* 37.12 (2018): 2663-2674.
- GitHub References: [xmengli/H-DenseUNet](https://github.com/xmengli/H-DenseUNet) | [asif-jc/H-DenseUNet-CNN](https://github.com/asif-jc/H-DenseUNet-CNN)

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Key Features](#key-features)
3. [Requirements & Installation](#requirements--installation)
4. [Data Preparation](#data-preparation)
5. [Model Architecture](#model-architecture)
6. [Training](#training)
7. [Usage](#usage)
8. [Contributing](#contributing)
9. [Citation](#citation)

---

## Project Overview

A state-of-the-art deep learning framework for automatic liver and tumor segmentation from CT scans. The project implements a hybrid 2D-3D architecture with advanced optimization techniques including attention mechanisms, progressive training, and memory optimization.

---

## Key Features

### Model Architecture
- **2D DenseUNet**: Deep supervision, SE attention, multi-scale features
- **3D DenseUNet**: Memory-efficient convolutions, adaptive pooling
- **Hybrid Feature Fusion**: Cross-modal attention, feature pyramid networks

### Training Optimizations
- Progressive multi-stage training strategy
- Mixed precision training and gradient checkpointing
- Advanced data augmentation (MixUp, CutMix, elastic deformation)
- AdamW optimizer with cosine annealing scheduler

---

## Requirements & Installation

### Requirements
- Python 3.8+, PyTorch 1.10+, CUDA 11.1+
- NVIDIA GPU with ≥8GB VRAM
- Dependencies: see `requirements.txt`

### Quick Setup
```bash
git clone https://github.com/Yunk-S/EH-DenseUNet.git
cd Enhanced-H-DenseUNet
pip install -r requirements.txt
```

---

## Data Preparation

Supports LiTS Challenge and 3DIRCADb datasets. Organize data as:
```
data/
├── data_Tr/imagesTr/    # Training CT scans (.nii.gz)
├── data_Tr/labelsTr/    # Training masks (0: background, 1: liver, 2: tumor)
└── data_Ts/imagesTs/    # Test CT scans
```

Preprocessing:
```bash
python preprocessing.py --input_dir data_Tr/imagesTr --output_dir preprocessed_data
```

---

## Model Architecture

Hybrid 2D-3D architecture combining:
- **2D DenseUNet**: Extracts intra-slice features with SE attention and deep supervision
- **3D DenseUNet**: Processes volumetric context with memory-efficient convolutions
- **Hybrid Feature Fusion**: Combines 2D/3D features using cross-modal attention

---

## Training

Three-stage progressive training:
1. **2D Pre-training** (50 epochs): Learn intra-slice features with deep supervision
2. **3D + HFF Training** (30 epochs): Learn volumetric context with frozen 2D weights
3. **End-to-End Fine-tuning** (20 epochs): Joint optimization of all components

Uses combined loss (CE + Focal + Boundary), AdamW optimizer with cosine annealing, and mixed precision training.

---

## Usage

### Training
```bash
# Basic training
python train_optimized.py --config config.yaml

# Stage-by-stage training
python train_optimized.py --stage 1 --config config.yaml  # 2D pre-training
python train_optimized.py --stage 2 --config config.yaml  # 3D + HFF
python train_optimized.py --stage 3 --config config.yaml  # Fine-tuning
```

### Inference
```bash
python inference.py --model hdenseunet_final.pth
```

### Evaluation
Evaluates using standard metrics: Dice coefficient, IoU, Hausdorff distance, and volume similarity.
```bash
python evaluate.py --pred_dir output/ --gt_dir data/data_Ts/labelsTs/
```

---

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

---

## Acknowledgments

- Original H-DenseUNet paper by Li, Xiaomeng, et al.
- LiTS Challenge for providing the benchmark dataset
- Medical imaging and PyTorch communities


---

## Citation

If you use this work in your research, please cite:

```bibtex
@article{li2018h,
  title={H-DenseUNet: hybrid densely connected UNet for liver and tumor segmentation from CT volumes},
  author={Li, Xiaomeng and Chen, Hao and Qi, Xiaojuan and Dou, Qi and Fu, Chi-Wing and Heng, Pheng-Ann},
  journal={IEEE transactions on medical imaging},
  volume={37},
  number={12},
  pages={2663--2674},
  year={2018},
  publisher={IEEE}
}
```
