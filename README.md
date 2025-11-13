# Reconstruction-of-PDE-without-Time-Label

Official repository of "BlinDNO: A Distributional Neural Operator for Time-Label-Free Dynamical System Reconstruction"


## Overview
This repository focuses on reconstructing partial differential equations (PDEs) from time-label-free dynamical system data using the BlinDNO (Distributional Neural Operator) model. It supports experiments on various PDEs, including:
- 1D/2D Gross-Pitaevskii Equation (GPE)
- 1D/2D Fokker-Planck Equation (FPE)
- 2D Non-conservative Force FPE
- 3D Protein

## Key Components
- **Model Implementations**: Includes BlinDNO and baseline models (e.g., NIO, NIO-FNO) for PDE reconstruction.
- **Data Generation**: Scripts to generate training/testing data for different PDEs (e.g., GPE, FPE).
- **Training Pipelines**: Per-PDE training scripts (e.g., `train_GPE_xxx.py` in `1d_GPE` directory).
- **Evaluation Tools**: 
  - Model evaluation scripts (`eval_fno.py`, `eval_nio.py`) for predicting drift/diffusion fields or forces (Fx, Fy).
  - Time error analysis with `compute_time_error.py` to evaluate density propagation accuracy.

## Usage

### 1. Clone the Repository
```bash
git clone https://github.com/yl602019618/Reconstruction-of-PDE-without-Time-Label.git
cd Reconstruction-of-PDE-without-Time-Label
```

### 2. Install Dependencies
Requires PyTorch, NumPy, Matplotlib, and other common scientific libraries (see code for details).

### 3. Training
Run training scripts for specific PDEs. Example for 1D GPE:
```bash
cd 1d_GPE
python train_GPE_xxx.py
```

### 4. Evaluation
Evaluate pre-trained models using evaluation scripts. Example for 2D FPE with FNO:
```bash
cd 2d_FPE
python eval_fno.py 
```



## License
This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.


