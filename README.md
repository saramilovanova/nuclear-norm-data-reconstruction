# Nuclear Norm Minimization for Data Reconstruction

This repository contains the implementation and experimental evaluation for the Master's thesis:

**"Nuclear Norm Minimization for Data Reconstruction"**

The project investigates classical and modern methods for recovering missing or corrupted data, with a focus on low-rank and sparse modeling approaches.

## Methods

The following methods are implemented and evaluated:

- k-SVD
- Singular Value Thresholding (SVT)
- Optimal SVT (Gavish & Donoho)
- Deep k-SVD
- Learned SVT
- Jacobian Nuclear Norm Regularization

## Experimental Setup

Each method is evaluated on two types of data:

### 1. Image Data (CBSD68)
- Reconstruction with missing pixels (20%, 40%, 60%)
- Denoising with additive Gaussian noise

### 2. Recommender System Data (Netflix Prize subset)
- Matrix completion (missing entries)
- Noise robustness experiments

### Evaluation Metrics
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- Peak Signal-to-Noise Ratio (PSNR)

## Repository Structure

- `src/` – core algorithm implementations
- `experiments/` – scripts for running experiments
- `data/`
- `results/` – experiment outputs
- `figures/` – plots used in thesis

## Environment Setup

Create the conda environment:

```bash
conda env create -f environment.yml
conda activate nuclear-norm-thesis
```

## References

Key references used in this work include:

- Aharon, Elad, Bruckstein (2006) – k-SVD 
- Cai, Candès, Shen (2010) – Singular Value Thresholding (adapted from the original MATLAB code by Candès and Becker, available at https://candes.su.domains/software/svt/code.html)
- Gavish & Donoho (2017) – Optimal Shrinkage
- Scetbon, Elad, Milanfar (2021) – Deep k-SVD
- Shanmugam & Kalyani (2023) – Learned SVT
- Scarvelis & Solomon (2024) – Jacobian Nuclear Norm Regularization
