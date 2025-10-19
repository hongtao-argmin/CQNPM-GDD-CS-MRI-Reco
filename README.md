# PyTorch Implementation of Randomized Nyström Preconditioners

**Reference**  
[Tao Hong](https://hongtao-argmin.github.io), Zhaoyi Xu, Jason Hu, and [Jeffrey A. Fessler](https://web.eecs.umich.edu/~fessler/), “Using Randomized Nyström Preconditioners to Accelerate Variational Image Reconstruction,” *to appear in* **IEEE Transactions on Computational Imaging**, 2025.  
Preprint: [https://arxiv.org/abs/2411.08178](https://arxiv.org/abs/2411.08178)

---

## Getting Started

### 1) Installation

Create a conda environment using the provided YAML file:

```bash
conda env create -f RandomizedNystromPre.yml
conda activate RandomizedNystromPre
```

---

### 2) Repository Structure & Demos

The repository contains two main folders for experiments:

**1. `lplq` folder – Impulsive-noise (lp–lq) reconstruction demos**  
- `DemoLpLqRecoTV.py`: demo for total variation (TV) regularization  
- `DemoLpLqRecoTVSketchSize.py`: demo to study the effect of sketch size  
- `OptAlgLpLq.py`: contains the optimization algorithm implementations  
- `utilities.py`: support and helper functions  

**2. `CT` folder – Computed tomography (CT) reconstruction demos**  
- `Demo2DCTl2RecoTV.py`, `Demo2DCTl2RecoWav.py`, `Demo2DCTl2RecoHS.py`: demos for TV, wavelet, and Hessian–Schatten regularizers  
- `CTutilities.py`, `utilities.py`, `utilitiesHS.py`: helper functions for CT reconstruction  
- `operatorTorch` folder and `operator2.py`: operators implementing different CT geometries (using the [ODL library](https://github.com/odlgroup/odl))  
- **Data:** Download the test dataset from [Google Drive](https://drive.google.com/drive/folders/1R9v5JrJFt7lZEoJ4DYPFNNXZwbEDo1fn?usp=sharing) and place it in the same `CT/` folder.  

---

### 3) Quick Run

```bash
# lp-lq TV demo
python3 lplq/DemoLpLqRecoTV.py

# Sketch-size study
python3 lplq/DemoLpLqRecoTVSketchSize.py

# CT demos (TV / Wavelet / Hessian–Schatten)
python3 CT/Demo2DCTl2RecoTV.py
python3 CT/Demo2DCTl2RecoWav.py
python3 CT/Demo2DCTl2RecoHS.py
```

---

## Example: Building the Nyström Preconditioner

<p align="center">
 <img src="SchemeImage.png" alt="Working pipline" width="110%"/>
</p>



Below is a minimal PyTorch example showing how to build and apply the Nyström preconditioner in the CT reconstruction setting.

```python
# Ax, ATx: forward model and its adjoint
# im_size / im_size_prod: image size and total number of elements
# sketch_size: sketch size for randomized Nyström approximation

import CTutilities as CTutl

U, S, lambda_l = CTutl.Build_Sketch_Real_Pred(
    Ax, ATx, im_size, im_size_prod, sketch_size, isBatch=True, device=device
)

# Build the preconditioner
U_temp = U * torch.sqrt(1 - (lambda_l + mu) / (S + mu))

# Define the preconditioner operator
P_inv = lambda x: CTutl.P_invx_SimpReal(x, U_temp, im_size)

# Now P_inv(x) can be used inside iterative solvers, e.g.:
# x_{k+1} = x_k + step_size * P_inv(r_k)
```


## Key Insights from the Paper

1. **On-the-fly Preconditioning**  
   Build an effective randomized Nyström preconditioner using only matrix–vector products Ax (A represents the forward model), without requiring explicit knowledge or structure of A.

2. **Acceleration for Variational Image Reconstruction**  
   Use the preconditioner to significantly accelerate iterative solvers with TV, wavelet, or Hessian–Schatten priors.
   
---

## Citation

If you find this work useful, please cite:

```bibtex
@article{hong2025nystrom-precond,
  title   = {Using Randomized Nyström Preconditioners to Accelerate Variational Image Reconstruction},
  author  = {Hong, Tao and Xu, Zhaoyi and Hu, Jason and Fessler, Jeffrey A.},
  journal = {IEEE Transactions on Computational Imaging},
  year    = {2025},
  note    = {to appear},
  eprint  = {2411.08178},
  archivePrefix = {arXiv},
  primaryClass  = {eess.IV}
}
```

---

## Contact

If you encounter bugs or have questions, feel free to reach out:  
- **tao.hong@austin.utexas.edu**  
- **zhaoyix@umich.edu**

If you are interested in discussing our work further, feel free to reach out as well. 

---






# CQNPM-GDD-CS-MRI-Reco.

[Tao Hong](https://hongtao-argmin.github.io), Zhaoyi Xu, [Se Young Chun](https://icl.snu.ac.kr), Luis Hernandez-Garcia, and [Jeffrey A. Fessler](https://web.eecs.umich.edu/~fessler/), ``[Convergent Complex Quasi-Newton Proximal Methods for Gradient-Driven Denoisers in Compressed Sensing MRI Reconstruction](https://arxiv.org/abs/2505.04820)'', arXiv:2505.04820, 2025.

We will release our implementation after paper acceptance. Feel free to shoot me an email (tao.hong@austin.utexas.edu) if you would like to learn more about our work.

[Project Website](https://hongtao-argmin.github.io/CQNPM-GD-CSMRI/)
