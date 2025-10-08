# EIGENSCORE: OOD DETECTION USING POSTERIOR COVARIANCE IN DIFFUSION MODELS

*Eigenvalue-based OOD detection framework.*

---

## Abstract
Out-of-distribution (OOD) detection is critical for the safe deployment of machine
learning systems in safety-sensitive domains. Diffusion models have recently
emerged as powerful generative models, capable of capturing complex data dis-
tributions through iterative denoising. Building on this progress, recent work has
explored their potential for OOD detection. We propose EigenScore, a new OOD
detection method that leverages the eigenvalue spectrum of the posterior covariance
induced by a diffusion model. We argue that posterior covariance provides a con-
sistent signal of distribution shift, leading to larger trace and leading eigenvalues
on OOD inputs, yielding a clear spectral signature. We further provide analysis
explicitly linking posterior covariance to distribution mismatch, establishing it as a
reliable signal for OOD detection. To ensure tractability, we adopt a Jacobian-free
subspace iteration method to estimate the leading eigenvalues using only forward
evaluations of the denoiser. Empirically, EigenScore achieves state-of-the-art per-
formance, with up to 5% AUROC improvement over the best baseline. Notably,
it remains robust in near-OOD settings such as CIFAR-10 vs CIFAR-100, where
existing diffusion-based methods often fail.

## Datasets/ Models
Download [datasets](https://drive.google.com/drive/folders/1PjuhJrJLxUhhG4Hd84X1n-QeFVFZOF0y?usp=share_link) CelebA, Cifar10, Cifar100, SVHN, and TinyImages.
Download [Models](https://drive.google.com/drive/folders/1Cptc8lxjpGY4MecaAzIbz7S18tc7J8mZ?usp=share_link)


## Requirement

Create the conda virtual enviroment.
```bash
conda env create -f EigenScore.yml
conda activate eigenscore
```

## Usage Example
### Eigenvalues Calculation
Read the description within the file and adjust the corresponding arguments.
```bash
python eigenscore_launcher.py

```
### OOD Detection 
Usage Example
```bash
python eigenscore_ood_detection.py -id <id dataset> -ood <ood-dataset> -e <eigenvalues file directory> -time <timesteps> -a <aggregation method> -rep <number of repetitions> -t <test fraction>
```
### Parameters Optimization
Read the description within the file and adjust the corresponding arguments.
```bash
python ES_optimize_parameters.py
```

### Returns optimal parameters 
Read the description within the file and adjust the corresponding arguments.
```bash
python ES_return_best_parameters.py
```