---
title: 'MIRTorch: A Differentiable Medical Image Reconstruction Toolbox'
tags:
  - Python
  - medical imaging
  - image reconstruction
  - deep learning
  - PyTorch
authors:
  - name: Guanhua Wang
    affiliation: 2
    orcid: 0000-0002-1622-5664
    corresponding: true
  - name: Neel Shah
    affiliation: 1
  - name: Keyue Zhu
    affiliation: 1
  - name: Tianrui Luo
    affiliation: 2
    orcid: 0000-0003-4770-530X
  - name: Naveen Murthy
    affiliation: 1
    orcid: 0000-0003-1365-3302
  - name: Zongyu Li
    affiliation: 1
    orcid: 0000-0003-1813-1722
  - name: Minseo Kim
    affiliation: 1
    orcid: 0009-0001-7404-8387
  - name: Douglas C Noll
    affiliation: 2
    orcid: 0000-0002-0983-3805
  - name: Jeffrey A Fessler
    affiliation: 1
    orcid: 0000-0001-9998-3315
affiliations:
 - name: Department of EECS, University of Michigan, USA
   index: 1
 - name: Department of Biomeidcal Engineering, University of Michigan, USA
   index: 2

date: 3 August 2024
bibliography: paper.bib

---

# Summary

Image reconstruction converts raw signals into digitized images and is an essential part of modern medical imaging. High-quality image reconstruction provides powerful tools for radiologists in diagnosing. Fast and accurate image reconstruction is an active topic across many research fields, including signal processing, computational imaging and machine learning. Image reconstruction toolboxes support these researches with algorithmic infrastructures and baselines.

For many imaging modalities, such as magnetic resonance imaging (MRI), image reconstruction is an inverse problem that is often underdetermined and large-scale. Reconstruction toolboxes can model the imaging physics, implement the regularization, and provide corresponding solvers. Recent years have also seen a surge in deep learning-based reconstruction, which learns to solve the inverse problem [@deep_learning]. One may integrate physics modeling into deep learning frameworks to combine the best of both worlds [@physics_informed]. These model-based or physics-informed deep learning methods received wide attention because of their robustness and explainability.

The main goal of MIRTorch is to assist research on image reconstruction algorithm development using data-driven approaches. Natively built with PyTorch, MIRTorch fully supports auto-differentiation (AD), with an affinity to deep learning modules such as CNN or Transformers.

Following the earlier MIRT (Michigan Image Reconstruction Toolbox) [@MIRT] and SigPy [@SigPy], MIRTorch has a clear and modular structure, facilitating fast prototyping of novel algorithms. The main features and components include:

1. Generalization of linear operators as matrices. In many image modalities such as MRI, the forward system model is linear; thus, one may regard these models as matrices (though they may have otherwise efficient implementations). By overloading operators such as +, -, *, MIRTorch aids researchers in defining system models and avoiding erroneous hardcoding.

2. Efficient iterative solvers. MIRTorch includes various solvers, including CG [@CG], FISTA [@FISTA], and primal-dual [@primal_dual] to handle multiple scenarios. Users can easily combine learnable modules (such as CNNs) with numerical solvers to investigate model-based deep learning methods. Several common proximal operators [@proximal] are also provided.

# Statement of need

There exist several well-received open-source medical image reconstruction toolboxes. MIRT contains comprehensive functionalities in image processing and image reconstruction. Its implementation is based on Matlab, complicating the development of learning-based methods. BART [@BART] and Gadgetron [@Gadgetron] use efficient C/C++/CUDA backends. BART delivers many high-level reconstruction tools. Gadgetron validated modular and online reconstruction and is being integrated into the clinical workflow. Still, both projects lack the support of differentiable programming. SigPy is a Python-based reconstruction toolbox focusing on rapid prototyping. Though its linear operators support AD, the other components (such as the solver) are non-differentiable.

With native PyTorch support, MIRTorch satisfies the following specific needs:

1. Fast prototyping of model-based deep learning. Many related projects hard-coded the physics-informed components, such as system operators and iterative solvers. This non-modular approach hampers reproducibility and comparability. MIRTorch provides a standardized and modular implementation. It also facilitates transferring algorithms across different imaging modalities. Additionally, the code using MIRTorch better matches the mathematical expressions, facilitating understanding.

2. Optimization of the imaging system. Since the toolbox is fully differentiable, it enables gradient methods for tuning imaging system parameters. For example, [@sampling] uses MIRTorch to optimize MRI sampling trajectories via stochastic gradient descent.

3. User-friendly fast reconstruction. Many applications, such as functional MRI (fMRI), are vectorized and large-dimensional. CPU-based computation can be very time-consuming for iterative algorithms. Benefitting from PyTorch's intrinsic multi-GPU support, user-friendly installation, and cross-platform capability, MIRTorch provides researchers with fast reconstruction at a minimal switching cost.

# Acknowledgements

This work is supported by NIH Grants R01 EB023618 and U01 EB026977, and NSF Grant IIS 1838179.

# References
