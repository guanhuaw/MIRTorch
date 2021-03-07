# MIRTorch

PyTorch-based modular **I**mage **R**econstruction **T**oolbox, with emphasis on algorithm unrolling/unfolding, developed in University of **M**ichigan.

This work is inspired by (but not limited to):

SigPy: https://github.com/mikgroup/sigpy

MIRT/MIRT.jl: https://web.eecs.umich.edu/~fessler/code/index.html

PyLops: https://github.com/PyLops/pylops

Main features include: 

##### Linear operators

The LinearMap class overload common matrix operations, like +, - , *.  Also support more efficient backpropagation.

Instances include basic linear operations (like convolution) and MRI system matrix. More is on the way ...

##### Proximal operators

Supports unitary transformation and majorization.

##### Iterative Reconstruction (IR) algorithms

Currently includes conjugate gradients (CG) methods. More first-order methods, like FISTA and POGM is being implemented.

##### Algorithm unrolling/unrolled networks

Currently supports MoDL. Currently, we are testing memory-efficient (MELD) and other unrolled networks.

#### Applications/demos

##### MRI reconstruction: 

example/demo_mri.ipynb includes SENSE, Non-Cartesian SENSE and B0-informed reconstruction with penalized weighted least squares (*PWLS*) methods.

##### MRI sampling pattern optimization:

example/demo_mri_traj.demo contains MRI sampling pattern optimization examples. The reconstruction loss is utilized to jointly optimize reconstruction algorithms and sampling pattern.

