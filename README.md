# MIRTorch

PyTorch-based modular image reconstruction box, with emphasis on algorithm unrolling/unfolding, developed in University of Michigan.

This work is inspired by (but not limited to):

SigPy: https://github.com/mikgroup/sigpy

MIRT/MIRT.jl: https://web.eecs.umich.edu/~fessler/code/index.html

PyLops: https://github.com/PyLops/pylops

Main materials include: 

##### Linear operators

The LinearMap class overload common matrix operations, like +, - , *.  Also support more efficient backpropagation.

Instances include basic linear operations (like convolution) and MRI system matrix. More like tomography is on the way ...

##### Proximal operators

Supports unitary transformation and majorization.

##### Iterative Reconstruction (IR) algorithms

Currently includes conjugate gradients (CG) methods. More first-order methods, like FISTA and POGM is being implemented.

##### Algorithm unrolling/unrolled networks

Currently supports MoDL. Currently, we are testing memory-efficient (MELD) and other unrolled networks.

#### Application:





