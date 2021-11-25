# MIRTorch

### Intro

A Py***Torch***-based differentiable ***I***mage ***R***econstruction ***T***oolbox, with emphasis on algorithm unrolling/unfolding, developed at the University of ***M***ichigan.

<u>This project is still in a active re-building mode. We plan to release a stable API with docs later this year.</u>

Main features include: 

### Linear maps

The LinearMap class overloads common matrix operations, such as `+, - , *`. It also supports an efficient back-propagation.

Instances include basic linear operations (like convolution), classical imaging processing and MRI system matrix. More is on the way...

### Proximal operators

Supports the multiplication with diagonal and unitary transformations.

### Iterative reconstruction (IR) algorithms

Currently the first-order methods include CG, FISTA and POGM.

### Applications/demos

#### MRI reconstruction: 

/example includes SENSE, Non-Cartesian SENSE and **B0**-informed reconstruction with penalized weighted least squares (*PWLS*), compressed sensing (CS) and dictionary learning (DL) methods

#### MRI sampling pattern optimization:

example/demo_mri_traj.demo contains MRI sampling pattern optimization examples. One may use the reconstruction loss as objective function to jointly optimize reconstruction algorithms and the sampling pattern.

### Credit

This work is inspired by (but not limited to):

SigPy: https://github.com/mikgroup/sigpy

MIRT/MIRT.jl: https://web.eecs.umich.edu/~fessler/code/index.html

PyLops: https://github.com/PyLops/pylops