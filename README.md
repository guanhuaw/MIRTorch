# MIRTorch

![GitHub release (latest by date including pre-releases)](https://img.shields.io/github/v/release/guanhuaw/mirtorch?include_prereleases)
![Read the Docs](https://img.shields.io/readthedocs/mirtorch)

A Py***Torch***-based differentiable ***I***mage ***R***econstruction ***T***oolbox, developed at the University of ***M***ichigan.

The work is inspired by [MIRT](https://github.com/JeffFessler/mirt), a well-acclaimed toolbox for medical imaging reconstruction. 

The overarching goal is to provide fast iterative and data-driven image reconstruction across CPUs and GPUs. Researchers can rapidly develop new model-based and learning-based methods (i.e., unrolled neural networks) with convenient abstraction layers. With the full support of auto-differentiation, one may optimize imaging protocols and image reconstruction parameters with gradient methods.

Documentation: https://mirtorch.readthedocs.io/en/latest/

------

### Installation

We recommend to [pre-install `PyTorch` first](https://pytorch.org/).
To install the `MIRTorch` package, after cloning the repo, please try `python setup.py install`. 

`requirements.txt` details the package dependencies. We recommend installing [pytorch_wavelets](https://github.com/fbcotter/pytorch_wavelets) directly from the source code instead of `pip`.  

------

### Features

#### Linear maps

The `LinearMap` class overloads common matrix operations, such as `+, - , *`.

Instances include basic linear operations (like convolution), classical imaging processing, and MRI system matrix (Cartesian and Non-Cartesian, sensitivity- and B0-informed system models). More is on the way...

Since the Jacobian matrix of a linear operator is [itself](https://en.wikipedia.org/wiki/Matrix_calculus/), the toolbox can actively calculate such Jacobians during backpropagation, avoiding the large cache cost required by auto-differentiation.

When defining linear operators, please make sure that all torch tensors are on the same device and compatible. For example, `torch.cfloat` are compatible with `torch.float` but not `torch.double`.

#### Proximal operators

The toolbox contains common proximal operators such as soft thresholding. These operators also support the regularizers that involve multiplication with diagonal or unitary matrices, such as orthogonal wavelets.

#### Iterative reconstruction (MBIR) algorithms

Currently, the package includes the conjugate gradient ([CG](https://en.wikipedia.org/wiki/Conjugate_gradient_method)), fast iterative thresholding ([FISTA](https://epubs.siam.org/doi/10.1137/080716542)), optimized gradient method ([POGM](https://dl.acm.org/doi/10.1007/s10957-018-1287-4)), forward-backward primal-dual ([FBPD](https://arxiv.org/abs/1406.5439)) algorithms for image reconstruction.

#### Dictionary learning

For dictionary learning-based reconstruction, we implemented an efficient dictionary learning algorithm ([SOUP-DIL](https://arxiv.org/abs/1511.06333)) and orthogonal matching pursuit ([OMP](https://ieeexplore.ieee.org/abstract/document/342465/?casa_token=aTDkQVCM9WEAAAAA:5rXu9YikP822bCBvkhYxKWlBTJ6Fn6baTQJ9kuNrU7K-64EmGOAczYvF2dTW3al3PfPdwJAiYw)). Due to PyTorch’s limited support of sparse matrices, we use SciPy as the backend. 

------

### Usage and examples

`/example` includes several examples. 

`/example/demo_mnist.ipynb` shows the [LASSO](https://en.wikipedia.org/wiki/Lasso_(statistics)) on MNIST with FISTA and POGM. 

`/example/demo_mri.ipynb` contains the SENSE (CG-SENSE) and **B0**-informed reconstruction with penalized weighted least squares (*PWLS*).

`/example/demo_cs.ipynb` shows the compressed sensing reconstruction of under-determined MRI signals.

`/example/demo_dl.ipynb` exhibits the dictionary learning results.

[Bjork repo](https://github.com/guanhuaw/Bjork) contains MRI sampling pattern optimization examples. One may use the reconstruction loss as the objective function to jointly optimize reconstruction algorithms and the sampling pattern.

------

### Acknowledgments

This work is inspired by (but not limited to):

* SigPy: https://github.com/mikgroup/sigpy

* MIRT: https://github.com/JeffFessler/mirt

* MIRT.jl: https://github.com/JeffFessler/MIRT.jl

* PyLops: https://github.com/PyLops/pylops

If the code is useful to your research, please cite:

```bibtex
@article{wang:22:bjork,
  author={Wang, Guanhua and Luo, Tianrui and Nielsen, Jon-Fredrik and Noll, Douglas C. and Fessler, Jeffrey A.},
  journal={IEEE Transactions on Medical Imaging}, 
  title={B-spline Parameterized Joint Optimization of Reconstruction and K-space Trajectories (BJORK) for Accelerated 2D MRI}, 
  year={2022},
  pages={1-1},
  doi={10.1109/TMI.2022.3161875}}
```

------

### License

This package uses the BSD3 license. 