# MIRTorch

![GitHub release (latest by date including pre-releases)](https://img.shields.io/github/v/release/guanhuaw/mirtorch?include_prereleases)
![Read the Docs](https://img.shields.io/readthedocs/mirtorch)

A Py***Torch***-based differentiable ***I***mage ***R***econstruction ***T***oolbox, developed at the University of ***M***ichigan.

The work is inspired by [MIRT](https://github.com/JeffFessler/mirt), a well-acclaimed toolbox for medical imaging reconstruction.

The main objective is to facilitate rapid, data-driven medical image reconstruction using CPUs and GPUs, for fast prototyping. Researchers can conveniently develop new model-based and learning-based methods (e.g., unrolled neural networks) with abstraction layers. The availability of auto-differentiation enables optimization of imaging protocols and reconstruction parameters using gradient methods.

Documentation: https://mirtorch.readthedocs.io/en/latest/

------

### Installation

We recommend to [pre-install `PyTorch` first](https://pytorch.org/).
Use `pip install mirtorch` to install.
To install the `MIRTorch` locally, after cloning the repo, please try `pip install -e .`(one may modify the package locally with this option.)

------

### Features

#### Linear maps

The `LinearMap` class overloads common matrix operations, such as `+, - , *`.

Instances include basic linear operations (like convolution), classical imaging processing, and MRI system matrix (Cartesian and Non-Cartesian, sensitivity- and B0-informed system models). ***NEW!*** MIRTorch recently adds the support for SPECT and CT.

Since the Jacobian matrix of a linear operator is itself, the toolbox can actively calculate such Jacobians during backpropagation, avoiding the large cache cost required by auto-differentiation.

When defining linear operators, please make sure that all torch tensors are on the same device and compatible. For example, `torch.cfloat` are compatible with `torch.float` but not `torch.double`. Similarly, `torch.chalf` is compatible with `torch.half`.
When the data is image, there are 2 empirical formats: `[num_batch, num_channel, nx, ny, (nz)]` and `[nx, ny, (nz)]`.
For some LinearMaps, there is a boolean `batchmode` to control the shape.

#### Proximal operators

The toolbox contains common proximal operators such as soft thresholding. These operators also support the regularizers that involve multiplication with diagonal or unitary matrices, such as orthogonal wavelets.

#### Iterative reconstruction (MBIR) algorithms

Currently, the package includes the conjugate gradient (CG), fast iterative thresholding (FISTA), optimized gradient method (POGM), forward-backward primal-dual (FBPD) algorithms for image reconstruction.

#### Dictionary learning

For dictionary learning-based reconstruction, we implemented an efficient dictionary learning algorithm ([SOUP-DIL](https://arxiv.org/abs/1511.06333)) and orthogonal matching pursuit ([OMP](https://ieeexplore.ieee.org/abstract/document/342465/?casa_token=aTDkQVCM9WEAAAAA:5rXu9YikP822bCBvkhYxKWlBTJ6Fn6baTQJ9kuNrU7K-64EmGOAczYvF2dTW3al3PfPdwJAiYw)). Due to PyTorch’s limited support of sparse matrices, we use SciPy as the backend.

#### Multi-GPU support

Currently, MIRTorch uses `torch.DataParallel` to support multiple GPUs. One may re-package the `LinearMap`, `Prox` or `Alg` inside a `torch.nn.Module` to enable data parallel. See [this tutorial](https://pytorch.org/tutorials/beginner/blitz/data_parallel_tutorial.html) for detail.

------

### Usage and examples

Generally, MIRTorch solves the image reconstruction problems that have the cost function $\textit{argmin}_{x} \|Ax-y\|_2^2 + \lambda \textit{R}(x)$. $A$ stands for the system matrix. When it is linear, one may use `LinearMap` to efficiently compute it. `y` usually denotes measurements. $\textit{R}(\cdot)$ denotes regularizers, which determines which `Alg` to be used. One may refer to [1](https://web.eecs.umich.edu/~fessler/book/), [2](https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf) and [3](https://www.youtube.com/watch?v=J6_5rPYnr_s) for more tutorials on optimization.

Here we provide several notebook tutorials focused on MRI, where $A$ is FFT or NUFFT.

- `/example/demo_mnist.ipynb` shows the LASSO on MNIST with FISTA and POGM.
- `/example/demo_mri.ipynb` contains the SENSE (CG-SENSE) and **B0**-informed reconstruction with penalized weighted least squares (*PWLS*).
- `/example/demo_3d.ipynb` contains the 3d non-Cartesian MR reconstruction. *New!* Try the Toeplitz-embedding version of B0-informed reconstruction, which reduce hour-long recon to 5 secs.
- `/example/demo_cs.ipynb` shows the compressed sensing reconstruction of under-determined MRI signals.
- `/example/demo_dl.ipynb` exhibits the dictionary learning results.
- `/example/demo_mlem` showcase SPECT recon algorithms, including EM and CNN.

Since MIRTorch is differentiable, one may use AD to update many parameters. For example, updating the reconstruction neural network's weights. More importantly, one may update the imaging system itself via gradient-based and data-driven methods. As a user case, [Bjork repo](https://github.com/guanhuaw/Bjork) contains MRI sampling pattern optimization examples. One may use the reconstruction loss as the objective function to jointly optimize reconstruction algorithms and the sampling pattern. See [this video](https://www.youtube.com/watch?v=sLFOf5EvVAs) on how to jointly optimize reconstruction and acquisition.

------

### Acknowledgments

This work is inspired by (but not limited to):

* SigPy: https://github.com/mikgroup/sigpy

* MIRT: https://github.com/JeffFessler/mirt

* MIRT.jl: https://github.com/JeffFessler/MIRT.jl

* PyLops: https://github.com/PyLops/pylops

If the code is useful to your research, please consider citing:

```bibtex
@article{wang:22:bjork,
  author={Wang, Guanhua and Luo, Tianrui and Nielsen, Jon-Fredrik and Noll, Douglas C. and Fessler, Jeffrey A.},
  journal={IEEE Transactions on Medical Imaging},
  title={B-spline Parameterized Joint Optimization of Reconstruction and K-space Trajectories ({BJORK}) for Accelerated {2D} {MRI}},
  year={2022},
  pages={1-1},
  doi={10.1109/TMI.2022.3161875}}
```

```bibtex
@inproceedings{wang:22:mirtorch,
  title={{MIRTorch}: A {PyTorch}-powered Differentiable Toolbox for Fast Image Reconstruction and Scan Protocol Optimization},
  author={Wang, Guanhua and Shah, Neel and Zhu, Keyue and Noll, Douglas C. and Fessler, Jeffrey A.},
  booktitle={Proc. Intl. Soc. Magn. Resonance. Med. (ISMRM)},
  pages={4982},
  year={2022}
}
```
If you use the SPECT code, please consider citing:

```bibtex
@ARTICLE{li:23:tet,
  author={Li, Zongyu and Dewaraja, Yuni K. and Fessler, Jeffrey A.},
  journal={IEEE Transactions on Radiation and Plasma Medical Sciences},
  title={Training End-to-End Unrolled Iterative Neural Networks for SPECT Image Reconstruction},
  year={2023},
  volume={7},
  number={4},
  pages={410-420},
  doi={10.1109/TRPMS.2023.3240934}}
```


------

### License

This package uses the BSD3 license.
