# MIRTorch

MIRTorch is a differentiable PyTorch toolbox for medical-image reconstruction,
developed at the University of Michigan. It provides composable linear maps,
proximal operators, iterative solvers, and MRI and SPECT system models.

New to reconstruction? Start with the self-contained
[`demo_mr_physics.ipynb`](https://github.com/guanhuaw/MIRTorch/blob/master/examples/demo_mr_physics.ipynb).

The project is inspired by
[MIRT](https://github.com/JeffFessler/mirt) and is intended for rapid
prototyping of model-based, learning-based, and acquisition-optimization
methods.

## New

Non-Cartesian and B0-informed MRI now use FINUFFT on supported non-macOS CPU
systems and cuFINUFFT on CUDA when installed. In warm NVIDIA A10 benchmarks,
the new paths measured up to **5.2× faster NUFFT**, **9.4× faster Toeplitz
normal operations**, and **12.3× faster iterative solvers**. These are
workload-specific measurements, not universal speedups.

## Install

Install [PyTorch](https://pytorch.org/) for your platform, then:

```bash
pip install MIRTorch
```

CUDA users can install cuFINUFFT with:

```bash
pip install "MIRTorch[cufinufft]"
```

For local development:

```bash
pip install -e ".[dev]"
```

## Backends and compilation

`NuSense`, `NuSenseGram`, `Gmri`, and `GmriGram` use an installed FINUFFT or
cuFINUFFT library when the device supports it, then fall back to torchkbnufft.
Base macOS, Apple Metal, and Linux ARM installs therefore work without a
native library. Set `backend="torchkbnufft"` or `backend="finufft"` to
override the automatic choice.

Real-valued CUDA runs of `Diff2dgram`, FISTA, and POGM use `torch.compile`
automatically when PyTorch provides it. Other inputs stay eager; pass
`compile=False` to disable compilation explicitly.

## Core API

MIRTorch represents a linear system as $y = A x$. A `LinearMap` can be called
directly, multiplied by a tensor, composed with other maps, or conjugate
transposed with `.H`:

```python
from mirtorch.linear import NuSense, NuSenseGram

A = NuSense(smaps, trajectory)
y = A(x)
x_adjoint = A.H(y)
AHA = NuSenseGram(smaps, trajectory.detach())
```

Linear and proximal operators can move their tensor state recursively:

```python
operator = operator.to("cuda")
regularizer = regularizer.to("cuda")
```

For MRI operators, place `smaps` and `trajectory` on the target device before
construction so the automatic NUFFT backend is selected for that device.

Operators using `batchmode=True` expect an explicit batch and channel layout,
such as `[batch, channel, nx, ny]`. See each operator's API page for its exact
input and output shapes.

## Examples

Each notebook has an **Open in Colab** badge and a Colab-only setup cell; local
runs continue to use the current checkout. Most examples choose CUDA, Apple
Metal, or CPU at runtime. The dictionary-learning example deliberately stays
on CPU because it exchanges sparse arrays with SciPy:

- [MR physics to inverse problems](https://github.com/guanhuaw/MIRTorch/blob/master/examples/demo_mr_physics.ipynb)
- [MRI and B0-informed PWLS](https://github.com/guanhuaw/MIRTorch/blob/master/examples/demo_mri.ipynb)
- [3D non-Cartesian MRI](https://github.com/guanhuaw/MIRTorch/blob/master/examples/demo_3d.ipynb)
- [SNOPY-style trajectory optimization](https://github.com/guanhuaw/MIRTorch/blob/master/examples/demo_trajectory_optimization.ipynb)
- [Compressed-sensing MRI](https://github.com/guanhuaw/MIRTorch/blob/master/examples/demo_cs.ipynb)
- [SPECT reconstruction](https://github.com/guanhuaw/MIRTorch/blob/master/examples/demo_mlem.ipynb)
- [CG, FISTA, and POGM](https://github.com/guanhuaw/MIRTorch/blob/master/examples/demo_mnist.ipynb)
- [Dictionary learning](https://github.com/guanhuaw/MIRTorch/blob/master/examples/demo_dl.ipynb)

For background on inverse problems and optimization, see
[Fessler's book](https://web.eecs.umich.edu/~fessler/book/) and
[Boyd and Vandenberghe](https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf).
The [BJORK repository](https://github.com/guanhuaw/Bjork) demonstrates joint
optimization of MRI reconstruction and sampling trajectories.

## Citation and acknowledgments

MIRTorch is inspired by:

- [MIRT](https://github.com/JeffFessler/mirt)
- [MIRT.jl](https://github.com/JeffFessler/MIRT.jl)
- [SigPy](https://github.com/mikgroup/sigpy)
- [PyLops](https://github.com/PyLops/pylops)

If MIRTorch or its differentiable MRI tools are useful in your research,
please cite:

```bibtex
@article{wang:22:bjork,
  author={Wang, Guanhua and Luo, Tianrui and Nielsen, Jon-Fredrik and
          Noll, Douglas C. and Fessler, Jeffrey A.},
  journal={IEEE Transactions on Medical Imaging},
  title={B-spline Parameterized Joint Optimization of Reconstruction and
         K-space Trajectories ({BJORK}) for Accelerated {2D} {MRI}},
  year={2022},
  pages={1-1},
  doi={10.1109/TMI.2022.3161875}}
```

```bibtex
@inproceedings{wang:22:mirtorch,
  title={{MIRTorch}: A {PyTorch}-powered Differentiable Toolbox for Fast Image
         Reconstruction and Scan Protocol Optimization},
  author={Wang, Guanhua and Shah, Neel and Zhu, Keyue and Noll, Douglas C. and
          Fessler, Jeffrey A.},
  booktitle={Proc. Intl. Soc. Magn. Reson. Med. (ISMRM)},
  pages={4982},
  year={2022}
}
```

If you use the SPECT model, please cite:

```bibtex
@article{li:23:tet,
  author={Li, Zongyu and Dewaraja, Yuni K. and Fessler, Jeffrey A.},
  journal={IEEE Transactions on Radiation and Plasma Medical Sciences},
  title={Training End-to-End Unrolled Iterative Neural Networks for SPECT
         Image Reconstruction},
  year={2023},
  volume={7},
  number={4},
  pages={410-420},
  doi={10.1109/TRPMS.2023.3240934}}
```

MIRTorch is distributed under the BSD 3-Clause License.
