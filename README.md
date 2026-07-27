# MIRTorch

![GitHub release (latest by date including pre-releases)](https://img.shields.io/github/v/release/guanhuaw/mirtorch?include_prereleases)
![Read the Docs](https://img.shields.io/readthedocs/mirtorch)

A differentiable PyTorch toolbox for medical-image reconstruction, developed
at the University of Michigan. MIRTorch provides composable linear maps,
proximal operators, iterative solvers, and MRI and SPECT system models.

[Documentation](https://mirtorch.readthedocs.io/en/latest/) ·
[Examples](https://github.com/guanhuaw/MIRTorch/tree/master/examples) ·
[API](https://mirtorch.readthedocs.io/en/latest/API.html)

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

## Examples

The notebooks choose CUDA, Apple Metal, or CPU at runtime:

- [`demo_mri.ipynb`](https://github.com/guanhuaw/MIRTorch/blob/master/examples/demo_mri.ipynb):
  CG-SENSE and B0-informed PWLS
- [`demo_3d.ipynb`](https://github.com/guanhuaw/MIRTorch/blob/master/examples/demo_3d.ipynb):
  3D non-Cartesian MRI and Toeplitz embedding
- [`demo_cs.ipynb`](https://github.com/guanhuaw/MIRTorch/blob/master/examples/demo_cs.ipynb):
  compressed-sensing MRI
- [`demo_mlem.ipynb`](https://github.com/guanhuaw/MIRTorch/blob/master/examples/demo_mlem.ipynb):
  SPECT reconstruction
- [`demo_mnist.ipynb`](https://github.com/guanhuaw/MIRTorch/blob/master/examples/demo_mnist.ipynb):
  CG, FISTA, and POGM
- [`demo_dl.ipynb`](https://github.com/guanhuaw/MIRTorch/blob/master/examples/demo_dl.ipynb):
  dictionary learning

## Citation and acknowledgments

MIRTorch is inspired by
[MIRT](https://github.com/JeffFessler/mirt),
[MIRT.jl](https://github.com/JeffFessler/MIRT.jl),
[SigPy](https://github.com/mikgroup/sigpy), and
[PyLops](https://github.com/PyLops/pylops).
See the
[documentation](https://mirtorch.readthedocs.io/en/latest/README.html#citation-and-acknowledgments)
for the MIRTorch, BJORK, and SPECT citations.

MIRTorch is distributed under the
[BSD 3-Clause License](https://github.com/guanhuaw/MIRTorch/blob/master/LICENSE).
