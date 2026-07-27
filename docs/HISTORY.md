# History

Unreleased
----------

- Add default FINUFFT/cuFINUFFT acceleration and Toeplitz normal operators for
  non-Cartesian and B0-informed MRI, with a torchkbnufft fallback.
- Add automatic compilation for supported CUDA finite-difference and iterative
  paths.
- Improve SPECT attenuation and PSF modeling while preserving an exact adjoint
  and differentiability.
- Correct complex adjoints, B0 timing and gradients, CG/FISTA behavior, and
  weighted proximal formulas.
- Improve platform-aware examples, validation, packaged wavelet data,
  documentation, CI, and release testing.

0.1.3 (2025-12-18)
------------------

- Correct example links and package metadata.

0.1.2 (2024-08-04)
------------------

- Update dependencies and packaging for current PyTorch releases.

0.0.3 (2023-02-10)
------------------

- Add Toeplitz embedding for B0-informed reconstruction.
- Update torchkbnufft support and fix the Gmri operator.
- Add linear operators.

0.0.2 (2022-06-05)
------------------

- Update documentation and fix the B0-informed system matrix.

0.0.1 (2022-02-04)
------------------

- Add Read the Docs documentation and CG preconditioning.
