# History

Unreleased
----------

- Add portable 2D parallel-/fan-beam CT with physical ray lengths, matched
  adjoints, differentiable Beer–Lambert counts, and a reconstruction notebook.
- Add full-field, anisotropic SPECT rotation grids, correctly oriented PSFs,
  calibrated Gaussian parallel-hole responses, and optional voxel-integrated
  self-attenuation. Update the SPECT tutorial with explicit physical units.
- Preserve small implicit CG gradients using a normalized relative backward
  solve; document its first-order contract and reject unsupported double backward.
- Reject ambiguous unrolled CG gradients through zero-residual recurrences
  without changing forward results or normal inference.
- Preserve complex proximal derivatives at zero and complex output dtypes in
  block-diagonal and Kronecker operators.
- Correct second derivatives of squared norms and L2 proximal operators,
  including weighted updates and zero step sizes at smooth points.
- Support complex patch adjoints and backpropagation on Metal; correct the
  legacy FFT convolution adjoint for asymmetric and complex kernels.
- Use an explicit wavelet adjoint, supporting inference mode and accurate
  double-precision filters while reducing repeated reconstruction overhead.
- Make B0 Toeplitz operators match the same time-segmented forward model,
  including cross terms; bound their increased kernel memory with a direct fallback.
- Prevent stale NUFFT plans and B0 coefficients after trajectory storage reuse,
  inference-mode updates, logical-view changes, and device moves; fix
  shared-trajectory B0 batching and native NUFFT conjugate/negative inputs.
- Reuse native plans across trajectory updates and reduce saved NUFFT activations,
  trajectory-gradient workspace, and B0 FFT temporaries. Avoid redundant CG
  reductions and full-size identity weights in unweighted proximal updates.
- Move torchvision to optional example dependencies and import it only when
  the legacy image-rotation helper is used. Remove the unused einops dependency.
- Execute self-contained physics and trajectory tutorials against an installed
  wheel in CI; add opt-in CUDA/Metal release acceptance on self-hosted runners.

0.3.1 (2026-08-20)
------------------

- Add a self-contained tutorial connecting MR physics, signal encoding,
  inverse problems, and optimization.
- Make every example notebook runnable from Colab with reproducible setup,
  complete dependencies, portable links, and safe opt-in large-data sections.

0.3.0 (2026-08-02)
------------------

- Add efficient first-order trajectory gradients for torchkbnufft and
  FINUFFT/cuFINUFFT, with a SNOPY-style optimization tutorial.
- Align B0 time-segmentation with MIRT's histogram-weighted fit while
  preserving gradients with respect to the field map and sampling times.
- Add solver diagnostics and warm starts, relative stopping for CG and
  proximal methods, and adaptive restart for FISTA and POGM.
- Rewrite the compressed-sensing tutorial with safer step sizes, comparable
  metrics, and clearer reconstruction diagnostics.

0.2.0 (2026-07-28)
------------------

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
