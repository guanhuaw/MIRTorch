Linear operators
================

.. autoclass:: mirtorch.linear.LinearMap

Linear map composition
----------------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   mirtorch.linear.Add
   mirtorch.linear.Multiply
   mirtorch.linear.Matmul
   mirtorch.linear.ConjTranspose
   mirtorch.linear.Kron
   mirtorch.linear.BlockDiagonal
   mirtorch.linear.Hstack
   mirtorch.linear.Vstack


Basic image processing operations
---------------------------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   mirtorch.linear.basics.Diffnd
   mirtorch.linear.basics.Diff1d
   mirtorch.linear.basics.Diff2dgram
   mirtorch.linear.basics.Diff3dgram
   mirtorch.linear.basics.Diag
   mirtorch.linear.basics.Identity
   mirtorch.linear.basics.Convolve1d
   mirtorch.linear.basics.Convolve2d
   mirtorch.linear.basics.Convolve3d
   mirtorch.linear.basics.Patch2D
   mirtorch.linear.basics.Patch3D
   mirtorch.linear.wavelets.Wavelet2D


MRI system models
-----------------
.. autosummary::
   :toctree: generated
   :nosignatures:

   mirtorch.linear.mri.FFTCn
   mirtorch.linear.mri.Sense
   mirtorch.linear.mri.NuSense
   mirtorch.linear.mri.NuSenseGram
   mirtorch.linear.mri.Gmri
   mirtorch.linear.mri.GmriGram


CT system models
----------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   mirtorch.linear.ct.CT

``CT`` supports two-dimensional parallel beams and flat-detector fan beams.
``A(mu)`` computes line integrals; ``A.expected_counts(mu, incident, background)``
computes the nonlinear monochromatic Poisson mean. Lengths share one physical
unit, and attenuation is in its inverse, not Hounsfield units. Angles are in
radians. The image axes are ``(y, x)``, with both coordinates increasing with
index; at zero angle, rays travel along +y and detector bins along +x.

Siddon intersections are exact for the piecewise-constant pixel model, and
``A.H`` is its discrete Euclidean adjoint, not an inverse. Geometry is fixed
at construction; image and count-calibration gradients support higher orders.
Intersection plans are cached only within a configurable memory budget;
larger plans are recomputed in ray chunks. The cache budget does not limit
autograd activations, which can retain all chunk weights in differentiable runs.
This portable research projector
does not model cone beams, finite detector/focal-spot integration, beam
hardening, or scatter transport. See the
`CT notebook <https://github.com/guanhuaw/MIRTorch/blob/master/examples/demo_ct.ipynb>`_
for analytic checks and noisy reconstruction, and
`Siddon (1985) <https://doi.org/10.1118/1.595715>`_ for the line-intersection model.


SPECT system models
-------------------
.. autosummary::
   :toctree: generated
   :nosignatures:

   mirtorch.linear.spect.SPECT
   mirtorch.linear.spect.required_rotation_shape
   mirtorch.linear.spect.parallel_hole_psfs

SPECT remains linear in activity for a fixed attenuation map and PSFs. Its
angles are in **degrees**, preserving the existing API. Attenuation coefficients
are inverse length; ``dy``, optional ``dx``, and PSF calibration lengths must
use the same unit. The detector-facing depth index is zero.

Use ``required_rotation_shape(size_in, dy, dx=dx)`` to size a centered grid
covering every image rotation. The lateral detector width is ``size_out[0]``;
the depth count is ``psfs.shape[2]``. Calibrated PSFs must explicitly cover that
depth range: they are never extrapolated. Existing image-sized grids remain
accepted but retain their limited support. Extra detector width can capture
PSF tails; genuine detector truncation is not corrected. Bilinear rotation
still has interpolation error and is not exactly count preserving for point
sources, even when its support is fully covered.

``psfs`` are impulse responses centered at ``(px//2, pz//2)``. Non-symmetric
kernels now have the physical convolution orientation; their old mirrored
behavior is intentionally corrected. Supplied PSF amplitudes are preserved.
``parallel_hole_psfs`` constructs normalized Gaussian responses from explicit
source-to-crystal distances and collimator/intrinsic-resolution parameters;
it does not supply absolute detector sensitivity or septal-penetration physics.

``attenuation="midpoint"`` preserves the old self-attenuation approximation.
The recommended ``attenuation="voxel"`` averages survival probability for
uniform emission inside each rotated depth cell:

.. math::

   a_j = \exp\!\left(-\Delta_y\sum_{k<j}\mu_k\right)
         \frac{1-\exp(-\mu_j\Delta_y)}{\mu_j\Delta_y}.

The ratio is evaluated stably at zero, including its derivatives. Neither
mode models scatter transport or scanner calibration. The
`SPECT notebook <https://github.com/guanhuaw/MIRTorch/blob/master/examples/demo_mlem.ipynb>`_
uses the full grid, physical PSFs, and voxel self-attenuation; its synthetic
reconstruction is a numerical demonstration, not independent scanner validation.
