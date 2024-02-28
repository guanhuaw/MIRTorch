Linear operators
================

.. automodule::
   mirtorch.linear.LinearMap

Attributes of LinearMap
---------------------------------

.. autosummary::
   :toctree: generated
   :nosignatures:

    mirtorch.linear.Add
    mirtorch.linear.Multiply
    mirtorch.linear.Matmul
    mirtorch.linear.ConjTranspose
    mirtorch.linear.Kron
    mirtorch.linear.BlockDiagonal


Basic image processing operations
---------------------------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   mirtorch.linear.basics.Diffnd
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
   mirtorch.linear.mri.NuSense
   mirtorch.linear.mri.Sense
   mirtorch.linear.mri.NuSenseGram
   mirtorch.linear.mri.Gmri


SPECT system models
-----------------
.. autosummary::
   :toctree: generated
   :nosignatures:

   mirtorch.linear.spect.SPECT


CT system models
-----------------
.. autosummary::
   :toctree: generated
   :nosignatures:

   mirtorch.linear.ct.Bdd
