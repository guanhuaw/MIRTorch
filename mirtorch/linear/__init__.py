from importlib import import_module

from .basics import (
    Convolve1d,
    Convolve2d,
    Convolve3d,
    Diag,
    Diff1d,
    Diff2dgram,
    Diff3dgram,
    Diffnd,
    Identity,
    Patch2D,
    Patch3D,
)
from .linearmaps import (
    Add,
    BlockDiagonal,
    ConjTranspose,
    Hstack,
    Kron,
    LinearMap,
    Matmul,
    Multiply,
    Vstack,
)
from .spect import SPECT
from .wavelets import Wavelet2D

_MRI_EXPORTS = frozenset(
    {"FFTCn", "Gmri", "GmriGram", "NuSense", "NuSenseGram", "Sense"}
)


def __getattr__(name):
    """Load the optional, comparatively expensive MRI module on first use."""
    if name in _MRI_EXPORTS:
        value = getattr(import_module(".mri", __name__), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "SPECT",
    "Add",
    "BlockDiagonal",
    "ConjTranspose",
    "Convolve1d",
    "Convolve2d",
    "Convolve3d",
    "Diag",
    "Diff1d",
    "Diff2dgram",
    "Diff3dgram",
    "Diffnd",
    "FFTCn",
    "Gmri",
    "GmriGram",
    "Hstack",
    "Identity",
    "Kron",
    "LinearMap",
    "Matmul",
    "Multiply",
    "NuSense",
    "NuSenseGram",
    "Patch2D",
    "Patch3D",
    "Sense",
    "Vstack",
    "Wavelet2D",
]
