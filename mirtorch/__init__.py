from importlib.metadata import PackageNotFoundError, version

from mirtorch import alg, dic, linear, prox

try:
    __version__ = version("MIRTorch")
except PackageNotFoundError:
    __version__ = "0+unknown"

__all__ = ["__version__", "alg", "dic", "linear", "prox"]
