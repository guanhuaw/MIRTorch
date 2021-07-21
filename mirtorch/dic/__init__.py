# I guess it is not a bad idea to not provide a overall parent class for
# the iterative algorithms.
# The reason is about auto-differentiation, etc.
# Doing it case by cases seems to be more convenient.
from .soup import SOUPDIL
from .omp import OMP

__all__ = ["soupdil.py", "omp.py"]