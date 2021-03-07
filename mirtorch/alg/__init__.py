# I guess it is not a bad idea to not provide a overall parent class for
# the iterative algorithms.
# The reason is about auto-differentiation, etc.
# Doing it case by cases seems to be more convenient.
from .cg import CG
from .spectral import power_iter

__all__ = ["CG", "power_iter"]