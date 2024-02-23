import logging

from .cg import CG
from .spectral import power_iter
from .pogm import POGM
from .fista import FISTA
from .fbpd import FBPD

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.info("Logger initialized in __init__.py")

__all__ = ["CG", "power_iter", "POGM", "FISTA", "FBPD"]
