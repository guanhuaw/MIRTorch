import math

import numpy as np
from scipy.fft import idct


def idct_basis_2d(len_basis, num_basis):
    """
    Generate basic 2D DCT basis for dictionary learning

    Args:
        len_basis: length of the flattened atom, e.g. 36 for 6x6 basis
        num_basis: number of the atoms. usually it is overcomplete (larger than len_basis)

    Returns:
        DCT basis in [len_basis, num_basis]

    """
    if len_basis > num_basis:
        raise ValueError("len_basis should be smaller than num_basis")
    ODCT = idct(np.identity(math.ceil(num_basis**0.5)), norm="ortho", axis=0)
    ODCT = ODCT[: math.ceil(len_basis**0.5), :]
    ODCT = np.kron(ODCT, ODCT)
    ODCT = ODCT[:len_basis, :num_basis]
    ODCT = np.column_stack((ODCT[:, 0], ODCT[:, 1:] - np.mean(ODCT[:, 1:], axis=0)))
    ODCT = ODCT / np.linalg.norm(ODCT, axis=0)
    return ODCT


def idct_basis_3d(len_basis, num_basis):
    """
    Generate basic 3D DCT basis for dictionary learning

    Args:
        len_basis: length of the flattened atom, e.g. 216 for 6x6x6 basis
        num_basis: number of the atoms. usually it is overcomplete (larger than len_basis)

    Returns:
            DCT basis in [len_basis, num_basis]

    """
    if len_basis > num_basis:
        raise ValueError("len_basis should be smaller than or equal to num_basis")
    ODCT = idct(np.identity(math.ceil(num_basis ** (1 / 3))), norm="ortho", axis=0)
    ODCT = ODCT[: math.ceil(len_basis ** (1 / 3)), :]
    ODCT = np.kron(ODCT, np.kron(ODCT, ODCT))
    ODCT = ODCT[:len_basis, :num_basis]
    ODCT = np.column_stack((ODCT[:, 0], ODCT[:, 1:] - np.mean(ODCT[:, 1:], axis=0)))
    ODCT = ODCT / np.linalg.norm(ODCT, axis=0)
    return ODCT
