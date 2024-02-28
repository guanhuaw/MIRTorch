import numpy as np
import scipy.sparse as sp
import numpy.random as random
import time
import logging

logger = logging.getLogger(__name__)


def soup(Y, D0, X0, lambd, numiter, rnd=False, only_sp=False, alert=False):
    r"""
    Efficient patch-based dictionary learning algorithm according to:
    Ravishankar, S., Nadakuditi, R. R., & Fessler, J. A. (2017). Efficient sum of outer products dictionary learning (SOUP-DIL)
    and its application to inverse problems. IEEE transactions on computational imaging, 3(4), 694-709.

    Generally, the algorithm solves the following problem:

    .. math::

         arg \min_{D, X} \|Y-DX\|_2^2 + \lambda \|X\|_0.

    (the '0-norm' means the number of non-zero elements across the whole matrix)

    Args:
        Y: [len_atom, num_patch] data matrix with (self-)training signal as columns (real/complex, numpy matrix)
        D0: [len_atom, num_atom] initial dictionary (real/complex, numpy matrix)
        X0: [num_atom, num_patch] initial sparse code (real/complex, should be numpy (sparse) matrix)
        rnd: when the atom update is non-unique, choose whether to use first column or a random column
        lambd: the sparsity weight
        numiter: number of iterations
        only_sp: only update sparse code

    Returns:
        D: learned dictionary
        X: sparse code
        DX: estimated results

    Since torch.sparse is still very basic, we chose SciPy as the ad-hoc backend.
    Use Tensor.cpu().numpy() and torch.from_numpy to avoid memory relocation
    TODO(guanhuaw@umich.edu): Migrate back to torch when its CSR/CSC gets better.
    The algorithm involves frequent update of sparse data; using GPU may not necessarily accelerate it.
    2021-06. Guanhua Wang, University of Michigan
    """

    assert (
        Y.dtype == X0.dtype == D0.dtype
    ), "datatype (complex/real) between dictionary and sparse code should stay the same!"
    D = D0
    [len_atom, num_atom] = D0.shape
    [_, num_patch] = X0.shape

    # Z is a unit element
    Z = np.zeros(len_atom).astype(D.dtype)
    Z[0] = 1

    # [num_patch, num_atom], each column is the sparse code for a atom (across patches)
    # Note the Hermitian here
    # CSR is more efficient for matrix-vector product, but less efficient for slicing
    C = sp.csr_matrix(X0.T.conj())

    # Outer loop: iterations
    for iouter in range(numiter):
        # Compute for just once
        YtD = np.matmul(Y.T.conj(), D)

        # Find the non-zero components
        [idx_row, idx_col] = np.nonzero(C)  # [num_nonzero(row), num_nonzero(column)]

        # Inner loop: atoms
        for iatom in range(num_atom):
            start = time.time()
            # Update of the sparse code: hard-thresholding
            # TODO: add soft-thresholding
            Ytdj = YtD[:, iatom]

            b = Ytdj - C.tocsr().dot((D.T.conj()).dot(D[:, iatom]))

            # cj = C[:, iatom] # slicing w/ CSR is inefficient
            # avoid column slicing, see https://stackoverflow.com/a/50860352
            idx_col_j = idx_col[idx_col == iatom]
            idx_row_j = idx_row[idx_col == iatom]
            if idx_row_j.size != 0:
                b[idx_row_j] += np.squeeze(np.asarray(C[idx_row_j, idx_col_j]))

            # hard-thresholding
            cj_new = b * (np.abs(b) > lambd)
            [idx_row_new] = np.nonzero(cj_new)

            # Update of the dictionary
            if ~only_sp:
                if np.abs(cj_new).sum() == 0:
                    if rnd:
                        h = random.randn(len_atom).astype(D.dtype)
                    else:
                        h = Z
                else:
                    h = np.zeros(len_atom).astype(D.dtype)

                    h = Y[:, idx_row_new].dot(cj_new[idx_row_new])

                    # h += -D.dot(X.dot(cj_new)) + D[:, iatom] * ((cj.conj()).dot(cj_new)) # inefficient manner
                    h = h - D.dot(C[idx_row_new, :].T.conj().dot(cj_new[idx_row_new]))
                    idx_ovlp, idx_ovlp_new, idx_ovlp_j = np.intersect1d(
                        idx_row_new, idx_row_j, return_indices=True
                    )
                    if idx_ovlp.size != 0:
                        h += (
                            D[:, iatom]
                            * (cj_new[idx_ovlp] * (C[idx_ovlp, iatom].conj())).item()
                        )
                h = h / np.linalg.norm(h, 2)
                D[:, iatom] = h

            # avoid column slicing, again
            # remember to eliminate untracked zeros in the end
            C[idx_row_j, iatom] = 0
            C[idx_row_new, iatom] = cj_new[idx_row_new]
            if alert:
                logger.info(
                    "Update of %dth atom costs %4f s," % (iatom, time.time() - start),
                    "sparse ratio from %5f to %5f"
                    % (len(idx_row_j) / num_patch, len(idx_row_new) / num_patch),
                )
        C.eliminate_zeros()
    return D, C.T.conj(), (C.dot(D.T.conj())).T.conj()
