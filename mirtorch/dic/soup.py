import numpy as np
import scipy as sc
import scipy.sparse as sp
import numpy.random as random
import torch

def SOUPDIL(Y, D0, X0, lambd, numiter, rnd=True, only_sp=False):
    '''
    Efficient patch-based dictionary learning algorithm according to:
    Ravishankar, S., Nadakuditi, R. R., & Fessler, J. A. (2017). Efficient sum of outer products dictionary learning (SOUP-DIL)
    and its application to inverse problems. IEEE transactions on computational imaging, 3(4), 694-709.
    Generally, we are trying to solve this problem:
        \argmin{D, X} \|Y-DX\|_2^2 + \lambda \|X\|_0 (the '0-norm' means the number of non-zero elements across the whole matrix)
    Inputs:
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
    Migrate back to torch when its CSR/CSC gets better.
    The algorithm involves frequent update of sparse data; using GPU may not necessarily accelerate it.
    '''
    D = D0
    [len_atom, num_atom] = D0.shape

    # Z is a basic element
    Z = np.zeros(len_atom).astype(D.dtype)
    Z[0] = 1

    # [num_patch, num_atom], each column is the sparse code for a atom (across patches)
    # Note the Hermitian here
    # CSR is more efficient for matrix-vector product
    C = sp.csr_matrix(X0.T.conj())

    # Outer loop: iterations
    for iouter in range(numiter):

        # Compute for just once
        YtD = np.matmul(Y.T.conj(), D)

        # Find the non-zero components
        [idx_row, idx_col] = np.nonzero(C)  # [num_nonzero(row), num_nonzero(column)]

        # Inner loop: atoms
        for iatom in range(num_atom):

            # Update of the sparse code: hard-thresholding
            # TODO: add soft-thresholding
            Ytdj = YtD[:, iatom]
            # cj = C[:, iatom] # slicing w/ CSR is inefficient
            b = Ytdj - C.dot((D.T.conj()).dot(D[:, iatom]))

            # avoid column slicing, see https://stackoverflow.com/a/50860352
            idx_col_j = idx_col[idx_col == iatom]
            idx_row_j = idx_row[idx_col == iatom]
            b[idx_row_j] += C[idx_row_j, idx_col_j]

            # hard-thresholding
            cj_new = b * (np.abs(b) > lambd)
            [idx_row_new] = np.nonzero(cj_new)

            # Update of the dictionary
            if ~only_sp:
                if cj_new.abs().sum() == 0:
                    if rnd:
                        h = random.randn(len_atom).astype(D.dtype)
                    else:
                        h = Z
                else:
                    h = np.zeros(len_atom).astype(D.dtype)
                    h[idx_row_new] = Y[:, idx_row_new].dot(cj_new[idx_row_new])
                    h += -D.dot(C[idx_row_new, :].T.dot(cj_new[idx_row_new]))
                    [idx_ovlp, idx_ovlp_new, idx_ovlp_j] = np.intersect1d(idx_row_new, idx_row_j)
                    h += D[:, iatom] * (C[idx_row_j, idx_col_j][idx_ovlp_j].conj().dot(cj_new[idx_ovlp_new]))
                    # h += -D.dot(X.dot(cj_new)) + D[:, iatom] * ((cj.conj()).dot(cj_new)) # inefficient manner
                    h = h / np.linalg.norm(h, 2)
                D[:, iatom] = h

            # avoid column slicing, again
            # remember to eliminate untracked zeros in the end
            C[idx_row_j, iatom] = 0
            C[idx_row_new, iatom] = cj_new[idx_row_new]
        C.eliminate_zeros()
    return D, C.T.conj(), (C.dot(D.T.conj())).T.conj()
