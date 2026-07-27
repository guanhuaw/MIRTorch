import numpy as np
import scipy.sparse as sp

from mirtorch.dic import soup


def test_soup_only_sparse_code_keeps_dictionary_fixed():
    data = np.array(
        [
            [1.0, 0.2, -0.8, 0.5],
            [0.1, 1.0, 0.4, -0.6],
        ]
    )
    dictionary = np.eye(2)
    initial_dictionary = dictionary.copy()
    coefficients = sp.csr_matrix((2, data.shape[1]), dtype=data.dtype)

    learned_dictionary, sparse_code, reconstruction = soup(
        data,
        dictionary,
        coefficients,
        lambd=0.1,
        numiter=1,
        only_sp=True,
    )

    np.testing.assert_array_equal(learned_dictionary, initial_dictionary)
    assert sparse_code.shape == (2, data.shape[1])
    assert reconstruction.shape == data.shape
