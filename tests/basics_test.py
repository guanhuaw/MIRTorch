import unittest
import sys, os
path = os.path.dirname(os.path.abspath(__file__))
path = path[:path.rfind('/')]
sys.path.insert(0, path)
from linear import basics
import torch
import torch.nn.functional as F
import numpy.testing as npt

class TestProx(unittest.TestCase):

    def test_conv1d(self):
        x = torch.randn(20, 16, 50)
        weight = torch.randn(33, 16, 3)

        conv = basics.Convolve1d(x.shape, weight, device='cpu')
        out = conv.apply(x)

        exp = F.conv1d(x, weight)
        npt.assert_allclose(out, exp, rtol=1e-3)

if __name__ == '__main__':
    unittest.main()