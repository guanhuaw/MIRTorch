import unittest
import sys, os
path = os.path.dirname(os.path.abspath(__file__))
path = path[:path.rfind('/')]
sys.path.insert(0, path)
from linear import basics
import torch
import torch.nn.functional as F

class TestBasic(unittest.TestCase):

    # ？
    # def test_diag(self):
    #     x = torch.randn(5, 5)
    #     P = torch.randn(5, 5)

    #     diag = basics.Diag(P)
    #     out = diag.apply(x)

    #     exp = torch.diag(P.reshape(-1,1)) * x.reshape(-1,1)
    #     assert(torch.allclose(out, exp, rtol=1e-3))

    def test_conv1d_apply(self):
        x = torch.randn(20, 16, 50)
        weight = torch.randn(33, 16, 3)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        x, weight = x.to(device), weight.to(device)

        conv = basics.Convolve1d(x.shape, weight, device=device)
        out = conv.apply(x)

        exp = F.conv1d(x, weight)
        assert(torch.allclose(out, exp, rtol=1e-3))

    def test_conv2d_apply(self):
        x = torch.randn(1, 4, 5, 5)
        weight = torch.randn(8, 4, 3, 3)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        x, weight = x.to(device), weight.to(device)

        conv = basics.Convolve2d(x.shape, weight, padding=1, device=device)
        out = conv.apply(x)

        exp = F.conv2d(x, weight, padding=1)
        assert(torch.allclose(out, exp, rtol=1e-3))

    def test_conv3d_apply(self):
        x = torch.randn(20, 16, 50, 10, 20)
        weight = torch.randn(33, 16, 3, 3, 3)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        x, weight = x.to(device), weight.to(device)

        conv = basics.Convolve3d(x.shape, weight, device=device)
        out = conv.apply(x)

        exp = F.conv3d(x, weight)
        assert(torch.allclose(out, exp, rtol=1e-3))

    # ？
    # def test_conv1d_adjoint(self):
    #     x = torch.randn(20, 16, 50)
    #     weight = torch.randn(16, 33, 5)
    #     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #     x, weight = x.to(device), weight.to(device)

    #     conv = basics.Convolve1d(x.shape, weight, device=device)
    #     out = conv.adjoint(x)

    #     exp = F.conv_transpose1d(x, weight)
    #     assert(torch.allclose(out, exp, rtol=1e-3))

if __name__ == '__main__':
    unittest.main()