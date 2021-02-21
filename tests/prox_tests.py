import unittest
import sys, os
path = os.path.dirname(os.path.abspath(__file__))
path = path[:path.rfind('/')]
sys.path.insert(0, path)
from prox import Prox
import torch
import numpy as np
import numpy.testing as npt

class TestProx(unittest.TestCase):

    def test_l1(self):
        lambd = np.random.random()
        prox = Prox.L1Regularizer(lambd)
        a = torch.rand((5,4,8), dtype=torch.float)
        exp = np.zeros((5,4,8)).flatten()
        out = prox(a)

        a = a.numpy().flatten()
        for i in range(a.shape[0]):
            if a[i]>lambd:
                exp[i] = a[i] - lambd
            elif a[i]<-lambd:
                exp[i] = a[i] + lambd
            else:
                exp[i] = 0

        exp = exp.reshape((5,4,8))

        npt.assert_allclose(out, exp, rtol=1e-3)

    def test_l2(self):
        a = torch.rand((3, 4, 2, 1), dtype=torch.float)
        lambd = np.random.random()
        prox = Prox.L2Regularizer(lambd)
        exp = 1.0 - lambd/max(np.linalg.norm(a.numpy()),lambd)
        npt.assert_allclose(prox(a), exp*a.numpy(), rtol=1e-3)

    def test_squaredl2(self):
        a = torch.rand((3, 4, 2, 1), dtype=torch.float)
        lambd = np.random.random()
        prox = Prox.SquaredL2Regularizer(lambd)
        exp = a.numpy()/(1.0 + 2*lambd)
        npt.assert_allclose(prox(a), exp, rtol=1e-3)

    def test_boxconstraint(self):
        lambd = np.random.random()
        lower, upper = np.random.randint(0, 10), np.random.randint(10, 20)
        prox = Prox.BoxConstraint(lambd, lower, upper)
        a = 100*torch.rand((5,4,8), dtype=torch.float)
        out = prox(a)
        exp = np.clip(a.numpy(),lower,upper)
        npt.assert_allclose(out, exp, rtol=1e-3)

    def test_l1_complex(self):
        lambd = np.random.random()
        prox = Prox.L1Regularizer(lambd)
        a = torch.rand(2,2, dtype=torch.cfloat, requires_grad=True)
        out = prox(a)
        torch.sum(out).backward()
        a.requires_grad=False
        exp = torch.exp(1j*a.angle())*prox(a.abs())
        npt.assert_allclose(out.detach(), exp, rtol=1e-3)

    def test_l2_complex(self):
        lambd = np.random.random()
        prox = Prox.L2Regularizer(lambd)
        a = torch.rand(2,2, dtype=torch.cfloat, requires_grad=True)
        out = prox(a)
        torch.sum(out).backward()
        a.requires_grad=False
        exp = torch.exp(1j*a.angle())*prox(a.abs())
        npt.assert_allclose(out.detach(), exp, rtol=1e-3)


    def test_squaredl2_complex(self):
        lambd = np.random.random()
        prox = Prox.SquaredL2Regularizer(lambd)
        a = torch.rand(2,2, dtype=torch.cfloat, requires_grad=True)
        out = prox(a)
        torch.sum(out).backward()
        a.requires_grad=False
        exp = torch.exp(1j*a.angle())*prox(a.abs())
        npt.assert_allclose(out.detach(), exp, rtol=1e-3)


    def test_angle(self):
        a = torch.complex(torch.Tensor([1]), torch.Tensor([-1]))
        npt.assert_allclose(a.angle(), torch.atan2(a.imag,a.real))

    def test_boxconstraint_complex(self):
        lambd = np.random.random()
        lower, upper = np.random.randint(0, 10), np.random.randint(10, 20)
        prox = Prox.BoxConstraint(lambd, lower, upper)
        a = torch.rand(2,2, dtype=torch.cfloat, requires_grad=True)
        out = prox(a)
        torch.sum(out).backward()
        a.requires_grad=False
        exp = torch.exp(1j*a.angle())*prox(a.abs())
        npt.assert_allclose(out.detach(), exp, rtol=1e-3)

    def test_complex_edge_cases(self):
        a = torch.complex(torch.Tensor([1]),torch.Tensor([0]))
        npt.assert_allclose(a.angle(), torch.atan2(a.imag,a.real))

    def test_complex_edge_cases2(self):
        a = torch.complex(torch.Tensor([0]),torch.Tensor([1]))
        npt.assert_allclose(a.angle(), torch.atan2(a.imag,a.real))

    def test_complex_edge_cases3(self):
        #Should we ever need to worry about this issue?
        a = torch.complex(torch.Tensor([0]),torch.Tensor([0]))
        npt.assert_allclose(a.angle(), torch.atan2(a.imag,a.real))




if __name__ == '__main__':
    print(f'PyTorch version: {torch.__version__}')
    unittest.main()

