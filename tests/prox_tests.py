import unittest
import sys, os
path = os.path.dirname(os.path.abspath(__file__))
path = path[:path.rfind('/')]
sys.path.insert(0, path)
from LinearMap import Prox
import torch
import numpy as np
import numpy.testing as npt

class TestProx(unittest.TestCase):
    
    def test_l1(self):
        lambd = np.random.random()
        prox = Prox.L1Regularizer(lambd)
        a = torch.rand((5,4,8))
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
        npt.assert_allclose(out.numpy(), exp, rtol=1e-3)
        #TODO: grad can only be created for scalar outputs, so can't explicitly test gradients
        #np.assert_allclose(out.grad, exp_grad) 

    def test_l2(self):
        a = torch.rand((3, 4, 2, 1))
        lambd = np.random.random()
        prox = Prox.L2Regularizer(lambd)
        exp = 1.0 - lambd/max(np.linalg.norm(a.numpy()),lambd)
        npt.assert_allclose(prox(a), exp*a.numpy(), rtol=1e-3)

    def test_squaredl2(self):
        a = torch.rand((3, 4, 2, 1))
        lambd = np.random.random()
        prox = Prox.SqauredL2Regularizer(lambd)
        exp = a.numpy()/(1.0 + 2*lambd)
        npt.assert_allclose(prox(a), exp)

    def test_boxconstraint(self):
        lambd = np.random.random()
        lower, upper = np.random.randint(0, 10), np.random.randint(10, 20)
        prox = Prox.BoxConstraint(lambd, lower, upper)
        a = 100*torch.rand((5,4,8))
        out = prox(a)
        exp = np.clip(a.numpy(),lower,upper)
        npt.assert_allclose(out, exp, rtol=1e-3)

if __name__ == '__main__':
    unittest.main()

