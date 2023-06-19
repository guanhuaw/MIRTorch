# spect_tests.py
"""
Adjoint tests for SPECT forward-backward projector
Author: Zongyu Li, zonyul@umich.edu
"""

import unittest
import sys, os
path = os.path.dirname(os.path.abspath(__file__))
path = path[:path.rfind('/')]
sys.path.insert(0, path)
from mirtorch.linear.spect import SPECT
import torch
import numpy as np


def gen_data(nx, ny, nz, nview, px, pz):
    img = torch.zeros(nx, ny, nz)
    img[1:-1,1:-1,1:-1] = torch.rand(nx-2,ny-2,nz-2)
    view = torch.zeros(nx, nz, nview)
    view[1:-1,1:-1,1:-1] = torch.rand(nx-2,nz-2,nview-2)
    mumap = torch.zeros(nx, ny, nz)
    mumap[1:-1, 1:-1, 1:-1] = torch.rand(nx - 2, ny - 2, nz - 2)
    psfs = torch.ones(px, pz, ny, nview) / (px * pz)
    return img, view, mumap, psfs


class TestSPECT(unittest.TestCase):

    def test_adjoint(self):
        torch.manual_seed(42)
        nx = 8
        ny = 8
        nz = 6
        nview = 9
        px = 3
        pz = 3
        dy = 4.8

        img, view, mumap, psfs = gen_data(nx, ny, nz, nview, px, pz)
        SPECT_sys = SPECT(size_in=(nx, ny, nz), size_out=(nx, nz, nview),
                          mumap=mumap, psfs=psfs, dy=dy)
        out1 = SPECT_sys * img
        out2 = SPECT_sys.H * view

        test1 = torch.dot(out1.reshape(-1), view.reshape(-1))
        test2 = torch.dot(out2.reshape(-1), img.reshape(-1))
        assert (torch.allclose(test1, test2, rtol=5e-3))


if __name__ == '__main__':
    t = TestSPECT()
    t.test_adjoint()