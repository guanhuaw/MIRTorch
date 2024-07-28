# # spect_tests.py
# """
# Adjoint tests for SPECT forward-backward projector
# Author: Zongyu Li, zonyul@umich.edu
# """

# import unittest
# import sys
# import os

# path = os.path.dirname(os.path.abspath(__file__))
# path = path[: path.rfind("/")]
# sys.path.insert(0, path)
# from mirtorch.linear.spect import SPECT
# import torch


# def gen_data(nx, ny, nz, nview, px, pz):
#     img = torch.zeros(nx, ny, nz)
#     img[1:-1, 1:-1, 1:-1] = torch.rand(nx - 2, ny - 2, nz - 2)
#     view = torch.zeros(nx, nz, nview)
#     view[1:-1, 1:-1, 1:-1] = torch.rand(nx - 2, nz - 2, nview - 2)
#     mumap = torch.zeros(nx, ny, nz)
#     mumap[1:-1, 1:-1, 1:-1] = torch.rand(nx - 2, ny - 2, nz - 2)
#     psfs = torch.ones(px, pz, ny, nview) / (px * pz)
#     return img, view, mumap, psfs


# class TestSPECT(unittest.TestCase):
#     def test_adjoint(self):
#         torch.manual_seed(42)
#         nx = 8
#         ny = 8
#         nz = 6
#         nview = 9
#         px = 3
#         pz = 3
#         dy = 4.8

#         img, view, mumap, psfs = gen_data(nx, ny, nz, nview, px, pz)
#         SPECT_sys = SPECT(
#             size_in=(nx, ny, nz),
#             size_out=(nx, nz, nview),
#             mumap=mumap,
#             psfs=psfs,
#             dy=dy,
#         )
#         out1 = SPECT_sys * img
#         out2 = SPECT_sys.H * view

#         test1 = torch.dot(out1.reshape(-1), view.reshape(-1))
#         test2 = torch.dot(out2.reshape(-1), img.reshape(-1))
#         assert torch.allclose(test1, test2, rtol=5e-3)


# if __name__ == "__main__":
#     t = TestSPECT()
#     t.test_adjoint()

import pytest
import torch
from mirtorch.linear.spect import SPECT, project, backproject

@pytest.fixture
def mumap():
    return torch.rand((32, 32, 32), dtype=torch.float32)

@pytest.fixture
def psfs():
    return torch.rand((16, 16, 32, 60), dtype=torch.float32)

@pytest.fixture
def dy():
    return 1.0

@pytest.fixture
def input_tensor(mumap):
    return torch.rand(mumap.shape, dtype=torch.float32)

@pytest.fixture
def view_tensor(psfs):
    return torch.rand((32, 32, 60), dtype=torch.float32)

def test_spect_init(mumap, psfs, dy):
    size_in = mumap.shape
    size_out = [mumap.shape[0], mumap.shape[2], psfs.shape[-1]]
    spect = SPECT(size_in, size_out, mumap, psfs, dy)
    assert spect.mumap.shape == mumap.shape
    assert spect.psfs.shape == psfs.shape
    assert spect.dy == dy

def test_project(input_tensor, mumap, psfs, dy):
    views = project(input_tensor, mumap, psfs, dy)
    assert views.shape == (32, 32, 60)

def test_backproject(view_tensor, mumap, psfs, dy):
    image = backproject(view_tensor, mumap, psfs, dy)
    assert image.shape == mumap.shape

def test_spect_apply(input_tensor, mumap, psfs, dy):
    size_in = mumap.shape
    size_out = [mumap.shape[0], mumap.shape[2], psfs.shape[-1]]
    spect = SPECT(size_in, size_out, mumap, psfs, dy)
    result = spect._apply(input_tensor)
    assert result.shape == (32, 32, 60)

def test_spect_apply_adjoint(view_tensor, mumap, psfs, dy):
    size_in = mumap.shape
    size_out = [mumap.shape[0], mumap.shape[2], psfs.shape[-1]]
    spect = SPECT(size_in, size_out, mumap, psfs, dy)
    result = spect._apply_adjoint(view_tensor)
    assert result.shape == mumap.shape
