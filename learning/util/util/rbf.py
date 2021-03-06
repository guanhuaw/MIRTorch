import torch
import torch.nn.functional as F
import numpy as np
import pykeops
import pykeops.numpy as pknp
import pykeops.torch as pktorch
from pykeops.torch import LazyTensor
pykeops.build_type = 'Debug'

def rbf(rbf_weights, im, sigma, range_u):
    num_batch, num_kernels, nx, ny = im.size()
    _, num_rbf = rbf_weights.size()
    im_r = torch.zeros_like(im)
    im1 = im.expand(num_rbf,-1,-1,-1,-1) # num_rbf, num_batch, num_kernels, nx, ny
    # print('im', im1.size())
    u = torch.arange(-range_u,range_u,2*range_u/num_rbf, dtype = im.dtype, device = im.device).expand(1, 1, 1, 1, -1).permute(4,0,1,2,3)
    # print('u', u.size())
    w = rbf_weights.expand(1, 1, 1, -1, -1).permute(4,0,3,1,2)
    # print('w', w.size())
    return torch.sum(w*torch.exp(-1 * (im1 - u).pow(2) / sigma),0)
    # for ii in torch.arange(num_rbf):
    #     u = -range_u + ii*(2*range_u)/num_rbf
    #     w = rbf_weights[:,ii].expand(1, 1, 1, -1).permute(0,3,1,2)
    #     im_r = im_r + w*torch.exp(-1 * (im - u).pow(2) / sigma)
    # after test: the for loop and repeat save no more memory ....
    # return im_r
def rbf_keops(rbf_weights, im, sigma, range_u):
    num_batch, num_kernels, nx, ny = im.size()
    _, num_rbf = rbf_weights.size()
    im1 = LazyTensor(im.expand(1,-1,-1,-1,-1).permute(1,2,0,3,4).reshape(num_batch,num_kernels,1, nx*ny, 1)) # num_batch, num_kernels, num_rbf, nx*ny, 1
    u = LazyTensor(torch.arange(-range_u,range_u,2*range_u/num_rbf, dtype = im.dtype, device = im.device).expand(1, 1, 1, 1, -1).permute(0,1,4,2,3))
    w = LazyTensor(rbf_weights.expand(1, 1, 1, -1, -1).permute(0,3,4,1,2))
    lazysum = (w*(((-1 * (im1 - u)**2) / sigma).exp())).sum(dim=2)
    im_resize = lazysum.reshape(num_batch,num_kernels,nx,ny)
    return im_resize


def gaussian(alpha):
    phi = torch.exp(-1 * alpha.pow(2))
    return phi


def linear(alpha):
    phi = alpha
    return phi


def quadratic(alpha):
    phi = alpha.pow(2)
    return phi


def inverse_quadratic(alpha):
    phi = torch.ones_like(alpha) / (torch.ones_like(alpha) + alpha.pow(2))
    return phi


def multiquadric(alpha):
    phi = (torch.ones_like(alpha) + alpha.pow(2)).pow(0.5)
    return phi


def inverse_multiquadric(alpha):
    phi = torch.ones_like(alpha) / (torch.ones_like(alpha) + alpha.pow(2)).pow(0.5)
    return phi


def spline(alpha):
    phi = (alpha.pow(2) * torch.log(alpha + torch.ones_like(alpha)))
    return phi


def poisson_one(alpha):
    phi = (alpha - torch.ones_like(alpha)) * torch.exp(-alpha)
    return phi


def poisson_two(alpha):
    phi = ((alpha - 2 * torch.ones_like(alpha)) / 2 * torch.ones_like(alpha)) \
          * alpha * torch.exp(-alpha)
    return phi


def matern32(alpha):
    phi = (torch.ones_like(alpha) + 3 ** 0.5 * alpha) * torch.exp(-3 ** 0.5 * alpha)
    return phi


def matern52(alpha):
    phi = (torch.ones_like(alpha) + 5 ** 0.5 * alpha + (5 / 3) \
           * alpha.pow(2)) * torch.exp(-5 ** 0.5 * alpha)
    return phi


def basis_func_dict():
    """
    A helper function that returns a dictionary containing each RBF
    """

    bases = {'gaussian': gaussian,
             'linear': linear,
             'quadratic': quadratic,
             'inverse quadratic': inverse_quadratic,
             'multiquadric': multiquadric,
             'inverse multiquadric': inverse_multiquadric,
             'spline': spline,
             'poisson one': poisson_one,
             'poisson two': poisson_two,
             'matern32': matern32,
             'matern52': matern52}
    return bases
