import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp
import math
from . import util
import scipy.ndimage


# Simulate small-angle affine transform in Python (call Numpy pkg, waste more time than
# direct implementation in torch).
# input: torch tensor of k-space [1,2,nx,ny]
# output: torch tensor of k-space [1,2,nx,ny]
def simu_Affine2D(raw_k, Trans_x, Trans_y, Angle):
    raw_I = torch.squeeze(
        torch.fft(raw_k.permute(0, 2, 3, 1), signal_ndim=2, normalized=True).permute(0, 3, 1, 2)).cpu().float().numpy()
    distoted_I = scipy.ndimage.rotate(
        scipy.ndimage.shift(raw_I, [0, (-0.5 + np.random.rand()) * Trans_x, (-0.5 + np.random.rand()) * Trans_y]),
        (-0.5 + np.random.rand()) * Angle, axes=(1, 2), reshape=False)
    distorted_k = torch.ifft(torch.Tensor(distoted_I).unsqueeze(0).permute(0, 2, 3, 1), signal_ndim=2,
                             normalized=True).permute(0, 3, 1, 2)
    return distorted_k


# Add gaussian white noise in the k-space
# input: torch tensor of k-space [1,2,nx,ny]
# output: torch tensor of k-space [1,2,nx,ny]
def simu_noise2D(raw_k, noise_level):
    noisy_k = raw_k + noise_level * torch.randn_like(raw_k)
    return noisy_k
