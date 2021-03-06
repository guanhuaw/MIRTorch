"""
Copyright (c) 2019 Imperial College London.
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def laplacian_of_gaussian_2d(window_size, sigma):
    # 2d gaussian
    log = np.zeros((window_size, window_size))
    sd = sigma * sigma
    for x in range(window_size):
        for y in range(window_size):
            x_sq = (x - window_size//2)**2 + (y - window_size//2)**2
            log[x,y] = (x_sq / (2*sd) - 1) / (np.pi* sd**2) * np.exp(-x_sq/(2*sd))

    return torch.from_numpy(log)

def create_window(window_size, channel, sigma=1.5):
    _2D_window = laplacian_of_gaussian_2d(window_size, sigma).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


def _hfen(img1, img2, window, window_size, channel, size_average=True, full=False):
    padd = 0
    bs = len(img1)
    # compute laplacian of gaussian
    LoG1 = F.conv2d(abs(img1), window, padding=padd, groups=channel).reshape((bs, -1))
    LoG2 = F.conv2d(abs(img2), window, padding=padd, groups=channel).reshape((bs, -1))

    hfen_norm =  (LoG1 - LoG2).norm(dim=1) / LoG1.norm(dim=1)

    return hfen_norm.mean()


class HFEN(torch.nn.Module):
    def __init__(self, window_size=13, sigma=1.5, size_average=True, device='cuda'):
        super(HFEN, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.sigma = sigma
        self.channel = 1
        self.window = create_window(window_size, self.channel, self.sigma)
        self.device = device

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel, self.sigma)
            window = window.to(self.device).type_as(img1)

            self.window = window
            self.channel = channel

        return _hfen(img1, img2, window, self.window_size, channel, self.size_average)


def hfen(img1, img2, window_size=11, size_average=True, full=False, device='cuda'):
    (_, channel, height, width) = img1.size()

    real_size = min(window_size, height, width)
    window = create_window(real_size, channel)

    window = window.to(device).type_as(img1)

    return _hfen(img1, img2, window, real_size, channel, size_average, full=full)