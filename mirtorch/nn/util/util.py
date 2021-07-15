from __future__ import print_function
import torch
import numpy as np
from PIL import Image
import os
from scipy.interpolate import griddata


# Converts a Tensor into an image array (numpy)
# |imtype|: the desired type of the converted numpy array
def tensor2imdiff(input_image, imtype=np.uint8):
    if isinstance(input_image, torch.Tensor):
        image_tensor = input_image.data
    else:
        return input_image
    image_numpy = image_tensor[0].cpu().float().numpy()
    if image_numpy.shape[0] == 2:
        img = np.sqrt(np.square(image_numpy[1, :, :]) + np.square(image_numpy[0, :, :]))
        image_numpy = np.repeat(img[:, :, np.newaxis], 3, axis=2)
        # print('sdsd')
        # print(np.amax(image_numpy))
        image_numpy = image_numpy * 5
        image_numpy = np.clip(image_numpy, a_min=0, a_max=2.829)
        image_numpy = image_numpy / 2.829 * 255.0
    return image_numpy.astype(imtype)

def tensor2traj(input_image, imtype=np.uint8, sz = 320):
    if isinstance(input_image, torch.Tensor):
        image_tensor = input_image.data
    else:
        return input_image
    image_numpy = image_tensor[0, :, :].cpu().float().numpy()
    image_numpy = np.transpose(np.remainder(image_numpy + np.pi, 2*np.pi))
    image_numpy = stupidgrid(image_numpy, sz)
    # print(np.amax(image_numpy))
    # print(np.amin(image_numpy))
    image_numpy = image_numpy / (np.amax(image_numpy) + 0.0000001) * 255.0
    image_numpy = np.repeat(image_numpy[:, :, np.newaxis], 3, axis=2)
    return image_numpy.astype(imtype)
def stupidgrid(image_numpy, sz):
    image_numpy = image_numpy/(2*np.pi)*(sz-1)
    im = np.zeros((sz,sz))
    im[np.floor(image_numpy[:,0]).astype(int), np.floor(image_numpy[:,1]).astype(int)] = 1
    im[np.ceil(image_numpy[:, 0]).astype(int), np.ceil(image_numpy[:, 1]).astype(int)] = 1
    im[np.ceil(image_numpy[:, 0]).astype(int), np.floor(image_numpy[:, 1]).astype(int)] = 1
    im[np.floor(image_numpy[:, 0]).astype(int), np.ceil(image_numpy[:, 1]).astype(int)] = 1
    return im

def tensor2im(input_image, imtype=np.uint8):
    if isinstance(input_image, torch.Tensor):
        image_tensor = input_image.data
    else:
        return input_image
    image_numpy = image_tensor[0].cpu().float().numpy()
    if len(image_numpy.shape) == 4:
        image_numpy = image_numpy[0]
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    if image_numpy.shape[0] == 3:
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    if image_numpy.shape[0] == 2:
        img = np.sqrt(np.square(image_numpy[1, :, :]) + np.square(image_numpy[0, :, :]))
        upp = np.percentile(img, 99.9)
        lowp = np.percentile(img, 0.1)
        img = np.clip(img, a_min=lowp, a_max=upp)
        image_numpy = np.repeat(img[:, :, np.newaxis], 3, axis=2)
        image_numpy = image_numpy / (np.amax(image_numpy) + 0.0000001) * 255.0

    return image_numpy.astype(imtype)


def tensor2imk(input_image, imtype=np.uint8):
    if isinstance(input_image, torch.Tensor):
        image_tensor = input_image.data
    else:
        return input_image
    image_numpy = image_tensor[0].cpu().float().numpy()
    if len(image_numpy.shape) == 4:
        image_numpy = image_numpy[0]
    img = np.sqrt(np.square(image_numpy[1, :, :]) + np.square(image_numpy[0, :, :]))
    # img = np.fft.fftshift(img)
    img = np.log(img + 0.000001)
    img = (img - np.amin(img)) / (np.amax(img) - np.amin(img))
    image_numpy = np.repeat(img[:, :, np.newaxis], 3, axis=2)
    image_numpy = image_numpy / (np.amax(image_numpy) + 0.0000001) * 255.0
    return image_numpy.astype(imtype)


def diagnose_network(net, name='network'):
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)


def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)


def print_numpy(x, val=True, shp=False):
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def get_phase(batch_input):
    temp = batch_input[:, 2, :, :] / batch_input[:, 1, :, :]
    return temp.unsqueeze(1)


def get_amp(batch_input):
    return torch.sqrt(torch.pow(batch_input[:, 2, :, :], 2) + torch.pow(batch_input[:, 1, :, :], 2))


def generate_mask_alpha(size=[320, 320], r_factor_designed=3.0, r_alpha=3, axis_undersample=1,
                        acs=3, seed=0, mute=0):
    # init
    mask = np.zeros(size)
    if seed >= 0:
        np.random.seed(seed)
    # get samples
    num_phase_encode = size[axis_undersample]
    num_phase_sampled = int(np.floor(num_phase_encode / r_factor_designed))
    # coordinate
    coordinate_normalized = np.array(range(num_phase_encode))
    coordinate_normalized = np.abs(coordinate_normalized - num_phase_encode / 2) / (num_phase_encode / 2.0)
    prob_sample = coordinate_normalized ** r_alpha
    prob_sample = prob_sample / sum(prob_sample)
    # sample
    index_sample = np.random.choice(num_phase_encode, size=num_phase_sampled,
                                    replace=False, p=prob_sample)
    # sample
    if axis_undersample == 0:
        mask[index_sample, :] = 1
    else:
        mask[:, index_sample] = 1
    mask_temp = np.zeros_like(mask)
    # acs
    if axis_undersample == 0:
        mask[:(acs + 1) // 2, :] = 1
        mask[-acs // 2:, :] = 1
    else:
        mask[:, :(acs + 1) // 2] = 1
        mask[:, -acs // 2:] = 1
    # compute reduction
    r_factor = len(mask.flatten()) / sum(mask.flatten())
    if not mute:
        print('gen mask size of {1} for R-factor={0:.4f}'.format(r_factor, mask.shape))
        print(num_phase_encode, num_phase_sampled, np.where(mask[0, :]))

    return mask, r_factor


def generate_mask_beta(size=[320, 320], r_factor_designed=3.0, axis_undersample=1,
                       acs=8, mute=0):
    # init
    mask = np.zeros(size)
    index_sample = range(0, size[0], int(r_factor_designed))
    # sample
    if axis_undersample == 0:
        mask[index_sample, :] = 1
    else:
        mask[:, index_sample] = 1
    mask_temp = np.zeros_like(mask)
    # acs
    if axis_undersample == 0:
        mask[:(acs + 1) // 2, :] = 1
        mask[-acs // 2:, :] = 1
        mask_temp[size[1] // 2:, :] = mask[:size[1] // 2, :]
        mask_temp[:size[1] // 2, :] = mask[size[1] // 2:, :]
    else:
        mask[:, :(acs + 1) // 2] = 1
        mask[:, -acs // 2:] = 1
        mask_temp[:, size[1] // 2:] = mask[:, :size[1] // 2]
        mask_temp[:, :size[1] // 2] = mask[:, size[1] // 2:]
    # compute reduction
    r_factor = len(mask.flatten()) / sum(mask.flatten())
    #    if not mute:
    #        print('gen mask size of {1} for R-factor={0:.4f}'.format(r_factor, mask.shape))
    #        print(num_phase_encode, num_phase_sampled, np.where(mask[0,:]))

    return mask_temp, r_factor


def complex_matmul(a, b):
    # function to multiply two complex variable in pytorch, the real/imag channel are in the third last two channels ((batch), (coil), 2, nx, ny).
    if len(a.size()) == 3:
        return torch.cat(((a[0, ...] * b[0, ...] - a[1, ...] * b[1, ...]).unsqueeze(0),
                          (a[0, ...] * b[1, ...] + a[1, ...] * b[0, ...]).unsqueeze(0)), dim=0)
    if len(a.size()) == 4:
        return torch.cat(((a[:, 0, ...] * b[:, 0, ...] - a[:, 1, ...] * b[:, 1, ...]).unsqueeze(1),
                          (a[:, 0, ...] * b[:, 1, ...] + a[:, 1, ...] * b[:, 0, ...]).unsqueeze(1)), dim=1)
    if len(a.size()) == 5:
        return torch.cat(((a[:, :, 0, ...] * b[:, :, 0, ...] - a[:, :, 1, ...] * b[:, :, 1, ...]).unsqueeze(2),
                          (a[:, :, 0, ...] * b[:, :, 1, ...] + a[:, :, 1, ...] * b[:, :, 0, ...]).unsqueeze(2)), dim=2)


def complex_conj(a):
    # function to multiply two complex variable in pytorch, the real/imag channel are in the last two channels.
    if len(a.size()) == 3:
        return torch.cat((a[0, ...].unsqueeze(0), -a[1, ...].unsqueeze(0)), dim=0)
    if len(a.size()) == 4:
        return torch.cat((a[:, 0, ...].unsqueeze(1), -a[:, 1, ...].unsqueeze(1)), dim=1)
    if len(a.size()) == 5:
        return torch.cat((a[:, :, 0, ...].unsqueeze(2), -a[:, :, 1, ...].unsqueeze(2)), dim=2)


def cplx_to_tensor(data):
    """
    Convert numpy array to PyTorch tensor. For complex arrays, the real and imaginary parts
    are stacked along the last dimension.
    Args:
        data (np.array): Input numpy array
    Returns:
        torch.Tensor: PyTorch version of data
    """
    if np.iscomplexobj(data):
        data = np.stack((data.real, data.imag), axis=-1)
    return torch.from_numpy(data)


def center_crop(data, shape):
    """
    Apply a center crop to the input real image or batch of real images.

    Args:
        data (torch.Tensor): The input tensor to be center cropped. It should have at
            least 2 dimensions and the cropping is applied along the last two dimensions.
        shape (int, int): The output shape. The shape should be smaller than the
            corresponding dimensions of data.

    Returns:
        torch.Tensor: The center cropped image
    """
    assert 0 < shape[0] <= data.shape[-2]
    assert 0 < shape[1] <= data.shape[-1]
    w_from = (data.shape[-2] - shape[0]) // 2
    h_from = (data.shape[-1] - shape[1]) // 2
    w_to = w_from + shape[0]
    h_to = h_from + shape[1]
    return data[..., w_from:w_to, h_from:h_to]


def roll(x, shift, dim):
    """
    Similar to np.roll but applies to PyTorch Tensors
    """
    if isinstance(shift, (tuple, list)):
        assert len(shift) == len(dim)
        for s, d in zip(shift, dim):
            x = roll(x, s, d)
        return x
    shift = shift % x.size(dim)
    if shift == 0:
        return x
    left = x.narrow(dim, 0, x.size(dim) - shift)
    right = x.narrow(dim, x.size(dim) - shift, shift)
    return torch.cat((right, left), dim=dim)


def fftshift(x, dim=None):
    """
    Similar to np.fft.fftshift but applies to PyTorch Tensors
    """
    if dim is None:
        dim = tuple(range(x.dim()))
        shift = [dim // 2 for dim in x.shape]
    elif isinstance(dim, int):
        shift = x.shape[dim] // 2
    else:
        shift = [x.shape[i] // 2 for i in dim]
    return roll(x, shift, dim)


def ifftshift(x, dim=None):
    """
    Similar to np.fft.ifftshift but applies to PyTorch Tensors
    """
    if dim is None:
        dim = tuple(range(x.dim()))
        shift = [(dim + 1) // 2 for dim in x.shape]
    elif isinstance(dim, int):
        shift = (x.shape[dim] + 1) // 2
    else:
        shift = [(x.shape[i] + 1) // 2 for i in dim]
    return roll(x, shift, dim)


def fft2(data):
    """
    Apply centered 2 dimensional Fast Fourier Transform.
    Args:
        data (torch.Tensor): Complex valued input data containing at least 3 dimensions: dimensions
            -3 & -2 are spatial dimensions and dimension -1 has size 2. All other dimensions are
            assumed to be batch dimensions.
    Returns:
        torch.Tensor: The FFT of the input.
    """
    assert data.size(-1) == 2
    data = ifftshift(data, dim=(-3, -2))
    data = torch.fft(data, 2, normalized=True)
    data = fftshift(data, dim=(-3, -2))
    return data


def ifft2(data):
    """
    Apply centered 2-dimensional Inverse Fast Fourier Transform.
    Args:
        data (torch.Tensor): Complex valued input data containing at least 3 dimensions: dimensions
            -3 & -2 are spatial dimensions and dimension -1 has size 2. All other dimensions are
            assumed to be batch dimensions.
    Returns:
        torch.Tensor: The IFFT of the input.
    """
    assert data.size(-1) == 2
    data = ifftshift(data, dim=(-3, -2))
    data = torch.ifft(data, 2, normalized=True)
    data = fftshift(data, dim=(-3, -2))
    return data

def imag_exp(a, dim=0):
    """Imaginary exponential, exp(ia), returns real/imag separate in dim.

    Args:
        a (tensor): A tensor where dimension dim is the complex dimension.
        dim (int, default=0): An integer indicating the complex dimension of
            the output.

    Returns:
        tensor: c = exp(i*a), where i is sqrt(-1).
    """
    c = torch.stack((torch.cos(a), torch.sin(a)), dim)

    return c

def conj_complex_mult(a, b, dim=0):
    """Complex multiplication, real/imag are in dimension dim.

    Args:
        a (tensor): A tensor where dimension dim is the complex dimension.
        b (tensor): A tensor where dimension dim is the complex dimension.
        dim (int, default=0): An integer indicating the complex dimension.

    Returns:
        tensor: c = a * conj(b), where * executes complex multiplication.
    """
    assert a.shape[dim] == 2
    assert b.shape[dim] == 2

    real_a = a.select(dim, 0)
    imag_a = a.select(dim, 1)
    real_b = b.select(dim, 0)
    imag_b = b.select(dim, 1)

    c = torch.stack(
        (real_a*real_b + imag_a*imag_b, imag_a*real_b - real_a*imag_b),
        dim
    )

    return c

def inner_product(a, b, dim=0):
    """Complex inner product, complex dimension is dim.

    Args:
        a (tensor): A tensor where dimension dim is the complex dimension.
        b (tensor): A tensor where dimension dim is the complex dimension.
        dim (int, default=0): An integer indicating the complex dimension.

    Returns:
        tensor: The complex inner product of a and b of size 2 (real, imag).
    """
    assert a.shape[dim] == 2
    assert b.shape[dim] == 2

    inprod = conj_complex_mult(b, a, dim=dim)

    real_inprod = inprod.select(dim, 0).sum()
    imag_inprod = inprod.select(dim, 1).sum()

    return torch.cat((real_inprod.view(1), imag_inprod.view(1)))

### SHOULD BE SUPER CAREFUL WHEN USING COS SQRT(0) GIVE NAN WHEN BACKPROPAGATION
# def normal_mean(a, dim=2):
#     assert a.shape[dim] == 2
#     return a/torch.mean(absolute(a, dim=dim))
#
# def normal_max(a, dim=2):
#     assert a.shape[dim] == 2
#     return a/torch.max(absolute(a, dim=dim))

def absolute(t, dim=0):
    """Complex absolute value, complex dimension is dim.

    Args:
        t (tensor): A tensor where dimension dim is the complex dimension.
        dim (int, default=0): An integer indicating the complex dimension.

    Returns:
        tensor: The absolute value of t.
    """
    assert t.shape[dim] == 2

    abst = torch.sqrt(
        t.select(dim, 0) ** 2 +
        t.select(dim, 1) ** 2
    ).unsqueeze(dim)

    return abst


def complex_sign(t, dim=0):
    """Complex sign function value, complex dimension is dim.

    Args:
        t (tensor): A tensor where dimension dim is the complex dimension.
        dim (int, default=0): An integer indicating the complex dimension.

    Returns:
        tensor: The complex sign of t.
    """
    assert t.shape[dim] == 2

    signt = torch.atan2(t.select(dim, 1), t.select(dim, 0))
    signt = imag_exp(signt, dim=dim)

    return signt