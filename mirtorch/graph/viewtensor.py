import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Union

def tim(img: Union[torch.Tensor, np.ndarray], viewtype='montage', rows=None, cols=None, offset=None, cmap='gray'):
    """
    Display a 3D image in different modes: montage, mid-slice, or maximum intensity projection.
    
    Parameters:
    img (torch.tensor or numpy.ndarray): 3D image to display. Assuming the image is in the format [(batch), Nx, Ny, Nz]
    viewtype (str): Type of view ('montage', 'mid3', 'mip3').
    rows (int): Number of rows for montage view.
    cols (int): Number of columns for montage view.
    offset (tuple): Offset for mid-slice view.
    cmap (str): Colormap for displaying the image.
    
    Returns:
    None
    """

    # if the image is a torch tensor, convert it to numpy
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().numpy()
    
    if np.iscomplex(img).any():
        print('Warning: image is complex - using absolute value')
        img = np.abs(img)

    # get image size
    if img.ndim == 2: # 2D
        Nx,Ny = img.shape
        Nz = 1
        Nbatch = 1
        img = img.unsqueeze(0).unsqueeze(0)
    else: # >3D
        Nx,Ny,Nz = img.shape[-3:]
        Nbatch = np.prod(img.shape[:-3])
        img = img.reshape((Nbatch,Nz,Nx,Ny))

    if viewtype == 'montage':
    # 3D montage (lightbox) mode

        # set default rows and cols
        if rows is None:
            rows = int(np.ceil(np.sqrt(Nz*Nbatch)))
        if cols is None:
            cols = int(np.ceil(Nbatch*Nz/rows))

        # create the montage
        lightbox = np.zeros((rows*Nx,cols*Ny))
        for i in range(rows):
            for j in range(cols):
                if i*cols+j < Nbatch*Nz:
                    ibatch = int(np.floor((i*cols+j)/Nz))
                    iz = int((i*cols+j) % Nz)
                    lightbox[i*Nx:(i+1)*Nx,j*Ny:(j+1)*Ny] = img[ibatch,:,:,iz]
        
        # show the montage
        plt.imshow(lightbox, cmap=cmap)
        plt.axis('off')
        plt.show()

    elif viewtype == 'mid3':
    # 3D mid-slice mode

        if offset is None:
            offset = (0,0,0,0)

        # get the slices
        ibatch = offset[0]
        ix = int(np.floor(Nx/2)) + offset[1]
        iy = int(np.floor(Ny/2)) + offset[2]
        iz = int(np.floor(Nz/2)) + offset[3]

        # check the slices
        if ibatch < 0 or ibatch >= Nbatch:
            raise ValueError('batch index out of bounds')
        if ix < 0 or ix >= Nx:
            raise ValueError('x slice out of bounds')
        if iy < 0 or iy >= Ny:
            raise ValueError('y slice out of bounds')
        if iz < 0 or iz >= Nz:
            raise ValueError('z slice out of bounds')
        
        # show the slices
        lightbox = np.zeros((Ny+Nx,Nz+Ny))
        lightbox[0:Ny,0:Nz] = img[ibatch,ix,:,:]
        lightbox[Ny:,0:Nz] = img[ibatch,:,iy,:]
        lightbox[0:Nx,Nz:] = img[ibatch,:,:,iz]
        plt.imshow(lightbox, cmap=cmap)
        plt.axis('off')
        plt.show()

    elif viewtype == 'mip3':
    # 3D maximum intensity projection mode
        
        # show the projections
        lightbox = np.zeros((Ny+Nx,Nz+Ny))
        lightbox[0:Ny,0:Nz] = np.max(img, axis=1)
        lightbox[Ny:,0:Nz] = np.max(img, axis=2)
        lightbox[0:Nx,Nz:] = np.max(img, axis=3)
        plt.imshow(lightbox, cmap=cmap)
        plt.axis('off')
        plt.show()