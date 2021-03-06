import os.path
from .base_dataset import BaseDataset, get_transform
from .image_folder import make_dataset
from PIL import Image
import random
import scipy.io as sp
import numpy as np
import torch
from util.util import generate_mask_alpha, generate_mask_beta
import scipy.ndimage
from util.util import fft2, ifft2, cplx_to_tensor, complex_conj, complex_matmul, absolute
import h5py
'''
    Loader of multicoil NYU fastMRI dataset
'''
class nyumultidataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        # pay attention to the format of input directory
        self.dir_A = os.path.join(opt.dataroot, opt.phase)
        self.A_paths = make_dataset(self.dir_A, opt.datalabel)
        self.A_paths = sorted(self.A_paths)
        self.A_size = len(self.A_paths)
        self.ralpha = opt.ralpha
        self.rfactor = opt.rfactor
        self.mask_alert = opt.mask_alert
        self.mask_type = opt.mask_type
        self.nx = opt.nx
        self.ny = opt.ny
        if (opt.mask_type == 'mat'):
            self.mask = np.load(opt.mask_path)

    def __getitem__(self, index):

        if self.opt.datamode == 'DLMRI':
            A_path = self.A_paths[index % self.A_size]
            A_temp = h5py.File(A_path, 'r')
            s_r = A_temp['S'][()]['real']
            s_i = A_temp['S'][()]['imag']
            k_r = A_temp['y'][()]['real']
            k_i = A_temp['y'][()]['imag']
            DL_r = A_temp['IOut'][()]['real']
            DL_i = A_temp['IOut'][()]['imag']
            mask = A_temp['Q1'][()]
            ncoil, nx, ny = s_r.shape
            k_r = np.reshape(k_r, [ncoil, nx, ny],order='A')
            k_i = np.reshape(k_i, [ncoil, nx, ny],order='A')
            # print(A_path)
            # if nx < 368 or ny < 320:
            #     print(nx, ny)
            #     A_temp.close()
            #     os.remove(A_path)
            k_np = np.stack((k_r, k_i), axis=0)
            s_np = np.stack((s_r[:,nx//2-self.nx//2:nx//2+self.nx//2,ny//2-self.ny//2:ny//2+self.ny//2], s_i[:,nx//2-self.nx//2:nx//2+self.nx//2,ny//2-self.ny//2:ny//2+self.ny//2]), axis=0)
            DL = np.stack((DL_r[ nx // 2 - self.nx // 2:nx // 2 + self.nx // 2,
                             ny // 2 - self.ny // 2:ny // 2 + self.ny // 2],
                             DL_i[nx // 2 - self.nx // 2:nx // 2 + self.nx // 2,
                             ny // 2 - self.ny // 2:ny // 2 + self.ny // 2]), axis=0)
            I1 = A_temp['I1'][()]
            I1 = np.stack((I1['real'], I1['imag']))
            I1 = torch.tensor(I1[:, nx//2-self.nx//2:nx//2+self.nx//2,ny//2-self.ny//2:ny//2+self.ny//2], dtype=torch.float32)
            A_k = torch.tensor(k_np, dtype=torch.float32).permute(1,0,2,3)
            A_I = ifft2(A_k.permute(0,2,3,1)).permute(0,3,1,2)
            A_I = A_I[:,:,nx//2-self.nx//2:nx//2+self.nx//2,ny//2-self.ny//2:ny//2+self.ny//2]
            A_s = torch.tensor(s_np, dtype=torch.float32).permute(1,0,2,3)
            SOS = torch.sum(complex_matmul(A_I, complex_conj(A_s)),dim=0)
            decim = torch.max(torch.abs(SOS)[:])
            A_I = A_I/decim
            A_DL = torch.tensor(DL, dtype=torch.float32)
            A_DL = A_DL/torch.max(torch.abs(A_DL)[:])
            A_k = fft2(A_I.permute(0,2,3,1)).permute(0,3,1,2)
            maskt = torch.tensor(np.repeat(mask[np.newaxis, nx//2-self.nx//2:nx//2+self.nx//2, ny//2-self.ny//2:ny//2+self.ny//2], 2, axis=0), dtype=torch.float32)
            A_temp.close()
            return {'kreal': A_k, 'path': A_path, 'smap': A_s, 'DLRecon': A_DL, 'mask': maskt, 'I1':I1}
        elif self.opt.datamode == 'CSMRI':
            A_path = self.A_paths[index % self.A_size]
            A_temp = h5py.File(A_path, 'r')
            s_r = A_temp['s_r'][()]
            s_i = A_temp['s_i'][()]
            k_r = A_temp['k_r'][()]
            k_i = A_temp['k_i'][()]
            DL_r = A_temp['I_r'][()]
            DL_i = A_temp['I_i'][()]
            mask = A_temp['mask'][()]
            ncoil, nx, ny = s_r.shape
            k_r = np.reshape(k_r, [ncoil, nx, ny],order='A')
            k_i = np.reshape(k_i, [ncoil, nx, ny],order='A')
            k_np = np.stack((k_r, k_i), axis=0)
            s_np = np.stack((s_r[:,nx//2-self.nx//2:nx//2+self.nx//2,ny//2-self.ny//2:ny//2+self.ny//2], s_i[:,nx//2-self.nx//2:nx//2+self.nx//2,ny//2-self.ny//2:ny//2+self.ny//2]), axis=0)
            DL = np.stack((DL_r[ nx // 2 - self.nx // 2:nx // 2 + self.nx // 2,
                             ny // 2 - self.ny // 2:ny // 2 + self.ny // 2],
                             DL_i[nx // 2 - self.nx // 2:nx // 2 + self.nx // 2,
                             ny // 2 - self.ny // 2:ny // 2 + self.ny // 2]), axis=0)
            A_k = torch.tensor(k_np, dtype=torch.float32).permute(1,0,2,3)
            A_I = ifft2(A_k.permute(0,2,3,1)).permute(0,3,1,2)
            A_I = A_I[:,:,nx//2-self.nx//2:nx//2+self.nx//2,ny//2-self.ny//2:ny//2+self.ny//2]
            A_s = torch.tensor(s_np, dtype=torch.float32).permute(1,0,2,3)
            SOS = torch.sum(complex_matmul(A_I, complex_conj(A_s)),dim=0)
            decim = torch.max(torch.abs(SOS)[:])
            A_I = A_I/decim
            A_DL = torch.tensor(DL, dtype=torch.float32)
            A_DL = A_DL/torch.max(torch.abs(A_DL)[:])
            A_k = fft2(A_I.permute(0,2,3,1)).permute(0,3,1,2)
            maskt = torch.tensor(np.repeat(mask[np.newaxis, nx//2-self.nx//2:nx//2+self.nx//2, ny//2-self.ny//2:ny//2+self.ny//2], 2, axis=0), dtype=torch.float32)
            A_temp.close()
            return {'kreal': A_k, 'path': A_path, 'smap': A_s, 'DLRecon': A_DL, 'mask': maskt}
        else:
            A_path = self.A_paths[index % self.A_size]
            A_temp = np.load(A_path)
            s_r = A_temp['s_r'] / 32767.0
            s_i = A_temp['s_i'] / 32767.0
            k_r = A_temp['k_r'] / 32767.0
            k_i = A_temp['k_i'] / 32767.0
            ncoil, nx, ny = s_r.shape
            k_np = np.stack((k_r, k_i), axis=0)
            s_np = np.stack((s_r[:,nx//2-160:nx//2+160,ny//2-160:ny//2+160], s_i[:,nx//2-160:nx//2+160,ny//2-160:ny//2+160]), axis=0)

            A_k = torch.tensor(k_np, dtype=torch.float32).permute(1,0,2,3)
            A_I = ifft2(A_k.permute(0,2,3,1)).permute(0,3,1,2)
            A_I = A_I[:,:,nx//2-160:nx//2+160,ny//2-160:ny//2+160]
            A_s = torch.tensor(s_np, dtype=torch.float32).permute(1,0,2,3)
            SOS = torch.sum(complex_matmul(A_I, complex_conj(A_s)),dim=0)
            # print('sossize',SOS.size())
            A_I = A_I/torch.max(torch.abs(SOS)[:])
            A_k = fft2(A_I.permute(0,2,3,1)).permute(0,3,1,2)

            if (self.mask_type == 'random_alpha'):
                mask, r_factor = generate_mask_alpha(size=[320, 320], r_alpha=self.ralpha, seed=-1,
                                                     r_factor_designed=self.rfactor, mute=True, acs=8)
            elif (self.mask_type == 'uniform'):
                mask, r_factor = generate_mask_beta(size=[320, 320], r_factor_designed=self.rfactor, mute=True)
            else:
                mask = self.mask
            mask = np.fft.fftshift(mask)
            maskt = torch.tensor(np.repeat(mask[np.newaxis, :, :], 2, axis=0), dtype=torch.float32)
            if self.opt.isTrain and self.opt.rot:
                k = np.random.randint(3)
                A_k = torch.rot90(A_k, k, [2, 3])
                A_s = torch.rot90(A_s, k, [2, 3])
                if np.random.rand() > 0.5:
                    A_k = torch.flip(A_k, [2])
                    A_s = torch.flip(A_s, [2])
                if np.random.rand() > 0.5:
                    A_k = torch.flip(A_k, [3])
                    A_s = torch.flip(A_s, [3])
            return {'kreal': A_k, 'path': A_path, 'smap': A_s, 'mask': maskt}

    def __len__(self):
        return self.A_size

    def name(self):
        return 'nyumultiDataset'

#  generate mask based on alpha
