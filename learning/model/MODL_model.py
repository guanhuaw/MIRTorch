import torch
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
from util.metrics import PSNR, roll_2
import pytorch_msssim
import time
from util.util import fft2, ifft2
import sys
import os


class MODLModel(BaseModel):
    def name(self):
        return 'MODLModel'

    # Initialize the model
    def initialize(self, opt):
        self.stocha_size = opt.stocha_size
        self.num_shots = opt.num_shots
        BaseModel.initialize(self, opt)  # ATTENTION HERE: NEED TO ALTER THE DEFAULT PLAN
        self.netG_I = networks.define_G(opt, opt.input_nc, opt.output_nc,
                                        opt.ngf, opt.which_model_netG_I, opt.norm, not opt.no_dropout, opt.init_type,
                                        opt.init_gain, self.gpu_ids)
        self.netDensity = networks.define_G(opt, opt.input_nc, opt.output_nc,
                                            32, 'density', opt.norm, not opt.no_dropout, opt.init_type,
                                            opt.init_gain, self.gpu_ids)
        self.netSampling = networks.define_G(opt, opt.input_nc, opt.output_nc,
                                             32, 'sampling3D', opt.norm, not opt.no_dropout, opt.init_type,
                                             opt.init_gain, self.gpu_ids)

        if self.isTrain:
            if self.train_phase == 'generator':
                self.model_names = ['G_I', 'Density', 'Sampling']
                self.loss_names = ['G_I_L1', 'G_I_L2', 'SSIM', 'PSNR', 'density', 'acc_ratio']
            else:
                self.model_names = ['G_I', 'D_I', 'Density', 'Sampling']
                self.loss_names = ['G_GAN_I', 'G_I_L1', 'G_I_L2', 'D_GAN_I', 'SSIM', 'PSNR', 'density', 'acc_ratio']
        else:  # during test time, only load Gs
            self.model_names = ['G_I', 'Density', 'Sampling']
            self.loss_names = ['SSIM', 'PSNR', 'acc_ratio']
        if self.train_phase == 'generator':
            self.visual_names = ['Ireal', 'Ifake', 'Preal', 'mask', 'density', 'kreal', 'ksimu', 'Iunder', 'ICG',
                                 'k_simu_1']
        else:
            self.visual_names = ['Ireal', 'Ifake', 'Preal', 'mask', 'density', 'kreal', 'ksimu', 'Iunder', 'ICG',
                                 'k_simu_1']

        self.criterionL1 = torch.nn.L1Loss()
        self.criterionMSE = torch.nn.MSELoss()
        self.ssim_loss = pytorch_msssim.SSIM(val_range=2)
        if self.isTrain:
            self.optimizers = []
            self.optimizer_G = torch.optim.Adam(list(
                list(self.netG_I.parameters()) + list(self.netDensity.parameters()) + list(
                    self.netSampling.parameters())),
                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)

        if self.isTrain and self.train_phase == 'together':
            self.no_wgan = opt.no_wgan
            self.no_wgan_gp = opt.no_wgan_gp
            if self.no_wgan_gp == False:
                self.disc_step = opt.disc_step
            else:
                self.disc_step = 1
            self.disc_model = opt.disc_model
            use_sigmoid = opt.no_lsgan
            if opt.disc_model == 'pix2pix':
                self.netD_I = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf,
                                                opt.which_model_netD_I,
                                                opt.n_layers_D_I, opt.norm, use_sigmoid, opt.init_type,
                                                opt.init_gain,
                                                self.gpu_ids)
            if opt.disc_model == 'traditional':
                self.netD_I = networks.define_D(opt.output_nc, opt.ndf, opt.which_model_netD_I,
                                                opt.n_layers_D_I, opt.norm, use_sigmoid, opt.init_type,
                                                opt.init_gain,
                                                self.gpu_ids)
            self.loss_wgan_gp = opt.loss_wgan_gp
            self.fake_I_pool = ImagePool(opt.pool_size)
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, use_l1=not opt.no_l1gan).to(self.device)
            self.optimizer_D_I = torch.optim.Adam(self.netD_I.parameters(),
                                                  lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_D_I)

    def set_input(self, input):
        self.kreal = input['kreal'].to(self.device).repeat(self.stocha_size, 1, 1, 1, 1)
        self.smap = input['smap'].to(self.device).repeat(self.stocha_size, 1, 1, 1, 1)
        self.AT = networks.OPAT(self.smap)
        self.A = networks.OPA(self.smap)
        self.Ireal = self.AT(self.kreal, torch.ones_like(self.kreal)[:, 0, :, :, :])
        # print('Irealsize', self.Ireal.size())
        self.Preal = torch.atan(self.Ireal[:, 1, :, :] / self.Ireal[:, 0, :, :]).unsqueeze(1)
        self.single_idx_all = input['single']
        self.ovlp_idx_all = input['ovlp']
        self.path = input['path']

    def backward_D(self):
        if self.disc_model == 'pix2pix':
            fake_AB_I = self.fake_k_pool.query(self.Ifake)
            pred_real_I = self.netD_I(torch.cat((self.Ireal, self.Iunder), 1))
            pred_fake_I = self.netD_I(fake_AB_I.detach())
        if self.disc_model == 'traditional':
            fake_AB_I = self.fake_I_pool.query(self.Ifake)
            pred_real_I = self.netD_I(self.Ireal)
            pred_fake_I = self.netD_I(fake_AB_I.detach())
        if self.no_wgan == False:
            self.loss_D_GAN_fake_I = pred_fake_I.mean()
            self.loss_D_GAN_real_I = -pred_real_I.mean()
        elif self.no_wgan_gp == False:
            self.loss_D_GAN_fake_I = pred_fake_I.mean()
            self.loss_D_GAN_real_I = -pred_real_I.mean()
            alpha = torch.rand(self.Ireal.size(0), 1, 1, 1).to(self.device)
            x_hat = (alpha * self.Ireal.data + (1 - alpha) * self.Ifake.data).requires_grad_(True)
            out_src = self.netD_I(x_hat)
            self.d_loss_gp_I = self.gradient_penalty(out_src, x_hat) * self.loss_wgan_gp
        else:
            self.loss_D_GAN_fake_I = self.criterionGAN(pred_fake_I, False)
            self.loss_D_GAN_real_I = self.criterionGAN(pred_real_I, True)
        # print('dfake',self.loss_D_GAN_fake_I)
        # print('dreal',self.loss_D_GAN_real_I)
        self.loss_D_GAN_I = 0.5 * (self.loss_D_GAN_fake_I + self.loss_D_GAN_real_I) * self.opt.loss_GAN_I
        self.loss_D_GAN = self.loss_D_GAN_I
        if self.no_wgan_gp == False:
            self.loss_D_GAN = self.loss_D_GAN + self.d_loss_gp_I

        self.loss_D_GAN.backward()

    def backward_G(self):
        # First, G(A) should fake the discriminator
        if self.isTrain and self.train_phase == 'together':
            if self.disc_model == 'pix2pix':
                fake_AB_I = torch.cat((self.Ifake, self.Iunder), 1)
                pred_fake_I = self.netD_I(fake_AB_I)
            if self.disc_model == 'traditional':
                pred_fake_I = self.netD_I(self.Ifake)
            if self.no_wgan == False:
                self.loss_G_GAN_I = -pred_fake_I.mean()
            else:
                self.loss_G_GAN_I = self.criterionGAN(pred_fake_I, True)
        else:
            self.loss_G_GAN_I = 0
        # print('greal',self.loss_G_GAN_I)
        self.loss_G_GAN_I = self.loss_G_GAN_I * self.opt.loss_GAN_I
        self.loss_G_GAN = self.loss_G_GAN_I
        self.loss_G_I_L1 = self.criterionL1(self.Ifake, self.Ireal) * self.opt.loss_content_I_l1
        self.k_simu_1 = fft2(torch.abs(self.Ifake - self.Ireal).permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        self.loss_G_I_L2 = self.criterionMSE(self.Ifake, self.Ireal) * self.opt.loss_content_I_l2
        self.loss_density = self.criterionMSE(self.density, torch.zeros_like(self.density)) * self.opt.beta
        self.loss_G_CON_I = self.loss_G_I_L1 + self.loss_G_I_L2
        # print('con',self.loss_G_CON_I)
        # print('maxI', torch.max(self.Ireal))
        # if torch.isnan(self.loss_G_CON_I).data:
        #     sys.exit()
        # if (self.loss_G_I_L1>5).data:
        #     print(self.path)
        #     if 'Layer15' in self.path[0] or 'Layer1.npz' in self.path[0]:
        #         os.remove(self.path[0])
        #     else:
        #         sys.exit()
        # if (self.loss_G_I_L1>8).data:
        #     os.remove(self.path[0])
        #     print(self.path)
        #     # sys.exit()
        self.loss_G = self.loss_G_CON_I + self.loss_G_GAN - self.loss_SSIM * self.opt.loss_ssim + self.loss_density
        self.loss_G.backward()

    def forward(self):
        self.density = self.netDensity(torch.ones(1, 1, self.opt.nx, self.opt.ny))
        # print('maxdensity', torch.max(self.density),'mindensity', torch.min(self.density))
        self.ksimu, self.mask = self.netSampling(self.density, self.kreal, self.single_idx_all.squeeze(0),
                                                 self.ovlp_idx_all.squeeze(0))
        self.mask = (self.mask > 0.5).to(dtype=self.mask.dtype)
        # self.k_simu_1 = self.ksimu-self.kreal*self.mask.repeat(self.smap.size(1), 1, 1, 1, 1).permute(1, 0, 2, 3, 4)
        self.Iunder = self.AT(self.ksimu, self.mask)
        # self.Iunder1 = self.AT(self.ksimu, torch.ones_like(self.mask))
        # self.Iunder2 = self.AT(self.kreal, self.mask)
        Ifake1 = torch.clone(self.Iunder)
        CG = networks.CG.apply
        self.loss_acc_ratio = (torch.sum(self.mask > -1).double()) / torch.sum(self.mask > 0.5)
        for ii in range(self.opt.num_blocks):
            Ifake = self.netG_I(Ifake1)
            Ifake1 = CG(Ifake, self.opt.MODLtol, self.opt.MODLLambda, self.smap, self.mask, self.Iunder)
            if ii == 0:
                self.ICG = Ifake
            # Ifake1 = self.netG_I(Ifake1)
            # self.ICG = Ifake1

        self.Ifake = Ifake1
        self.Pfake = torch.atan(self.Ifake[:, 1, :, :] / self.Ifake[:, 0, :, :]).unsqueeze(1)
        self.loss_PSNR = PSNR(self.Ireal, self.Ifake)
        self.loss_SSIM = self.ssim_loss(self.Ireal, self.Ifake)

    def optimize_parameters(self):
        if self.isTrain and self.train_phase == 'together':
            self.forward()
            self.set_requires_grad(self.netD_I, True)
            for iter_d in range(self.disc_step):
                self.optimizer_D_I.zero_grad()
                self.backward_D()
                self.optimizer_D_I.step()
            self.set_requires_grad(self.netD_I, False)
            self.optimizer_G.zero_grad()
            self.backward_G()
            self.optimizer_G.step()
        else:
            self.forward()
            self.optimizer_G.zero_grad()
            self.backward_G()
            self.optimizer_G.step()
