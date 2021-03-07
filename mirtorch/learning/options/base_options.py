import argparse
import os
from util import util
import torch
import models
import data


class BaseOptions():
    def __init__(self):
        self.initialized = False

    def initialize(self, parser):
        parser.add_argument('--dataroot', required=True, help='path to images (should have subfolders trainA, trainB, valA, valB, etc)')
        parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
        parser.add_argument('--loadSize', type=int, default=286, help='scale images to this size')
        parser.add_argument('--fineSize', type=int, default=256, help='then crop to this size')
        parser.add_argument('--input_nc', type=int, default=2, help='# of input image channels')
        parser.add_argument('--output_nc', type=int, default=2, help='# of output image channels')
        parser.add_argument('--input_k_nc', type=int, default=2, help='# of input kspace channels')
        parser.add_argument('--output_k_nc', type=int, default=2, help='# of output kspace channels')
        parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in first conv layer')
        parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in first conv layer')
        parser.add_argument('--which_model_netD_I', type=str, default='basic', help='selects model to use for netD')
        parser.add_argument('--which_model_netD_K', type=str, default='basic', help='selects model to use for netD')
        parser.add_argument('--which_model_netG_I', type=str, default='resnet_9blocks', help='selects model to use for netG')
        parser.add_argument('--which_model_netG_K', type=str, default = 'resnet_9blocks', help = '#of k-space interpolation')
        parser.add_argument('--n_layers_D_K', type=int, default=2, help='only used if which_model_netD == n_layers')
        parser.add_argument('--n_layers_D_I', type=int, default=2, help='only used if which_model_netD == n_layers')
        parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        parser.add_argument('--name', type=str, default='experiment_name', help='name of the experiment. It decides where to store samples and models')
        parser.add_argument('--dataset_mode', type=str, default='unaligned', help='chooses how datasets are loaded. [unaligned | aligned | single]')
        parser.add_argument('--model', type=str, default='cycle_gan',
                            help='chooses which model to use. cycle_gan, pix2pix, test')
        parser.add_argument('--which_direction', type=str, default='AtoB', help='AtoB or BtoA')
        parser.add_argument('--nThreads', default=4, type=int, help='# threads for loading data')
        parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        parser.add_argument('--norm', type=str, default='instance', help='instance normalization or batch normalization')
        parser.add_argument('--serial_batches', action='store_true', help='if true, takes images in order to make batches, otherwise takes them randomly')
        parser.add_argument('--display_winsize', type=int, default=1024, help='display window size')
        parser.add_argument('--display_id', type=int, default=1, help='window id of the web display')
        parser.add_argument('--display_server', type=str, default="http://localhost", help='visdom server of the web display')
        parser.add_argument('--display_port', type=int, default=8097, help='visdom port of the web display')
        parser.add_argument('--no_dropout', action='store_true', help='no dropout for the generator')
        parser.add_argument('--max_dataset_size', type=int, default=float("inf"), help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')
        parser.add_argument('--resize_or_crop', type=str, default='resize_and_crop', help='scaling and cropping of images at load time [resize_and_crop|crop|scale_width|scale_width_and_crop]')
        parser.add_argument('--no_flip', action='store_true', help='if specified, do not flip the images for data augmentation')
        parser.add_argument('--init_type', type=str, default='normal', help='network initialization [normal|xavier|kaiming|orthogonal]')
        parser.add_argument('--init_gain', type=float, default=0.02, help='scaling factor for normal, xavier and orthogonal.')
        parser.add_argument('--verbose', action='store_true', help='if specified, print more debugging information')
        parser.add_argument('--suffix', default='', type=str, help='customized suffix: opt.name = opt.name + suffix: e.g., {model}_{which_model_netG}_size{loadSize}')
        parser.add_argument('--k_ngf', type=int, default=64, help='# of gen filters in first ksapce conv layer')
        parser.add_argument('--k_norm', type=str, default='instance', help='instance normalization or batch normalization')
        parser.add_argument('--k_no_dropout', action='store_true', help='no dropout for the generator')
        parser.add_argument('--k_init_type', type=str, default='normal', help='network initialization [normal|xavier|kaiming|orthogonal]')
        parser.add_argument('--k_init_gain', type=float, default=0.02, help='scaling factor for normal, xavier and orthogonal.')
        parser.add_argument('--ralpha',type=float, default=1, help='R-factor alpha')
        parser.add_argument('--rfactor',type=float, default=4, help='Rfactor')
        parser.add_argument('--mask_alert', action='store_true', help='show the details of mask generator')
        parser.add_argument('--loss_content_K_l1', type=float, default=0, help='content loss, l1')
        parser.add_argument('--loss_content_K_l2', type=float, default=0, help='content loss, l2')
        parser.add_argument('--loss_content_K_perceptual', type=float, default=0, help='content loss, perceptual')
        parser.add_argument('--loss_content_I_l1', type=float, default=0, help='content loss, l1')
        parser.add_argument('--loss_content_I_l2', type=float, default=0, help='content loss, l2')
        parser.add_argument('--loss_content_I_perceptual', type=float, default=0, help='content loss, perceptual')
        parser.add_argument('--loss_GAN_I', type=float, default=1, help='content loss, perceptual')
        parser.add_argument('--loss_GAN_K', type=float, default=1, help='content loss, perceptual')
        parser.add_argument('--loss_wgan_gp', type=float, default=10, help='wgan gradient penalty')
        parser.add_argument('--loss_ssim', type = float, default=0, help='The ratio of SSIM loss')
        parser.add_argument('--matname', type=str, default='k', help = 'the name of kspace in .mat')
        parser.add_argument('--mask_path', type=str, default='uniform', help='type of kspace mask')
        parser.add_argument('--mask_type', type=str, default='uniform', help='type of kspace mask: uniform random_alpha mat')
        parser.add_argument('--alpha', type=float, default=1.0, help='the ratio of original kspace')
        parser.add_argument('--beta', type=float, default=1, help = 'the ratio of density layer')
        parser.add_argument('--noise_level', type=float, default=0, help = 'the magnitude of noise')
        parser.add_argument('--tanh_alpha', type =float, default = 100, help = 'the stretch rate of activation function')
        parser.add_argument('--Trans', type = float, default = 1, help = 'the size of translation in simulation ')
        parser.add_argument('--angle', type = float, default = 0.5, help = 'the size of rotation in simulation')
        parser.add_argument('--num_blocks', type=int, default=10, help='number of blocks in unrolled networks')
        parser.add_argument('--num_FOE', type=int, default=24, help='number of convolutional kernels in VN')
        parser.add_argument('--num_rbf', type=int, default=24, help='number of RBF interpolation in VN')
        parser.add_argument('--MODLtol', type=float, default=0.000001, help='convergence tolerance in MODL')
        parser.add_argument('--MODLLambda', type=float, default=0.001, help='topletiz constraint in MODL')
        parser.add_argument('--MODLLambdaTV', type=float, default=1.5, help='topletiz constraint in MODL')
        parser.add_argument('--MODLtolTV', type=float, default=0.5, help='convergence tolerance in MODL')
        parser.add_argument('--datalabel', type=str, default='', help='certain label of file name to be included in the dataset')
        parser.add_argument('--nx', type=int, default=48, help='nx')
        parser.add_argument('--ny', type=int, default=48, help='ny')
        parser.add_argument('--datamode', type=str, default='magic', help='which dataset')
        parser.add_argument('--num_shots', type=int, default=5, help='how many shots')
        parser.add_argument('--num_acq', type=int, default=10000, help='how many acquisitions in the PE directions in total')
        parser.add_argument('--nfe', type=int, default=320, help='how many sampling points per echo')
        parser.add_argument('--decim_rate', type=int, default=4, help='how many folds to accelerate')
        parser.add_argument('--dt', type=float, default=0.000004, help='dwell time')
        parser.add_argument('--res', type=float, default=0, help='resolution in cm')
        parser.add_argument('--ReconVSTraj', type=int, default=10, help='how many updates of recon against sampling')
        parser.add_argument('--no_global_residual', action='store_true', help='add residual connection for unet/didn')
        parser.add_argument('--padding', type=int, default=40, help='how many padding for the b-spline kernels')
        parser.add_argument('--didn_blocks', type=int, default=3, help='')

        self.initialized = True
        return parser

    def gather_options(self):
        # initialize parser with basic options
        if not self.initialized:
            parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        # get the basic options
        opt, _ = parser.parse_known_args()

        # modify model-related parser options
        model_name = opt.model
        model_option_setter = models.get_option_setter(model_name)
        parser = model_option_setter(parser, self.isTrain)
        opt, _ = parser.parse_known_args()  # parse again with the new defaults

        # modify dataset-related parser options
        dataset_name = opt.dataset_mode
        dataset_option_setter = data.get_option_setter(dataset_name)
        parser = dataset_option_setter(parser, self.isTrain)

        self.parser = parser

        return parser.parse_args()

    def print_options(self, opt):
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

        # save to the disk
        expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
        util.mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')

    def parse(self):

        opt = self.gather_options()
        opt.isTrain = self.isTrain   # train or test

        # process opt.suffix
        if opt.suffix:
            suffix = ('_' + opt.suffix.format(**vars(opt))) if opt.suffix != '' else ''
            opt.name = opt.name + suffix

        self.print_options(opt)

        # set gpu ids
        str_ids = opt.gpu_ids.split(',')
        opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                opt.gpu_ids.append(id)
        if len(opt.gpu_ids) > 0:
            torch.cuda.set_device(opt.gpu_ids[0])

        self.opt = opt
        return self.opt
