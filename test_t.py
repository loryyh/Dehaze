#!/usr/bin/python3

import argparse
import functools
import sys
import os

import numpy as np
import torchvision.transforms as transforms
from torchvision.transforms import ToPILImage
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch

from datasets import ImageDataset
from models import create_model
from options.base_options import BaseOptions

class TestOptions(BaseOptions):
    """This class includes test options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)  # define shared options

        parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        # Dropout and Batchnorm has different behavioir during training and test.
        parser.add_argument('--eval', action='store_true', help='use eval mode during test time.')
        parser.add_argument('--num_test', type=int, default=500, help='how many test images to run')

        parser.add_argument('--batchSize', type=int, default=1, help='size of the batches')
        parser.add_argument('--dataroot_t', type=str, default='D:/PyCharm/PyProjects/Cycle-SNSPGAN-tttest/datasets/hazy2clear', help='root directory of the dataset')
        parser.add_argument('--sizew', type=int, default=512, help='size of the data (squared assumed)') #512
        parser.add_argument('--sizeh', type=int, default=320, help='size of the data (squared assumed)') #320
        parser.add_argument('--cuda', action='store_true', help='use GPU computation')
        parser.add_argument('--n_cpu', type=int, default=0, help='number of cpu threads to use during batch generation')
        parser.add_argument('--generator_A2B', type=str, default='output/netG_A.pth', help='A2B generator checkpoint file')
        parser.add_argument('--generator_B2A', type=str, default='output/netG_B.pth', help='B2A generator checkpoint file')
        self.isTrain = False
        return parser

opt = TestOptions().parse()
print(opt)

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")


model = create_model(opt)      # create a model given opt.model and other options
model.setup(opt)
netG_A2B = model.netG

if opt.cuda:
    netG_A2B.cuda()


checkpoint = torch.load('checkpoints/dehaze/latest_net_G.pth')
netG_A2B.load_state_dict(checkpoint)  #['model']
netG_A2B.eval()


Tensor = torch.cuda.FloatTensor if opt.cuda else torch.Tensor
input_A = Tensor(opt.batchSize, opt.input_nc, opt.sizeh, opt.sizew)

transforms_ = [ transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) ]
dataloader = DataLoader(ImageDataset(opt.dataroot_t, transforms_=transforms_, mode='test'),
                        batch_size=1, shuffle=False, num_workers=opt.n_cpu)

if not os.path.exists('output/B'):
    os.makedirs('output/B')

for i, batch in enumerate(dataloader):
    npy_img = np.array(batch['A'])
    resize_transform = transforms.Resize((opt.sizeh, opt.sizew))
    #resize_transform = transforms.Resize((256, 256))
    real_A = Variable(input_A.copy_(resize_transform(batch['A'])))
    fake_B = 0.5 * (netG_A2B(real_A).data + 1.0)
    resize_transform = transforms.Resize((npy_img.shape[2], npy_img.shape[3]))
    fake_B = resize_transform(fake_B)
    save_image(fake_B, 'output/B/%d.png' % (i + 1))

    sys.stdout.write('\rGenerated images %d of %d' % (i+1, len(dataloader)))

sys.stdout.write('\n')
sys.stdout.write('\n')
###################################