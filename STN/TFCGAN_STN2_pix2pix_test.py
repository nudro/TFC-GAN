import argparse
import os
import numpy as np
import math
import itertools
import time
import datetime
import sys
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch
import cv2
from torch.distributed import Backend
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from datasets_stn import *

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0, help="epoch to load")
parser.add_argument("--dataset_name", type=str, default="eurecom_warped_pairs", help="name of the dataset")
parser.add_argument("--img_height", type=int, default=256, help="size of image height")
parser.add_argument("--img_width", type=int, default=256, help="size of image width")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--experiment", type=str, default="tfcgan_stn", help="experiment name")
opt = parser.parse_args()


os.makedirs("/home/local/AD/cordun1/experiments/TFC-GAN/images/test_results/%s" % opt.experiment, exist_ok=True)
cuda = True if torch.cuda.is_available() else False

"""


Test Script


"""

    
#################
# STN
#################
                          
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.localization = nn.Sequential(
            nn.Conv2d(opt.channels*2, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )
        
        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(10*60*60, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )

        self.fc_loc[2].weight.data.normal_(mean=0.0, std=5e-4)
        self.fc_loc[2].bias.data.zero_()
        
    # Spatial transformer network forward function
    def stn_phi(self, x):
        xs = self.localization(x) # convolves the cat channel=6
        xs = xs.view(-1, 10*60*60) 
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)
        return theta
    
    def get_grid(self, theta, src):
        grid = F.affine_grid(theta, src.size())
        return grid
                    
    def stn_resample(self, src, rs_grid):
        Rs = F.grid_sample(src, grid)
        return Rs

    def forward(self, img_A, img_B, src):
        img_input = torch.cat((img_A, img_B), 1)
        dtheta = self.stn_phi(img_input) # deformation field for real A and real B
        identity_theta = torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float).cuda()

        dtheta = dtheta.reshape(img_B.size(0), 2*3)
        theta = dtheta + identity_theta.unsqueeze(0).repeat(img_B.size(0), 1)

        theta_batches = []
        for t in theta:
            this_theta = (t.view(-1, 2, 3)).reshape(1,2,3)
            theta_batches.append(this_theta)

        images_B = []
        for img in src[0]: #real_B only at test time
            this_img_B = img.reshape(1, img.size(0), img.size(1), img.size(2)) # affine_grid only takes 4D - was [3,256,256]-> make it [1,3,256,256]
            images_B.append(this_img_B)


        images_fA = []
        for img in src[1]: #fake_A
            this_img_fA = img.reshape(1, img.size(0), img.size(1), img.size(2)) # affine_grid only takes 4D - was [3,256,256]-> make it [1,3,256,256]
            images_fA.append(this_img_fA)


        warped_B = []
        for i in range(len(images_B)): #1:1 match with theta, matching with theta is important
            rs_grid_B = F.affine_grid(theta_batches[i], images_B[i].size())
            Rs_B = F.grid_sample(images_B[i], rs_grid_B,  mode='bilinear', padding_mode='zeros', align_corners=False)
            warped_B.append(Rs_B)


        warped_fA = []
        for i in range(len(images_fA)): #1:1 match with theta, matching with theta is important
            rs_grid_fA = F.affine_grid(theta_batches[i], images_fA[i].size())
            Rs_fA = F.grid_sample(images_fA[i], rs_grid_fA,  mode='bilinear', padding_mode='zeros', align_corners=False)
            warped_fA.append(Rs_fA)


        reg_term = torch.mean(torch.abs(dtheta))
        
        return warped_B, warped_fA, reg_term

                    
##############################
#           U-NET
##############################

class UNetDown(nn.Module):
    def __init__(self, in_size, out_size, normalize=True, dropout=0.0):
        super(UNetDown, self).__init__()
        layers = [nn.Conv2d(in_size, out_size, 4, 2, 1, bias=False)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_size))
        layers.append(nn.LeakyReLU(0.2))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        #print("UNetDown x.shape", x.size())
        # print("UNetDown x is:", x)
        return self.model(x)


class UNetUp(nn.Module):
    def __init__(self, in_size, out_size, dropout=0.0):
        super(UNetUp, self).__init__()
        layers = [
            nn.ConvTranspose2d(in_size, out_size, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(out_size),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        x = self.model(x)
        x = torch.cat((x, skip_input), 1)
        return x


class GeneratorUNet(nn.Module):
    def __init__(self, img_shape):
        super(GeneratorUNet, self).__init__()
        channels, self.h, self.w = img_shape

        self.down1 = UNetDown(channels, 64, normalize=False)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 512, dropout=0.5)
        self.down5 = UNetDown(512, 512, normalize = False, dropout=0.5)
        self.down6 = UNetDown(512, 512, dropout=0.5)

        self.up1 = UNetUp(512, 512, dropout=0.5)
        self.up2 = UNetUp(1024, 512, dropout=0.5)        
        self.up3 = UNetUp(1024, 256)
        self.up4 = UNetUp(512, 128)
        self.up5 = UNetUp(256, 64)

        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(128, channels, 4, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        u1 = self.up1(d6, d5)
        u2 = self.up2(u1, d4)
        u3 = self.up3(u2, d3)
        u4 = self.up4(u3, d2)
        u5 = self.up5(u4, d1)
        out = self.final(u5)
        return out




##############################
# Transforms and Dataloaders
##############################

#Resizing happens in the ImageDataset() to 256 x 256 so that I can get patches
transforms_ = [
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]

test_dataloader = DataLoader(
    TestImageDataset(root = "/home/local/AD/cordun1/experiments/data/%s" % opt.dataset_name,
        transforms_=transforms_,
        mode="test"),
    batch_size=1,
    shuffle=True,
    num_workers=1,
)

    
##########
# UTILS
###########

# Tensor type - only use HalfTensor in this AMP script
HalfTensor = torch.cuda.HalfTensor if cuda else torch.HalfTensor
Tensor = torch.cuda.FloatTensor 


def load_clean_state(model_name, checkpoint_path):
    from collections import OrderedDict
    state_dict = torch.load(checkpoint_path)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v
        
    # load params
    model_name.load_state_dict(new_state_dict)
    print("Loaded successfully {} state dict".format(model_name))
    

input_shape_global = (opt.channels, opt.img_height, opt.img_width)
generator = GeneratorUNet(input_shape_global)
generator = generator.cuda()
model = Net().cuda()

g_path = "/home/local/AD/cordun1/experiments/TFC-GAN/saved_models/%s/generator_%d.pth" % (opt.experiment, opt.epoch)
load_clean_state(generator, g_path)

stn_path = "/home/local/AD/cordun1/experiments/TFC-GAN/saved_models/%s/stn_%d.pth" % (opt.experiment, opt.epoch)
load_clean_state(model, stn_path)

    
##############################
#       Test
##############################
    
for i, batch in tqdm(enumerate(test_dataloader)):
    real_A = Variable(batch["A"]).type(Tensor)
    real_B = Variable(batch["B"]).type(Tensor)
    
    # At test time, only the STN is used 
    with torch.no_grad(): 
        fake_B = generator(real_A)
        warped_B, warped_fB, reg_term = model(img_A=real_A, img_B=real_B, src=[real_B, fake_B]) 
        
        reg_B = torch.cat(warped_B) # turn from list into tensor
        
        fake_RT_B = generator(reg_B) # translate what was registered

        fake_TR_B = torch.cat(warped_fB) # registered what was translated - which was already done 

            
    
    # GLOBAL
    img_sample_global = torch.cat((real_A.data, reg_B.data,  fake_RT_B.data, fake_TR_B.data, real_B.data), -2)
    save_image(img_sample_global, "/home/local/AD/cordun1/experiments/TFC-GAN/images/test_results/%s/%s_g.png" % (opt.experiment, i), nrow=5, normalize=True)
    
