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
from datasets_temp import *  # temperature
import torch.nn as nn
import torch.nn.functional as F
import torch
from lpips_pytorch import LPIPS, lpips
import cv2
from torch.distributed import Backend
import antialiased_cnns
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from datasets_temp_sampling import *
from torch.utils.data.sampler import RandomSampler

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--img_height", type=int, default=256, help="size of image height")
parser.add_argument("--img_width", type=int, default=256, help="size of image width")
parser.add_argument("--patch_height", type=int, default=128, help="size of patch height")
parser.add_argument("--patch_width", type=int, default=128, help="size of patch width")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--experiment", type=str, default="none", help="experiment name")
opt = parser.parse_args()

os.makedirs("/home/local/AD/cordun1/experiments/faPVTgan/images/test_results/%s" % opt.experiment, exist_ok=True)
cuda = True if torch.cuda.is_available() else False

##############################
#           U-NET
##############################

class UNetDown(nn.Module):
    def __init__(self, in_size, out_size, normalize=True, dropout=0.0):
        super(UNetDown, self).__init__()
        layers = [nn.Conv2d(in_size, out_size, 4, 1, 1, bias=False)] # originally stride = 2
        if normalize:
            layers.append(nn.InstanceNorm2d(out_size))
        layers.append(nn.LeakyReLU(0.2))
        layers.append(antialiased_cnns.BlurPool(out_size, stride=2)) #downsample stride2
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UNetUp(nn.Module):
    def __init__(self, in_size, out_size, dropout=0.0):
        super(UNetUp, self).__init__()
        layers = [
                nn.ConvTranspose2d(in_size, out_size, 4, 2, 1, bias=False),
                antialiased_cnns.BlurPool(out_size, stride=1), #upsampling with stride=1 is just blurring no pooling
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
        self.down3 = UNetDown(128, 256, dropout=0.5)
        self.down4 = UNetDown(256, 512, dropout=0.5)
        self.down5 = UNetDown(512, 512, normalize = False)
        self.down6 = UNetDown(512, 512)

        self.up1 = UNetUp(512, 512)
        self.up2 = UNetUp(1024, 512, dropout=0.5)
        self.up3 = UNetUp(1024, 256, dropout=0.5)
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
        output = self.final(u5).type(HalfTensor)
        return output

##########
# UTILS
###########

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
    

##############################
#       Initialize
##############################
input_shape_patch = (opt.channels, opt.patch_height, opt.patch_width)
generator = GeneratorUNet(input_shape_patch)
generator = generator.cuda()

g_path = "/home/local/AD/cordun1/experiments/faPVTgan/saved_models/%s/generator_%d.pth" % (opt.experiment, opt.epoch)
load_clean_state(generator, g_path)


##############################
# Transforms and Dataloaders
##############################

transforms_ = [

    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]

#====TEST DATALOADER====

test_dataloader = DataLoader(
    TestDataset(root = "/home/local/AD/cordun1/experiments/data/EA_updated_test_set",
        transforms_=transforms_,
        mode="test"),
    batch_size=1, 
    shuffle=False,
    num_workers=8,
)



##############################
#       Testing
##############################

for i, batch in tqdm(enumerate(test_dataloader)):
    real_A = Variable(batch["A"].type(Tensor))
    real_B = Variable(batch["B"].type(Tensor))
    
    A1 = real_A[:, :, 0:0+opt.img_width//2, 0:0+opt.img_height//2] #(x,y) = (0,0)
    A2 = real_A[:, :, 0:0+opt.img_width//2, 128:128+opt.img_height//2] #(x,y) = (0, 128)
    A3 = real_A[:, :, 128:128+opt.img_width//2, 0:0+opt.img_height//2] #(x,y)=(128,0)
    A4 = real_A[:, :, 128:128+opt.img_width//2, 128:128+opt.img_height//2] #(x,y) = (128,128)
    
    B1 = real_B[:, :, 0:0+opt.img_width//2, 0:0+opt.img_height//2] #(x,y) = (0,0)
    B2 = real_B[:, :, 0:0+opt.img_width//2, 128:128+opt.img_height//2] #(x,y) = (0, 128)
    B3 = real_B[:, :, 128:128+opt.img_width//2, 0:0+opt.img_height//2] #(x,y)=(128,0)
    B4 = real_B[:, :, 128:128+opt.img_width//2, 128:128+opt.img_height//2] #(x,y) = (128,128)
    
    with torch.no_grad(): 
        fake_B = generator(real_A)
        
    # B patches
    fake_B1 = fake_B[:, :, 0:0+opt.img_width//2, 0:0+opt.img_height//2] #(x,y) = (0,0)
    fake_B2 = fake_B[:, :, 0:0+opt.img_width//2, 128:128+opt.img_height//2] #(x,y) = (0, 128)
    fake_B3 = fake_B[:, :, 128:128+opt.img_width//2, 0:0+opt.img_height//2] #(x,y)=(128,0)
    fake_B4 = fake_B[:, :, 128:128+opt.img_width//2, 128:128+opt.img_height//2] #(x,y) = (128,128)
    
    # SAVE PATCHES
    img_sample1 = torch.cat((A1.data, fake_B1.data, B1.data), -2)
    img_sample2 = torch.cat((A2.data, fake_B2.data, B2.data), -2)
    img_sample3 = torch.cat((A3.data, fake_B3.data, B3.data), -2)
    img_sample4 = torch.cat((A4.data, fake_B4.data, B4.data), -2)
    img_sample_patch = torch.cat((img_sample1.data, img_sample2.data, img_sample3.data, img_sample4.data), -2)
    save_image(img_sample_patch, "/home/local/AD/cordun1/experiments/faPVTgan/images/test_results/%s/%s_p.png" % (opt.experiment, i), nrow=5, normalize=True)

    # GLOBAL
    img_sample_global = torch.cat((real_A.data, fake_B.data, real_B.data), -2)
    save_image(img_sample_global, "/home/local/AD/cordun1/experiments/faPVTgan/images/test_results/%s/%s_g.png" % (opt.experiment, i), nrow=5, normalize=True)