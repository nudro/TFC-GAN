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
#from torch.nn.parallel.distributed import DistributedDataParallel
import antialiased_cnns
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from datasets_temp import *

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--dataset_name", type=str, default="facades", help="name of the dataset")
parser.add_argument("--img_height", type=int, default=256, help="size of image height")
parser.add_argument("--img_width", type=int, default=256, help="size of image width")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--experiment", type=str, default="none", help="experiment name")
opt = parser.parse_args()

os.makedirs("/home/local/AD/cordun1/experiments/TFC-GAN/images/test_results/%s" % opt.experiment, exist_ok=True)
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
        output = self.final(u5)
        return output

##########
# UTILS
###########

def make_16_patches(B):
    # B is real_B or fake_B
    
    patch_w = opt.img_width//4
    patch_h = opt.img_height//4
    
    B1 = B[:, :, 0:0+patch_w, 0:0+patch_h] #(x,y) = (0,0)
    B2 = B[:, :, 0:0+patch_w, 64:64+patch_h] #(x,y) = (0, 64)
    B3 = B[:, :, 0:0+patch_w, 128:128+patch_h] #(x,y)=(0,128)
    B4 = B[:, :, 0:0+patch_w, 192:192+patch_h] #(x,y) = (0,192)

    B5 = B[:, :, 64:64+patch_w, 0:0+patch_h] #(64,0)
    B6 = B[:, :, 64:64+patch_w, 64:64+patch_h] #(64, 64)
    B7 = B[:, :, 64:64+patch_w, 128:128+patch_h] #(64, 128)
    B8 = B[:, :, 64:64+patch_w, 192:192+patch_h] #(64, 192)

    B9 = B[:, :, 128:128+patch_w, 0:0+patch_h] #(128,0)
    B10 = B[:, :, 128:128+patch_w, 64:64+patch_h] #(128,64)
    B11 = B[:, :, 128:128+patch_w, 128:128+patch_h] #(128,128)
    B12 = B[:, :, 128:128+patch_w, 192:192+patch_h] #(128,192)

    B13 = B[:, :, 192:192+patch_w, 0:0+patch_h] #(192,0)
    B14 = B[:, :, 192:192+patch_w, 64:64+patch_h] #(192,64)
    B15 = B[:, :, 192:192+patch_w, 128:128+patch_h] #(192,128)
    B16 = B[:, :, 192:192+patch_w, 192:192+patch_h] #(192,192)
           
    return B1, B2, B3, B4, B5, B6, B7, B8, B9, B10, B11, B12, B13, B14, B15, B16

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
    
    
##############################
#       Initialize
##############################
input_shape_global = (opt.channels, opt.img_height, opt.img_width)
generator = GeneratorUNet(input_shape_global)
generator = generator.cuda()

g_path = "/home/local/AD/cordun1/experiments/TFC-GAN/saved_models/%s/generator_%d.pth" % (opt.experiment, opt.epoch)
load_clean_state(generator, g_path)


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
    shuffle=False,
    num_workers=1,
)

##############################
#       Testing
##############################

# Try AMP
#scaler = GradScaler()

"""
Note: That when you use AMP, although you set the outputs as FP-16 in the forward() to make the training go faster,
the weights are actually stored as master weights as FP-32. So at test time, you actually give it the full dataset
of FP-32, not FP-16. During training, the weights are copied over as FP-16. 

https://docs.fast.ai/callback.fp16.html#The-solution:-mixed-precision-training

At test time you use the regular FP32, not Half Tensor. That's only used during training. 

"""

for i, batch in tqdm(enumerate(test_dataloader)):
    real_A = Variable(batch["A"]).type(Tensor)
    real_B = Variable(batch["B"]).type(Tensor)
    
    with torch.no_grad(): 
        fake_B = generator(real_A)
    
    # B patches
    B1, B2, B3, B4, B5, B6, B7, B8, B9, B10, B11, B12, B13, B14, B15, B16 = make_16_patches(real_B)
    # fake_B patches
    fB1, fB2, fB3, fB4, fB5, fB6, fB7, fB8, fB9, fB10, fB11, fB12, fB13, fB14, fB15, fB16 = make_16_patches(fake_B)


    # SAVE PATCHES
    img_sample1 = torch.cat((fB1.data, B1.data), -2)
    img_sample2 = torch.cat((fB2.data, B2.data), -2)
    img_sample3 = torch.cat((fB3.data, B3.data), -2)
    img_sample4 = torch.cat((fB4.data, B4.data), -2)
    
    img_sample5 = torch.cat((fB5.data, B5.data), -2)
    img_sample6 = torch.cat((fB6.data, B6.data), -2)
    img_sample7 = torch.cat((fB7.data, B7.data), -2)
    img_sample8 = torch.cat((fB8.data, B8.data), -2)
    
    img_sample9 = torch.cat((fB9.data, B9.data), -2)
    img_sample10 = torch.cat((fB10.data, B10.data), -2)
    img_sample11 = torch.cat((fB11.data, B11.data), -2)
    img_sample12 = torch.cat((fB12.data, B12.data), -2)
    
    img_sample13 = torch.cat((fB13.data, B13.data), -2)
    img_sample14 = torch.cat((fB14.data, B14.data), -2)
    img_sample15 = torch.cat((fB15.data, B15.data), -2)
    img_sample16 = torch.cat((fB16.data, B16.data), -2)
    
    #PATCH
    img_sample_patch = torch.cat((img_sample1.data, 
                                  img_sample2.data, 
                                  img_sample3.data, 
                                  img_sample4.data,
                                  img_sample5.data,
                                  img_sample6.data,
                                  img_sample7.data,
                                  img_sample8.data,
                                  img_sample9.data,
                                  img_sample10.data,
                                  img_sample11.data,
                                  img_sample12.data,
                                  img_sample13.data,
                                  img_sample14.data,
                                  img_sample15.data,
                                  img_sample16.data
                                 ), 1)
    save_image(img_sample_patch, "/home/local/AD/cordun1/experiments/TFC-GAN/images/test_results/%s/%s_p.png" % (opt.experiment, i), nrow=5, normalize=True)


    # GLOBAL
    img_sample_global = torch.cat((real_A.data, fake_B.data, real_B.data), -2)
    save_image(img_sample_global, "/home/local/AD/cordun1/experiments/TFC-GAN/images/test_results/%s/%s_g.png" % (opt.experiment, i), nrow=5, normalize=True)