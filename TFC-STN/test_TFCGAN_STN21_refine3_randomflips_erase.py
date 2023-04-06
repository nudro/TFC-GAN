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
import antialiased_cnns
import kornia
import torch.nn as nn
import kornia.contrib as K
from datasets_augs_erase import * # augmention random H/V flisp A,B together

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0, help="epoch to load")
parser.add_argument("--dataset_name", type=str, default="eurecom_warped_pairs", help="name of the dataset")
parser.add_argument("--img_height", type=int, default=256, help="size of image height")
parser.add_argument("--img_width", type=int, default=256, help="size of image width")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--experiment", type=str, default="tfcgan_stn", help="experiment name")
opt = parser.parse_args()

"""
Random Flips for A-B pairs to test robustness of model at inference.
"""

os.makedirs("/home/local/AD/cordun1/experiments/TFC-GAN-STN/images/test_results/%s_FLIPPED_ERASED" % opt.experiment, exist_ok=True)
# images/test_results/%s_FLIPPED
cuda = True if torch.cuda.is_available() else False


# Set fixed random number seed
torch.manual_seed(42)

#################    
# ViT for STN
################

class LocalizerVIT(nn.Module):
    def __init__(self, img_shape):
        super(LocalizerVIT, self).__init__()
        channels, self.h, self.w = img_shape
        self.vit = nn.Sequential(
            K.VisionTransformer(image_size=self.h, patch_size=64, in_channels=channels*2) # (A,B), changed from 16-> 32 patch
        )
        
    def forward(self, x):
        # patch 16: torch.Size([batch, 257, 768])
        # patch 32: torch.Size([batch, 65, 768]) 
        # patch 64: torch.Size([batch, 17, 768])
        with autocast():
            out = self.vit(x).type(HalfTensor) # returns at patch_size = 16, torch.Size([batch, 257, 768])
            #print("out Vit:", out.size())
        return out


####################################################################
# Stacked STN, based on Densely Fused Transformer Networks
####################################################################
                          
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        input_shape = (opt.channels, opt.img_height, opt.img_width)

        self.localization = LocalizerVIT(input_shape)

        self.theta_emb = nn.Linear(1, opt.img_height * opt.img_width)

        # nn.Linear(1*257*768, 1024), # (hard-coded in, 257, 768 based on ViT output)
        self.fc_loc = nn.Sequential(
            nn.Linear(1*17*768, 1024), 
            nn.ReLU(True),
            nn.Linear(1024, 512), # Added more layers, will this stop the affine warp?
            nn.ReLU(True),
            nn.Linear(512, 256),
            nn.Sigmoid(), # <--- SIGMOID will this force the matrix values between [-1,+1]
            nn.Linear(256, 3*2))

        self.fc_loc[2].bias.data.zero_() # DO NOT CHANGE! the problem for everything is this, don't change it

       
    def stn_phi(self, x):
        xs = self.localization(x) # (A,B), 6 ch
        xs = xs.view(-1, 1 * xs.size(1) * xs.size(2)) 
        theta = self.fc_loc(xs)  
        theta = theta.view(-1, 2, 3)
        return theta
    

    def forward(self, img_A, img_B, src):
        identity_matrix = [1, 0, 0, 0, 1, 0] 
        
        with autocast(): 
            
            img_input = torch.cat((img_A, img_B), 1)
            dtheta = self.stn_phi(img_input) 
            identity_theta = torch.tensor(identity_matrix, dtype=torch.float).cuda()
            dtheta = dtheta.reshape(img_A.size(0), 2*3)
            dtheta = dtheta + identity_theta.unsqueeze(0).repeat(img_A.size(0), 1)

            # get each theta for the batch
            theta_batches = []
            for t in dtheta:
                this_theta = (t.view(-1, 2, 3)).reshape(1,2,3)
                theta_batches.append(this_theta)

            # just the tensors of the source image to be aligned
            src_tensors = []
            for img in src: 
                this_img_src = img.reshape(1, img.size(0), img.size(1), img.size(2)) 
                src_tensors.append(this_img_src)

            # result
            warped = []
            for i in range(len(src_tensors)):
                rs_grid = F.affine_grid(theta_batches[i], src_tensors[i].size(), align_corners=True) 
                Rs = F.grid_sample(src_tensors[i], rs_grid,  mode='bicubic', padding_mode='border', align_corners=True)  
                warped.append(Rs.type(HalfTensor))
                
        return torch.cat(warped)


##############################
#     Generator  U-NET
# These have BlurPool b/c it's TFC-GAN
##############################

class UNetDown(nn.Module):
    def __init__(self, in_size, out_size, normalize=True, dropout=0.0):
        super(UNetDown, self).__init__()
        layers = [nn.Conv2d(in_size, out_size, 4, 1, 1, bias=False)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_size))
        layers.append(nn.LeakyReLU(0.2))
        layers.append(antialiased_cnns.BlurPool(out_size, stride=2))
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
                antialiased_cnns.BlurPool(out_size, stride=1), 
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
    

class GeneratorUNet1(nn.Module):
    def __init__(self, img_shape):
        super(GeneratorUNet1, self).__init__()
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
        with autocast():
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
    
    
class GeneratorUNet2(nn.Module):
    def __init__(self, img_shape):
        super(GeneratorUNet2, self).__init__()
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
        with autocast():
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
input_shape_patch = (opt.channels, opt.img_height, opt.img_width)
generator1 = GeneratorUNet1(input_shape_patch).cuda() # for fake_B
generator2 = GeneratorUNet2(input_shape_patch).cuda() # for fake_A
model = Net().cuda()


g1_path = "/home/local/AD/cordun1/experiments/TFC-GAN-STN/saved_models/%s/generator1_%d.pth" % (opt.experiment, opt.epoch)
load_clean_state(generator1, g1_path)

g2_path = "/home/local/AD/cordun1/experiments/TFC-GAN-STN/saved_models/%s/generator2_%d.pth" % (opt.experiment, opt.epoch)
load_clean_state(generator2, g2_path)

stn_path = "/home/local/AD/cordun1/experiments/TFC-GAN-STN/saved_models/%s/stn_%d.pth" % (opt.experiment, opt.epoch)
load_clean_state(model, stn_path)


##############################
# Transforms and Dataloaders
##############################
# To Keep A,B, transformed as a flipped pair, together, you use 'datasets_augs.py 'class
# The trasnforms are 'baked into' the script

test_dataloader = DataLoader(
    TestDataset(root = "/home/local/AD/cordun1/experiments/data/%s" % opt.dataset_name,
        transforms_= None, mode="test"),
    batch_size=1, 
    shuffle=False,
    num_workers=8,
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
    real_A = Variable(batch["A"].type(Tensor))
    real_B = Variable(batch["B"].type(Tensor))
    
    with torch.no_grad():    
        fake_B = generator1(real_A)
        fake_A1 = generator2(real_B)

        # pass to generator 2 for fake_A
        warped_B = model(img_A=real_A, img_B=fake_A1, src=real_B) 
        fake_A2 = generator2(warped_B)
    
    # GLOBAL
    img_sample_global = torch.cat((real_A.data, real_B.data, warped_B.data, fake_A1.data, fake_B.data, fake_A2.data), -1)
    save_image(img_sample_global, "images/test_results/%s_FLIPPED_ERASED/%s.png" % (opt.experiment, i), nrow=4, normalize=True)
