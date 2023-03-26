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
from lpips_pytorch import LPIPS, lpips
import cv2
from torch.distributed import Backend
#from torch.nn.parallel.distributed import DistributedDataParallel
#import antialiased_cnns
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
# from datasets import * # <- if using labels
from datasets_nolabels import *
from diffusers import DDPMScheduler, UNet2DModel
import antialiased_cnns
from pynvml import *
from transformers import TrainingArguments, Trainer, logging
from accelerate import Accelerator


"""
No TFC-GAN, Just pass B and condition on A
Grayscaled to 1 channel
"""

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=201, help="number of epochs of training") # just start with 10 epochs
parser.add_argument("--checkpoint_interval", type=int, default=20, help="interval between model checkpoints")
parser.add_argument("--dataset", type=str, default="DEVCOM_5perc", help="train/test")
parser.add_argument("--experiment", type=str, default="prototype", help="unique name")
opt = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class ClassConditionedUnet(nn.Module):
    
    def __init__(self, num_classes=4, class_emb_size=4):
        super().__init__()

        # The embedding layer will map the class label to a vector of size class_emb_size
        self.class_emb = nn.Embedding(num_classes, class_emb_size)

        # Self.model is an unconditional UNet with extra input channels to accept the conditioning information (the class embedding)
        # Note: that a time-embedding is already built into this model 
        self.model = UNet2DModel(
            sample_size=128,            # the target image resolution
            in_channels=1*2, # Additional input channels for class cond.
            out_channels=1,           # the number of output channels ====> leave at 3 channels dspite inputs gray
            layers_per_block=1,       # how many ResNet layers to use per UNet block, changed from 2
            block_out_channels=(32, 64, 64), # More channels -> more parameters
            down_block_types=( 
                "DownBlock2D",        # a regular ResNet downsampling block
                "AttnDownBlock2D",    # a ResNet downsampling block with spatial self-attention
                "AttnDownBlock2D",
            ), 
            up_block_types=(
                "AttnUpBlock2D", 
                "AttnUpBlock2D",      # a ResNet upsampling block with spatial self-attention
                "UpBlock2D",          # a regular ResNet upsampling block
              ),
        )
        
    def forward(self, x, t, A):
        # Net(noisy_img, timesteps, labels)
        with autocast(): 
            bs, ch, w, h = x.shape
         
            #A_cond = self.class_emb(class_labels) # Map to embedding dinemsion
            #class_cond = class_cond.view(bs, class_cond.shape[2], 1, 1).expand(bs, class_cond.shape[2], w, h)

            #net_input = torch.cat((x, A), 1) # torch.Size([batch, 3, 128, 128]) + torch.Size([batch, 3, 128, 128])
            net_input = torch.cat((x, A), 1) #grayscale, 1+1 chann
            
            output = (self.model(net_input, t).sample) # (bs, 1, 28, 28)
    
        return output


######################
# Init
######################

loss_fn = nn.MSELoss()
loss_fn.to(device)

Net = ClassConditionedUnet().to(device)

Net = torch.nn.DataParallel(Net, device_ids=[0,1,2])

optimizer_N = torch.optim.Adam(Net.parameters(), lr=1e-3) 

transforms_ = [
    transforms.Resize((128, 128), Image.BICUBIC),
    transforms.Grayscale(), # will make all 1 channel, now ?? will this stop all the colors?
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
    #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]

# use DEVCOM_train_annots_factorized.csv, DEVCOM_5Perc
dataloader = DataLoader(
    ImageDataset(root = "/home/local/AD/cordun1/experiments/data/{}".format(opt.dataset),
        #annots_csv =  "/home/local/AD/cordun1/experiments/data/labels/DEVCOM_updated_complete_annots_factorized.csv",
        transforms_=transforms_),
    batch_size=12,
    shuffle=True,
    num_workers=4,
    drop_last=True,
)


# Tensor type - only use HalfTensor in this AMP script
HalfTensor = torch.cuda.HalfTensor if device else torch.HalfTensor
Tensor = torch.cuda.FloatTensor if device else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if device else torch.LongTensor


##############################
#       Training
##############################

# Scheduler
noise_scheduler = DDPMScheduler(num_train_timesteps=500, beta_schedule='squaredcos_cap_v2') # cut down from 1000 -> 500
print(noise_scheduler)

# Try AMP
scaler = GradScaler()

for epoch in range(opt.epoch, opt.n_epochs):

    for i, batch in tqdm(enumerate(dataloader)):
        real_A = Variable(batch["A"].type(HalfTensor))
        real_B = Variable(batch["B"].type(HalfTensor))
        #labels = Variable(batch["LAB"].type(LongTensor)) # ethnicity
        
        optimizer_N.zero_grad()

        with autocast():
            noise = torch.randn_like(real_B) 
            timesteps = torch.randint(0, 499, (real_B.shape[0],)).long().to(device)

            noisy_img = noise_scheduler.add_noise(real_B, noise, timesteps)
                   
            # pred noise, condition on the visible image this time
            pred = Net(noisy_img, timesteps, real_A)

            # noise loss
            loss_noise = (loss_fn(pred, noise)).mean()

        scaler.scale(loss_noise).backward()
        scaler.step(optimizer_N)
        scaler.update()

     # Save
    if opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0:
        torch.save(Net.state_dict(), "/home/local/AD/cordun1/experiments/TF-Diff/saved_models/%s/Net_%d.pth" % (opt.experiment, epoch))
    
