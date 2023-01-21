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
from datasets import *
from diffusers import DDPMScheduler, UNet2DModel
import antialiased_cnns
from pynvml import *
from transformers import TrainingArguments, Trainer, logging
from accelerate import Accelerator

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=5, help="number of epochs of training") # just start with 10 epochs
opt = parser.parse_args()

def print_gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used//1024**2} MB.")


def print_summary(result):
    print(f"Time: {result.metrics['train_runtime']:.2f}")
    print(f"Samples/second: {result.metrics['train_samples_per_second']:.2f}")
    print_gpu_utilization()
    
    
device = 'cuda' if torch.cuda.is_available() else 'cpu'

###### // Diffuser // 
class ClassConditionedUnet(nn.Module):
    
    def __init__(self, num_classes=4, class_emb_size=4):
        super().__init__()

        # The embedding layer will map the class label to a vector of size class_emb_size
        self.class_emb = nn.Embedding(num_classes, class_emb_size)

        # Self.model is an unconditional UNet with extra input channels to accept the conditioning information (the class embedding)
        # Note: that a time-embedding is already built into this model 
        self.model = UNet2DModel(
            sample_size=128,            # the target image resolution
            in_channels=3 + class_emb_size, # Additional input channels for class cond.
            out_channels=3,           # the number of output channels
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
        
    def forward(self, x, t, class_labels):
        with autocast(): 
            bs, ch, w, h = x.shape
            class_cond = self.class_emb(class_labels) # Map to embedding dinemsion
            class_cond = class_cond.view(bs, class_cond.shape[2], 1, 1).expand(bs, class_cond.shape[2], w, h)
            net_input = torch.cat((x, class_cond), 1) # torch.Size([4, 3, 128, 128])
            output = (self.model(net_input, t).sample) # (bs, 1, 28, 28)
            
            print_gpu_utilization()
    
        return output


###### // TFC-GAN//

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
            output = self.final(u5)
            
        print_gpu_utilization()
            
        return output
    

##### / weight init for UNET /
def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

######################
# Init
######################

# / Generator 
input_shape_global = (3, 128, 128)
generator = GeneratorUNet(input_shape_global).to(device)
generator = generator.to(device)

criterion_lpips = LPIPS(
    net_type='vgg',  # choose a network type from ['alex', 'squeeze', 'vgg']
    version='0.1'  # Currently, v0.1 is supported
)

criterion_lpips.to(device)

# / Diffuser
loss_fn = nn.MSELoss()
loss_fn.to(device)

Net = ClassConditionedUnet().to(device)

if opt.epoch != 0:
    Net.load_state_dict(torch.load("/home/local/AD/cordun1/experiments/TF-Diff/saved_models/prototype2/Net_%d.pth" % opt.epoch))
    generator.load_state_dict(torch.load("/home/local/AD/cordun1/experiments/TF-Diff/saved_models/prototype2/generator_%d.pth" % opt.epoch))

else:
    generator.apply(weights_init_normal)

Net = torch.nn.DataParallel(Net, device_ids=[0,1,2])
generator = torch.nn.DataParallel(generator, device_ids=[0,1,2])

optimizer_N = torch.optim.Adam(Net.parameters(), lr=1e-3) 
optimizer_G = torch.optim.Adam(generator.parameters(), lr=1e-3)

transforms_ = [
    transforms.Resize((128, 128), Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]

# use DEVCOM_train_annots_factorized.csv, DEVCOM_5Perc
dataloader = DataLoader(
    ImageDataset(root = "/home/local/AD/cordun1/experiments/data/DEVCOM_5perc",
        annots_csv =  "/home/local/AD/cordun1/experiments/data/labels/DEVCOM_updated_complete_annots_factorized.csv",
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
noise_scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule='squaredcos_cap_v2')
print(noise_scheduler)

# Try AMP
scaler = GradScaler()

for epoch in range(opt.epoch, opt.n_epochs):

    for i, batch in tqdm(enumerate(dataloader)):
        real_A = Variable(batch["A"]).to(device)
        real_B = Variable(batch["B"]).to(device)
        labels = Variable(batch["LAB"].type(LongTensor)) # ethnicity
        
        optimizer_N.zero_grad()
        optimizer_G.zero_grad()

        with autocast():
            fake_B = generator(real_A).to(device) # I really don't know if this needs a discriminator?????
            #print("fake_B: {} | real_B: {}".format(fake_B.size(), real_B.size()))
            loss_recon = criterion_lpips(fake_B, real_B)
    
            noise = torch.randn_like(fake_B) 
            timesteps = torch.randint(0, 999, (fake_B.shape[0],)).long().to(device)

            noisy_img = noise_scheduler.add_noise(fake_B, noise, timesteps)
                   
            # pred noise
            pred = Net(noisy_img, timesteps, labels)

            # noise loss
            loss_noise = (loss_fn(pred, noise)).mean()
        
            # total noise
            #loss = loss_noise + loss_recon
            
            print("recon loss: {} | noise loss: {}".format(loss_recon, loss_noise))

        scaler.scale(loss_noise).backward(retain_graph=True)
        scaler.scale(loss_recon).backward(retain_graph=True)
        scaler.step(optimizer_N)
        scaler.step(optimizer_G)
        scaler.update()
        


    # Save
    torch.save(Net.state_dict(), "/home/local/AD/cordun1/experiments/TF-Diff/saved_models/prototype2/Net_%d.pth" % epoch)
    torch.save(generator.state_dict(), "/home/local/AD/cordun1/experiments/TF-Diff/saved_models/prototype2/G_%d.pth" % epoch)
