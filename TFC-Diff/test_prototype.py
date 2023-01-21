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
from torch.autograd import Variable

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
        bs, ch, w, h = x.shape
        print("-----x {} ---- t {} --".format(x.get_device(), t.get_device()))
        class_cond = self.class_emb(class_labels) # Map to embedding dinemsion
        print("class_cond HERE:", class_cond)
        class_cond = class_cond.view(bs, class_cond.shape[2], 1, 1).expand(bs, class_cond.shape[2], w, h)
        net_input = torch.cat((x, class_cond), 1) # torch.Size([4, 3, 128, 128])
        output = self.model(net_input, t).sample # (bs, 1, 28, 28)

        return output


######################
# Init
######################

Net = ClassConditionedUnet().to(device)

#Net = torch.nn.DataParallel(Net, device_ids=[0,1,2])

transforms_ = [
    transforms.Resize((128, 128), Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]

test_dataloader = DataLoader(
    ImageDataset(root = "/home/local/AD/cordun1/experiments/data/DEVCOM_5perc",
        transforms_=transforms_,
        annots_csv = "/home/local/AD/cordun1/experiments/data/labels/DEVCOM_updated_complete_annots_factorized.csv",
        mode="test"),
    batch_size=1,
    shuffle=True,
    num_workers=1,
)


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
    
    
Net_path = "/home/local/AD/cordun1/experiments/TF-Diff/saved_models/prototype/Net_1.pth"
load_clean_state(Net, Net_path)

# OK THE MODEL LOADS!

##############################
#       Initialize
##############################
LongTensor = torch.cuda.LongTensor if device else torch.LongTensor

# Scheduler
noise_scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule='squaredcos_cap_v2')

# X,Y have to be on CUDA
for k, batch in tqdm(enumerate(test_dataloader)):
    real_A = Variable(batch["A"]).to(device)
    real_B = Variable(batch["B"]).to(device)
    labels = Variable(batch["LAB"].type(LongTensor)).to(device)

    # random x to start from?
    sample = torch.randn(real_B.shape[0], 3, 128, 128).to(device)
    #labels = labels.to(device)

    # Sampling loop
    for i, t in tqdm(enumerate(noise_scheduler.timesteps)):
        print("* * * i: {} * * * ".format(i))
        t = t.to(device)
        t = torch.flatten(t)

        with torch.no_grad():
            # INPUTS MUST ALL BE CUDA
            print("NET() -> CUDA: sample {} | t {} | labels {}:".format(sample.get_device(), t.get_device(), labels.get_device()))
            residual = Net(sample, t, labels) 
        
        #print("PHASE 1: residual {} | t {} | sample {}".format(residual.get_device(), t.get_device(), sample.get_device()))
        # maybe these need to be put on CPU? YES INPUTS MUST ALL BE CPU
        residual = residual.to('cpu')
        t = t.to('cpu')
        sample = sample.to('cpu')
        print("NOISE SCHED -> CPU: residual {} | t {} | sample {}".format(residual.get_device(), t.get_device(), sample.get_device()))
        
        sample = noise_scheduler.step(residual, t, sample).prev_sample
        sample = sample.to(device)
        

    printable_sample = sample.clone()        
    img_sample_global = torch.cat((printable_sample.to(device).data, real_B.to(device).data), -2)
    save_image(img_sample_global, "images/prototype/%s.png" % k, nrow=3, normalize=True)
# Show the results, which are x from the noise_scheduler
#fig, ax = plt.subplots(1, 1, figsize=(12, 12))
#ax.imshow(torchvision.utils.make_grid(x.detach().cpu().clip(-1, 1), nrow=8)[0], cmap='Greys')