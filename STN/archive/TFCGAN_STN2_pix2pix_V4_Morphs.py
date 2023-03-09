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
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import antialiased_cnns
from datasets_stn import *
from kornia import morphology as morph

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=210, help="number of epochs of training")
parser.add_argument("--dataset_name", type=str, default="eurecom_warped_pairs", help="name of the dataset")
parser.add_argument("--batch_size", type=int, default=12, help="size of the batches")
parser.add_argument("--lr", type=float, default=1e-4, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_height", type=int, default=256, help="size of image height")
parser.add_argument("--img_width", type=int, default=256, help="size of image width")
#parser.add_argument("--patch_height", type=int, default=128, help="size of patch height")
#parser.add_argument("--patch_width", type=int, default=128, help="size of patch width")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=200, help="interval between sampling of images from generators")
parser.add_argument("--checkpoint_interval", type=int, default=50, help="interval between model checkpoints")
parser.add_argument("--gpu_num", type=int, default=0, help="gpu card")
#parser.add_argument("--out_file", type=str, default="out", help="name of output log files")
parser.add_argument("--experiment", type=str, default="tfcgan_stn_arar", help="experiment name")
opt = parser.parse_args()


os.makedirs("/home/local/AD/cordun1/experiments/TFC-GAN/images/%s" % opt.experiment, exist_ok=True)
os.makedirs("/home/local/AD/cordun1/experiments/TFC-GAN/saved_models/%s" % opt.experiment, exist_ok=True)

cuda = True if torch.cuda.is_available() else False
torch.cuda.set_device(opt.gpu_num) 
# Set fixed random number seed
torch.manual_seed(42)

"""
Paper: The experiments were conducted on single GeForce RTX 2080 Ti. 
We use Adam Optimizer [17] on a mini- batch of size 12 with parameters 
lr = 1 × e−4, β1 = 0.5 and β2 = 0.999. We train our model for 200 epochs, 
and activate linear learning rate decay after 100 epochs.
"""


##########################
# Loss functions
##########################
#criterion_GAN = torch.nn.BCEWithLogitsLoss() 
criterion_GAN = torch.nn.MSELoss()
criterion_L1= torch.nn.L1Loss()
criterion_lpips = LPIPS(
    net_type='vgg',  # choose a network type from ['alex', 'squeeze', 'vgg']
    version='0.1'  # Currently, v0.1 is supported
)

############################
#  Utils
############################

# Calculate output of image discriminator (PatchGAN)
patch = (1, opt.img_height // 2 ** 4, opt.img_width // 2 ** 4)

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

           
def sample_images(batches_done):
    imgs = next(iter(test_dataloader)) # batch_size = 1
    real_A = Variable(imgs["A"].type(HalfTensor))
    real_B = Variable(imgs["B"].type(HalfTensor))

    fake_A_init = generator(real_B) 
    warped_fA = model(img_A=real_A, img_B=real_B, src=fake_A_init)
    T_ = torch.cat(warped_fA) 

    warped_B = model(img_A=real_A, img_B=real_B, src=real_B)
    R_ = torch.cat(warped_B)
    fake_A_R = generator(R_)
    
    # Masks
    mask_A = morph_masks(real_A)
    mask_reg_B = morph_masks(R_)

    # GLOBAL
    """
    <A, 
    warped fake_A, 
    warped real_B, # <---- this what you're looking for
    fake_A from warped_B,
    real_B>
    """
    img_sample_global = torch.cat((real_A.data, T_.data, R_.data, fake_A_R.data, real_B.data, mask_A.data, mask_reg_B.data), -2)
    save_image(img_sample_global, "images/%s/%s_g.png" % (opt.experiment, batches_done), nrow=5, normalize=True)

"""
this_img_src = img.reshape(1, img.size(0), img.size(1), img.size(2))
RuntimeError: shape '[1, 1, 3, 256]' is invalid for input of size 196608
"""
    
    
    
def print_network(model, name):
    """Print out the network information."""
    num_params = 0
    for p in model.parameters():
        num_params += p.numel()
    print(model)
    print(name)
    print("The number of parameters: {}".format(num_params))


def morph_masks(tensor):
    kernel = torch.tensor([[0, 1, 0],[1, 1, 1],[0, 1, 0]])
    morph_grad = morph.gradient(tensor.cuda(), kernel.cuda()) # Morphological gradient
    morph_tensor = 1. - morph_grad
    return morph_tensor

#################    
# UNET for STN
################

class LocalizerUNet(nn.Module):
    def __init__(self, img_shape):
        super(LocalizerUNet, self).__init__()
        channels, self.h, self.w = img_shape
        self.down1 = UNetDown(channels*2, 64)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 512)
        self.down5 = UNetDown(512, 512)
        self.down6 = UNetDown(512, 512)

        self.up1 = UNetUp(512, 512)
        self.up2 = UNetUp(1024, 512)        
        self.up3 = UNetUp(1024, 256)
        self.up4 = UNetUp(512, 128)
        self.up5 = UNetUp(256, 64) # Then it'll only be 64 x 64

        # Not sure if I need this?
        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, channels, 3, padding=1),
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
            #out = u5.type(HalfTensor)
            #print("STN UNET OUT:", out.size()) # no self.final layer b/c I have an error in gradient dims
        
            out = self.final(u5).type(HalfTensor)
        return out
    
#################
# STN
#################
                          
class Net(nn.Module):
    """simplified from official: 
    https://github.com/moabarar/nemar/blob/nemar_deploy/models/stn/affine_stn.py"""
    def __init__(self):
        super(Net, self).__init__()

        """
        # FIRST MODEL: Spatial transformer localization-network
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

        # SECOND MODEL 
        self.localization = nn.Sequential(
            nn.Conv2d(opt.channels*2, 64, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 3, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 3, 2, 1),
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(512*64*64, 256), # <- change this
            nn.ReLU(True),
            nn.Linear(256, 3 * 2)
        )
        
        """
        # UNET
        input_shape = (opt.channels, opt.img_height, opt.img_width)
        self.localization = LocalizerUNet(input_shape)
        
         # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(3*256*256, 256), 
            nn.ReLU(True),
            nn.Linear(256, 3 * 2)
        )

        #self.fc_loc[2].weight.data.normal_(mean=0.0, std=5e-4) # DO NOT INITIATE THIS! IT WILL NEVER LEARN THE WEIGHTS AND THIS HAS BEEN THE BANE OF EVERYTHING. 
        self.fc_loc[2].bias.data.zero_() # DO NOT CHANGE! the problem for everything is this, don't change it
        # what happens if I don't do this ---> If you on't add this, it will cause gray boxes around everything
       
    # Spatial transformer network forward function
    def stn_phi(self, x):
        xs = self.localization(x) # convolves the cat channel=6
        #print("stn_phi xs:", xs.size()) 
        xs = xs.view(-1, 3*256*256) # May need to change since I changed STN UNET to u5 as out
        theta = self.fc_loc(xs)
        #print("theta.size:", theta.size())
        
        theta = theta.view(-1, 2, 3)
        #print("theta.view.size:", theta.size())
        
        return theta

    def forward(self, img_A, img_B, src):
        with autocast():
            img_input = torch.cat((img_A, img_B), 1)
            dtheta = self.stn_phi(img_input) # deformation field for real A and real B

            identity_theta = torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float).cuda()
            dtheta = dtheta.reshape(img_B.size(0), 2*3)
            theta = dtheta + identity_theta.unsqueeze(0).repeat(img_B.size(0), 1)
            # The theta is what's being learned by the NN - grid and grid_sample are just functions

            theta_batches = []
            for t in theta:
                this_theta = (t.view(-1, 2, 3)).reshape(1,2,3)
                theta_batches.append(this_theta)

            src_tensors = []
            for img in src: 
                #print("src img size:", img.size())
                this_img_src = img.reshape(1, img.size(0), img.size(1), img.size(2)) 
                src_tensors.append(this_img_src)
                
            warped = []
            for i in range(len(src_tensors)): #1:1 match with theta, matching with theta is important
                rs_grid = F.affine_grid(theta_batches[i], src_tensors[i].size(), align_corners=True)
                Rs = F.grid_sample(src_tensors[i], rs_grid,  mode='nearest', padding_mode='zeros', align_corners=True)
                warped.append(Rs.type(HalfTensor))

        return warped
    
    
    """
    def forward(self, img_A, img_B, src):
        # src = [real_A, fake_B] a list of tensors
        with autocast():
            img_input = torch.cat((img_A, img_B), 1)
            dtheta = self.stn_phi(img_input) # deformation field for real A and real B

            identity_theta = torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float).cuda()
            dtheta = dtheta.reshape(img_B.size(0), 2*3)
            theta = dtheta + identity_theta.unsqueeze(0).repeat(img_B.size(0), 1)
            # The theta is what's being learned by the NN - grid and grid_sample are just functions

            theta_batches = []
            for t in theta:
                # torch.Size([1, 2, 3]) for theta0, theta1, theta2, theta3, since there are 4 batches
                #this_theta = (theta.view(-1, 2, 3)[t]).reshape(1,2,3)
                this_theta = (t.view(-1, 2, 3)).reshape(1,2,3)
                theta_batches.append(this_theta)

            src0 = []
            for img in src[0]: 
                this_img_src0 = img.reshape(1, img.size(0), img.size(1), img.size(2)) 
                src0.append(this_img_src0)

            src1 = []
            for img_ in src[1]: 
                this_img_src1 = img_.reshape(1, img_.size(0), img_.size(1), img_.size(2)) 
                src1.append(this_img_src1)

            warped_0 = []
            for i in range(len(src0)): #1:1 match with theta, matching with theta is important
                rs_grid_0 = F.affine_grid(theta_batches[i], src0[i].size(), align_corners=True)
                Rs_0 = F.grid_sample(src0[i], rs_grid_0,  mode='nearest', padding_mode='zeros', align_corners=True)
                warped_0.append(Rs_0.type(HalfTensor))

            # match with src - fake_B
            warped_1 = []
            for i in range(len(src1)): #1:1 match with theta, matching with theta is important
                rs_grid_1 = F.affine_grid(theta_batches[i], src1[i].size(), align_corners=True)
                Rs_1 = F.grid_sample(src1[i], rs_grid_1,  mode='nearest', padding_mode='zeros', align_corners=False)
                warped_1.append(Rs_1.type(HalfTensor))

            reg_term = torch.mean(torch.abs(dtheta))

        return warped_0, warped_1, reg_term
        """
            
                     

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
"""    
     
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

"""

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
            out = self.final(u5).type(HalfTensor)
        return out

    
##############################
#        Discriminator
##############################

class Discriminator(nn.Module):

    def __init__(self, img_shape):
        super(Discriminator, self).__init__()
        channels, self.h, self.w = img_shape

        """
        def discriminator_block(in_filters, out_filters, normalization=True):
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalization:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block((channels * 2), 64, normalization=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1, bias=False)
        )
        """
        # This one has BlurPool and SN
        def discriminator_block(in_filters, out_filters):
            layers = [torch.nn.utils.parametrizations.spectral_norm(
                nn.Conv2d(in_filters, out_filters, 4, stride=1, padding=1))] # changed to stride=1 instead of 2
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            layers.append(antialiased_cnns.BlurPool(out_filters, stride=2)) #blurpool downsample stride=2
            
            return layers

        self.model = nn.Sequential(
            *discriminator_block((channels * 2), 64),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1, bias=False)
        )
       
    def forward(self, img_A, img_B):
        with autocast():
            img_input = torch.cat((img_A, img_B), 1)
            d_in = img_input
            output = self.model(d_in)
        return output.type(HalfTensor)
    
    
    
# ===========================================================
# Initialize generator and discriminator
input_shape = (opt.channels, opt.img_height, opt.img_width)

generator = GeneratorUNet(input_shape)
discriminator = Discriminator(input_shape)

# mask loss
criterion_mse = torch.nn.MSELoss()

if cuda:
    generator = generator.cuda()
    discriminator = discriminator.cuda()
    criterion_GAN.cuda()
    criterion_L1.cuda()
    criterion_lpips.cuda()
    model = Net().cuda()
    criterion_mse.cuda()

generator = torch.nn.DataParallel(generator, device_ids=[0, 1, 2])
discriminator = torch.nn.DataParallel(discriminator, device_ids=[0, 1, 2])
model = torch.nn.DataParallel(model, device_ids=[0, 1, 2])                       

if opt.epoch != 0:
    # Load pretrained models
    generator.load_state_dict(torch.load("/home/local/AD/cordun1/experiments/TFC-GAN/saved_models/%s/generator_%d.pth" % (opt.experiment, opt.epoch)))
    discriminator.load_state_dict(torch.load("/home/local/AD/cordun1/experiments/TFC-GAN/saved_models/%s/discriminator_%d.pth" % (opt.experiment, opt.epoch)))
    model.load_state_dict(torch.load("/home/local/AD/cordun1/experiments/TFC-GAN/saved_models/%s/stn_%d.pth" % (opt.experiment, opt.epoch)))

else:
    # Initialize weights
    generator.apply(weights_init_normal)
    discriminator.apply(weights_init_normal)
    model.apply(weights_init_normal) # Need this for the STN UNET and Affine

# Optimizers
optimizer_G = torch.optim.Adam(itertools.chain(generator.parameters(), model.parameters()), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

# Tensor type - only use HalfTensor in this AMP script                    
Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor
HalfTensor = torch.cuda.HalfTensor if cuda else torch.HalfTensor


##############################
# Transforms and Dataloaders
##############################

#Resizing happens in the ImageDataset() to 256 x 256 so that I can get patches
transforms_ = [
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]

dataloader = DataLoader(
    ImageDataset(root = "/home/local/AD/cordun1/experiments/data/%s" % opt.dataset_name,
        transforms_=transforms_),
    batch_size=opt.batch_size,
    shuffle=True,
    num_workers=opt.n_cpu,
    drop_last=True,
)

test_dataloader = DataLoader(
    TestImageDataset(root = "/home/local/AD/cordun1/experiments/data/%s" % opt.dataset_name,
        transforms_=transforms_,
        mode="test"),
    batch_size=1,
    shuffle=True,
    num_workers=1,
)


##############################
#       Training
##############################

prev_time = time.time()

# Try AMP
scaler = GradScaler()

for epoch in range(opt.epoch, opt.n_epochs):
    for i, batch in enumerate(dataloader):
        print("****** {} ******\n".format(i))
        real_A = Variable(batch["A"].type(HalfTensor))
        real_B = Variable(batch["B"].type(HalfTensor))

        # Adversarial ground truths
        valid_ones = Variable(HalfTensor(np.ones((real_A.size(0), *patch))), requires_grad=False)
        valid = valid_ones.fill_(0.9)
        fake = Variable(HalfTensor(np.zeros((real_A.size(0), *patch))), requires_grad=False) 
               
        # ------------------
        #  Train Generator
        # ------------------
        optimizer_G.zero_grad()
        print("+ + + optimizer_G.zero_grad() + + +")   
        with autocast(): 
            
            print("Flow 1 - Translation First")
            fake_A_init = generator(real_B) 
            warped_fA = model(img_A=real_A, img_B=real_B, src=fake_A_init)
            T_ = torch.cat(warped_fA) 
            
            print("Flow 2 - Register First")
            warped_B = model(img_A=real_A, img_B=real_B, src=real_B)
            R_ = torch.cat(warped_B)
            fake_A_R = generator(R_)
         
            # Rconstruction error
            reconstruction_loss = criterion_L1(T_, real_A) + criterion_L1(fake_A_R, real_A)
            
            # LPIPS loss - perceptual similarity
            """Very interesting: You can't have LPIPS at all, even if it contributes to a fraction of the G loss
            It will cause mode collapse, even if added to L1"""
            #loss_pix_g = criterion_lpips(T_, real_A) + criterion_lpips(fake_A_R, real_A)
            
            # MSE Mask Loss
            # MSE loss - how well alligned is the registered B with real Visible? 
            mask_A = morph_masks(real_A)
            mask_reg_B = morph_masks(R_)
            loss_mask = criterion_mse(mask_A, mask_reg_B)
                    
            # Adv 
            pred_fake_Transfirst = discriminator(T_.cuda(), real_B)
            pred_fake_Regfirst = discriminator(fake_A_R.cuda(), real_B)
            loss_GAN = criterion_GAN(pred_fake_Transfirst, valid) + criterion_GAN(pred_fake_Regfirst, valid) # Changed to MSE Loss like pix2pix

            # Total Loss
            loss_G = loss_GAN + reconstruction_loss + loss_mask
            """
            RuntimeError: Function AddmmBackward returned an invalid gradient at index 1 - got [4, 196608] but expected shape compatible with [4, 2097152]
            Got: (3,256,256), but expects (128,128,128)
            # -> I fixed this by just adding back the STN UNET final layer to get it to output 3,256,256
            """
        scaler.scale(loss_G).backward()
        scaler.step(optimizer_G)
        print("+ + + optimizer_G.step() + + +")
        
        # -----------------------
        #  Train Discriminator 
        # -----------------------

        optimizer_D.zero_grad()
        print("+ + + optimizer_D.zero_grad() + + +")
        with autocast():

            # Real loss
            pred_real = discriminator(real_A, real_B)
            loss_real = criterion_GAN(pred_real, valid)

            # Fake loss of Transfirst
            pred_fake_TR_d = discriminator(T_.detach(), real_B)
            loss_fake_TR_d = criterion_GAN(pred_fake_TR_d, fake)
                    
            # Fake loss of Regfirst
            pred_fake_RT_d = discriminator(fake_A_R.detach(), real_B)
            loss_fake_RT_d = criterion_GAN(pred_fake_RT_d, fake)

            # Total loss
            loss_D = 0.5*(loss_real + loss_fake_TR_d + loss_fake_RT_d)

        scaler.scale(loss_D).backward()
        scaler.step(optimizer_D)
        print("+ + + optimizer_D.step() + + +")
        
        # one scalar update for all scaler
        # instead of one for each model
        scaler.update()

        # --------------
        #  Log Progress
        # --------------

        # Determine approximate time left
        batches_done = epoch * len(dataloader) + i
        #print("batches_done:", batches_done)
        batches_left = opt.n_epochs * len(dataloader) - batches_done
        time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
        prev_time = time.time()

        # Print log
        sys.stdout.write(
            "\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, L1: %f, Mask: %f, GAN: %f] ETA: %s"
            % (
                epoch, #%d
                opt.n_epochs, #%d
                i, #%d
                len(dataloader), #%d
                loss_D.item(), #%f
                loss_G.item(), #%f - total G loss
                reconstruction_loss.item(),
                loss_mask.item(),
                loss_GAN.item(), #%f - adv G loss
                time_left, #%s
            )
        )
        
        # If at sample interval save image
        if batches_done % opt.sample_interval == 0:
            sample_images(batches_done)

    # You must save the STN model b/c this is what's used for registration
    """Once trained, only the spatial transformation network R is used in test time. 
    The network takes two images Ia and Ib representing the same scene, 
    captured from slightly different viewpoint, in two different modalities, 
    A and B, respectively, and aligns Ia with Ib."""
    if opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0:
        # Save model checkpoints
        torch.save(generator.state_dict(), "/home/local/AD/cordun1/experiments/TFC-GAN/saved_models/%s/generator_%d.pth" % (opt.experiment, epoch))
        torch.save(discriminator.state_dict(), "/home/local/AD/cordun1/experiments/TFC-GAN/saved_models/%s/discriminator_%d.pth" % (opt.experiment, epoch))
        torch.save(model.state_dict(), "/home/local/AD/cordun1/experiments/TFC-GAN/saved_models/%s/stn_%d.pth" % (opt.experiment, epoch))

                   