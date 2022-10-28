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
from torchvision.models import resnet18
from datasets_thermalgan import *

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--dataset_name", type=str, default="facades", help="name of the dataset")
parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.00002, help="adam: learning rate - from ThermalGAN paper") #inf loss, so i pushed to 1e-5
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_height", type=int, default=256, help="size of image height")
parser.add_argument("--img_width", type=int, default=256, help="size of image width")
parser.add_argument("--patch_height", type=int, default=128, help="size of patch height")
parser.add_argument("--patch_width", type=int, default=128, help="size of patch width")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=10, help="interval between sampling of images from generators")
parser.add_argument("--checkpoint_interval", type=int, default=100, help="interval between model checkpoints")
parser.add_argument("--gpu_num", type=int, default=0, help="0")
parser.add_argument("--out_file", type=str, default="out", help="name of output log files")
parser.add_argument("--experiment", type=str, default="none", help="experiment name")
parser.add_argument("--latent_dim", type=int, default=8, help="number of latent codes - BicycleGAN")
parser.add_argument("--lambda_pixel", type=float, default=10, help="pixelwise loss weight - BicycleGAN")
parser.add_argument("--lambda_latent", type=float, default=0.5, help="latent loss weight - BicycleGAN")
parser.add_argument("--lambda_kl", type=float, default=0.01, help="kullback-leibler loss weight - BicycleGAN")
opt = parser.parse_args()

"""
My ThermalGAN Implementation

"""

os.makedirs("/home/local/AD/cordun1/experiments/TFC-GAN/images/%s" % opt.experiment, exist_ok=True)
os.makedirs("/home/local/AD/cordun1/experiments/TFC-GAN/saved_models/%s" % opt.experiment, exist_ok=True)

cuda = True if torch.cuda.is_available() else False
torch.cuda.set_device(opt.gpu_num) 

# Set fixed random number seed
torch.manual_seed(42)
                    
                    
#################
# Loss functions
#################
                    
"""BicycleGAN"""
bic_pixel_loss = torch.nn.L1Loss()
bic_thermal_loss = torch.nn.L1Loss()
vae_loss = torch.nn.MSELoss()
kl_loss = nn.KLDivLoss(reduction="batchmean")

                    
""" pix2pix """
criterion_GAN = torch.nn.MSELoss()
criterion_pixelwise = torch.nn.L1Loss()
# Loss weight of L1 pixel-wise loss between translated image and real image
lambda_pixel = 100                    
                    
                    
##############################
#  G1: BicycleGAN
##############################        

class UNetDown_bic(nn.Module):
    def __init__(self, in_size, out_size, normalize=True, dropout=0.0):
        super(UNetDown_bic, self).__init__()
        layers = [nn.Conv2d(in_size, out_size, 3, stride=2, padding=1, bias=False)]
        if normalize:
            layers.append(nn.BatchNorm2d(out_size, 0.8))
        layers.append(nn.LeakyReLU(0.2))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UNetUp_bic(nn.Module):
    def __init__(self, in_size, out_size):
        super(UNetUp_bic, self).__init__()
        self.model = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_size, out_size, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_size, 0.8),
            #nn.ReLU(inplace=True),
            nn.LeakyReLU(inplace=True) # will it help with vanishing gradients?
        )

    def forward(self, x, skip_input):
        x = self.model(x)
        x = torch.cat((x, skip_input), 1)
        return x

# ThermalGAN paper says it added an additional down and up module to increase output resolution
# For now, I'm going to leave it and see what happens with the intact/original UNET for both pix2pix and BicycleGAN
                    
class Generator_G1(nn.Module):
    def __init__(self, img_shape):
        super(Generator_G1, self).__init__()
        channels, self.h, self.w = img_shape
                    
        self.fc = nn.Linear(1, self.h * self.w) # [1, 256*256]
               
        self.down1 = UNetDown_bic(channels + 1, 64, normalize=False) #takes A and T
        self.down2 = UNetDown_bic(64, 128)
        self.down3 = UNetDown_bic(128, 256)
        self.down4 = UNetDown_bic(256, 512)
        self.down5 = UNetDown_bic(512, 512)
        self.down6 = UNetDown_bic(512, 512)
        self.down7 = UNetDown_bic(512, 512, normalize=False)
        self.up1 = UNetUp_bic(512, 512)
        self.up2 = UNetUp_bic(1024, 512)
        self.up3 = UNetUp_bic(1024, 512)
        self.up4 = UNetUp_bic(1024, 256)
        self.up5 = UNetUp_bic(512, 128)
        self.up6 = UNetUp_bic(256, 64)

        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2), 
            nn.Conv2d(128, channels, 3, stride=1, padding=1), nn.Tanh()
        )

    def forward(self, x, z):
        with autocast():
            t = z.reshape(z.size(0), 1, z.size(1), z.size(2))
 
            # x: #[32, 3, 256, 256]
            # t needs to be [32, 1, 256, 256]
            d1 = self.down1(torch.cat((x, t), 1))
            d2 = self.down2(d1)
            d3 = self.down3(d2)
            d4 = self.down4(d3)
            d5 = self.down5(d4)
            d6 = self.down6(d5)
            d7 = self.down7(d6)
            u1 = self.up1(d7, d6)
            u2 = self.up2(u1, d5)
            u3 = self.up3(u2, d4)
            u4 = self.up4(u3, d3)
            u5 = self.up5(u4, d2)
            u6 = self.up6(u5, d1)
            output = self.final(u6).type(HalfTensor)
                    
        return output


# You still need this for the KL divergence loss
class Encoder(nn.Module):
    def __init__(self, latent_dim): #(8, 64, 256, 256)
        super(Encoder, self).__init__()
        
        resnet18_model = resnet18(pretrained=False)
        self.feature_extractor = nn.Sequential(*list(resnet18_model.children())[:-3])
        self.pooling = nn.AvgPool2d(kernel_size=8, stride=8, padding=0)
        #self.fc_mu = nn.Linear(256, latent_dim) #(256,8)
        #self.fc_logvar = nn.Linear(256, latent_dim) #(256,8)
        self.fc_mu = nn.Linear(1024, latent_dim) #(256,8)
        self.fc_logvar = nn.Linear(1024, latent_dim) #(256,8)

    def forward(self, img):
        """
        out: torch.Size([32, 256, 16, 16])
        a: torch.Size([32, 256, 2, 2])
        b: torch.Size([32, 1024])
        mu: torch.Size([32, 8])
        logvar: torch.Size([32, 8])
        """
        with autocast():
            out = self.feature_extractor(img).type(HalfTensor)  
            #print("out:", out.size()) # torch.Size([32, 256, 16, 16])
            
            a = self.pooling(out).type(HalfTensor) 
            #print("a:", a.size()) #  torch.Size([32, 256, 2, 2])
            
            b = a.view(a.size(0), -1).type(HalfTensor) 
            #print("b:", b.size()) # 32, 1024
            
            # RuntimeError: mat1 and mat2 shapes cannot be multiplied (32x1024 and 256x8)
            mu = self.fc_mu(b).cuda() # 32,8 = (256 x 8) x (8 x 32)
            #print("mu:", mu.size())
                  
            logvar = self.fc_logvar(b).cuda() # 32,8
            #print("logvar:", logvar.size())
            
        return mu.type(HalfTensor), logvar.type(HalfTensor)


    
class MultiDiscriminator(nn.Module):
    # was leading to numerical instability, so changed it to the pix2pix D
    
    def __init__(self, img_shape):
        super(MultiDiscriminator, self).__init__()
        
        channels, self.h, self.w = img_shape

        def discriminator_block(in_filters, out_filters, normalization=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalization:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(channels, 64, normalization=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.Conv2d(512, 1, 4, padding=1, bias=False)
        )

    def forward(self, img):
        with autocast():
            out = self.model(img).type(HalfTensor)
            print("Multidiscriminator out:", out.size())
                    
        return out
    



##############################
#  G2: pix2pix
##############################
# Calculate output of image discriminator (PatchGAN)
patch = (1, opt.img_height // 2 ** 4, opt.img_width // 2 ** 4) 
                    
                    
class UNetDown_pix(nn.Module):
    def __init__(self, in_size, out_size, normalize=True, dropout=0.0):
        super(UNetDown_pix, self).__init__()
        layers = [nn.Conv2d(in_size, out_size, 4, 2, 1, bias=False)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_size))
        layers.append(nn.LeakyReLU(0.2))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UNetUp_pix(nn.Module):
    def __init__(self, in_size, out_size, dropout=0.0):
        super(UNetUp_pix, self).__init__()
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

# ThermalGAN paper says it added an additional down and up module to increase output resolution
# For now, I'm going to leave it and see what happens with the intact/original UNET for both pix2pix and BicycleGAN

class GeneratorUNet_G2(nn.Module):
    def __init__(self, img_shape):
        super(GeneratorUNet_G2, self).__init__()
        channels, self.h, self.w = img_shape
                    
        self.down1 = UNetDown_pix(channels, 64, normalize=False) # don't know what S^ is size
        self.down2 = UNetDown_pix(64, 128)
        self.down3 = UNetDown_pix(128, 256)
        self.down4 = UNetDown_pix(256, 512, dropout=0.5)
        self.down5 = UNetDown_pix(512, 512, dropout=0.5)
        self.down6 = UNetDown_pix(512, 512, dropout=0.5)
        self.down7 = UNetDown_pix(512, 512, dropout=0.5)
        self.down8 = UNetDown_pix(512, 512, normalize=False, dropout=0.5)
        self.up1 = UNetUp_pix(512, 512, dropout=0.5)
        self.up2 = UNetUp_pix(1024, 512, dropout=0.5)
        self.up3 = UNetUp_pix(1024, 512, dropout=0.5)
        self.up4 = UNetUp_pix(1024, 512, dropout=0.5)
        self.up5 = UNetUp_pix(1024, 256)
        self.up6 = UNetUp_pix(512, 128)
        self.up7 = UNetUp_pix(256, 64)

        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(128, channels, 4, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        #print("S^ input size to G2:", x.size())
        with autocast():
            with torch.autograd.set_detect_anomaly(True):
                # U-Net generator with skip connections from encoder to decoder
                d1 = self.down1(x)
                d2 = self.down2(d1)
                d3 = self.down3(d2)
                d4 = self.down4(d3)
                d5 = self.down5(d4)
                d6 = self.down6(d5)
                d7 = self.down7(d6)
                d8 = self.down8(d7)
                u1 = self.up1(d8, d7)
                u2 = self.up2(u1, d6)
                u3 = self.up3(u2, d5)
                u4 = self.up4(u3, d4)
                u5 = self.up5(u4, d3)
                u6 = self.up6(u5, d2)
                u7 = self.up7(u6, d1)
                output = self.final(u7).type(HalfTensor)
                    
        return output


class Discriminator_pix(nn.Module):
    def __init__(self, img_shape):
        super(Discriminator_pix, self).__init__()
        
        channels, self.h, self.w = img_shape

        def discriminator_block(in_filters, out_filters, normalization=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalization:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(channels * 2, 64, normalization=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1, bias=False)
        )

    def forward(self, img_A, img_B):
        with autocast():
            img_input = torch.cat((img_A, img_B), 1)
            out = self.model(img_input).type(HalfTensor)
                    
        return out

                      
##############################
#       Utils
##############################

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)
          

def sample_images(batches_done):
    imgs = next(iter(test_dataloader))
    real_A = Variable(imgs["A"].type(HalfTensor))
    real_B = Variable(imgs["B"].type(HalfTensor))
    TB = Variable(imgs["T_B"].type(HalfTensor)) 
    #TB = TB.reshape(TB.size(0), 1, TB.size(1), TB.size(2))
    print("...Sampling...")
    print("...real_A:", real_A.size())
    print("...TB:", TB.size())
    
    fake_S = generator_G1(real_A, TB)
    fake_B = generator_G2(fake_S)
    real_S = enlarge_and_bitwise_thermal(real_B) # this may encounter some problems - I need to scale for batch = 1
    
    # GLOBAL
    img_sample_global = torch.cat((real_A.data, fake_S.data, fake_B, real_B.data), -2)
    save_image(img_sample_global, "images/%s/%s_g.png" % (opt.experiment, batches_done), nrow=5, normalize=True)
             
                    
##############################
# My Custom Additions
##############################
                    
# for passing to vectorizer for fake_B temperature calculation
T = np.linspace(24, 38, num=256) # 0 - 255 indices of temperatures in Celsius
d = dict(enumerate((T).flatten(), 0)) # dictionary like {0: 24.0, 1: 24.054901960784314, etc.}

def vectorize_temps(fake_B):
    TFB = []
    for t in range(0, opt.batch_size):
        b = transforms.ToPILImage()(fake_B[t, :, :, :]).convert("RGB")
        fake_vectorizer = TempVector_PyTorch(b, d) # calls datasets_temp .detach().cpu()
        tfb = torch.Tensor(fake_vectorizer.make_pixel_vectors()).cuda()
        TFB.append(tfb)
    
    TFB_tensor = torch.cat(TFB).reshape(opt.batch_size, 1, opt.img_height, opt.img_width)     
    return TFB_tensor

                    
"""Because I do not have segments, I will use binary thresh + bitwise
Do this for the real and fake thermal images to substitute as S """
def enlarge_and_bitwise_thermal(thermal_tensor): 
    masks = []
    
    for m in range(0, thermal_tensor.size(0)): # use face_tensor.size(0) b/c batch size is spread over multiple GPUs  
        ther = thermal_tensor[m, :, :, :].data.view(256,256,3) # get the data in cv2 format
        thr_arr = ther.detach().cpu().numpy() #turn into array 
        thr_arr = thr_arr.astype(np.float32) # cast to fp32

        # ensure dtype is the same as array going in
        t_gray = cv2.cvtColor(thr_arr, cv2.COLOR_BGR2GRAY) # convert to grayscale
        invert = cv2.bitwise_not(t_gray)
        masks.append(invert)
                    
    # convert list of arrays to torch.tensor
    mask_tensor = torch.Tensor(masks)
    # must be 3D instead of 1D [batch, 256, 256] -> [batch, 3, 256, 256]
    targ = [mask_tensor.size(0), 3, mask_tensor.size(1), mask_tensor.size(1)]
    mask_tensor_3D = mask_tensor[:, None, :, :].expand(targ).cuda()
    #must normalize, otherwise values are too big and will lead to -inf, i think? 
    n = F.normalize(mask_tensor_3D, p=2.0, dim=1, eps=1e-12)
    
    print("--before norm-")
    print(mask_tensor_3D[0])
    print("--after norm--")
    print(n[0])
    
    return n
    
    
##############################
#       Initialize
##############################
input_shape = (opt.channels, opt.img_height, opt.img_width)
                    
"""BicycleGAN"""
# Initialize generator, encoder and discriminators
generator_G1 = Generator_G1(input_shape)
encoder = Encoder(opt.latent_dim)
D_VAE = MultiDiscriminator(input_shape)

generator_G1 = generator_G1.cuda()
encoder.cuda()
D_VAE = D_VAE.cuda()     
bic_pixel_loss.cuda()
bic_thermal_loss.cuda()
kl_loss.cuda()

vae_loss.cuda()
                    
"""pix2pix"""
# Initialize generator and discriminator
generator_G2 = GeneratorUNet_G2(input_shape)
discriminator_pix = Discriminator_pix(input_shape)

generator_G2 = generator_G2.cuda()
discriminator_pix = discriminator_pix.cuda()
criterion_GAN.cuda()
criterion_pixelwise.cuda()

################    
# nn.DataParallel
################
        
# BicycleGAN
generator_G1 = torch.nn.DataParallel(generator_G1, device_ids=[0,1])
encoder = torch.nn.DataParallel(encoder, device_ids=[0,1])
D_VAE = torch.nn.DataParallel(D_VAE, device_ids=[0,1])
                    
#pix2pix
generator_G2 = torch.nn.DataParallel(generator_G2, device_ids=[0,1])
discriminator_pix = torch.nn.DataParallel(discriminator_pix, device_ids=[0,1])

# Don't forget to enter the right n ames
if opt.epoch != 0:
    # BicycleGAN
    generator_G1.load_state_dict(torch.load("/home/local/AD/cordun1/experiments/TFC-GAN/saved_models/%s/generator_G1_%d.pth" % (opt.experiment, opt.epoch)))
    encoder.load_state_dict(torch.load("/home/local/AD/cordun1/experiments/TFC-GAN/saved_models/%s/encoder_%d.pth" % (opt.experiment, opt.epoch)))
    D_VAE.load_state_dict(torch.load("/home/local/AD/cordun1/experiments/TFC-GAN/saved_models/%s/D_VAE_%d.pth" % (opt.experiment, opt.epoch)))
   
    # pix2pix
    generator_G2.load_state_dict(torch.load("/home/local/AD/cordun1/experiments/TFC-GAN/saved_models/%s/generator_G2_%d.pth" % (opt.experiment, opt.epoch)))
    discriminator_pix.load_state_dict(torch.load("/home/local/AD/cordun1/experiments/TFC-GAN/saved_models/%s/discriminator_pix_%d.pth" % (opt.experiment, opt.epoch)))

else:
    # Initialize weights
    # BicycleGAN
    generator_G1.apply(weights_init_normal)
    encoder.apply(weights_init_normal)
    D_VAE.apply(weights_init_normal)
                  
    #pix2pix
    generator_G2.apply(weights_init_normal)
    discriminator_pix.apply(weights_init_normal)
                    
################    
# Optimizers
################
# ThermalGAN paper doesn't mention anything about jointly optimizing, so I keep them separate

"""BicycleGAN"""
# Optimizers
optimizer_G1 = torch.optim.Adam(generator_G1.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_E = torch.optim.Adam(encoder.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D_VAE = torch.optim.Adam(D_VAE.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
                 
"""pix2pix"""     
optimizer_G2 = torch.optim.Adam(generator_G2.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D_PIX = torch.optim.Adam(discriminator_pix.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
                    
                    
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

# Tensor type - only use HalfTensor in this AMP script
HalfTensor = torch.cuda.HalfTensor if cuda else torch.HalfTensor
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor                   
                                      
##############################
#       Training
##############################

prev_time = time.time()
f = open('/home/local/AD/cordun1/experiments/TFC-GAN/LOGS/{}.txt'.format(opt.out_file), 'a+')

# Try AMP
scaler = GradScaler()
                    
for epoch in range(opt.epoch, opt.n_epochs):
    
    for i, batch in enumerate(dataloader):
        print("-------{}-------".format(i))
        
        real_A = Variable(batch["A"].type(HalfTensor))
        real_B = Variable(batch["B"].type(HalfTensor))
        TB = Variable(batch["T_B"].type(HalfTensor), requires_grad=True) # I will use my own temperature vector for this 

        # Adversarial ground truth: BicycleGAN
        valid_G1 = torch.ones(real_A.size(0), 1, 15, 15).type(HalfTensor).cuda()
        fake_G1 = torch.zeros(real_A.size(0), 1, 15, 15).type(HalfTensor).cuda()
        
        #valid_G1 = Variable(HalfTensor(np.ones((real_A.size(0), *patch))), requires_grad=False)
                             
        # Adversarial ground truths: pix2pix - keep these separate from BicyleGAN that computes it in a different way
        valid_G2 = Variable(HalfTensor(np.ones((real_A.size(0), *patch))), requires_grad=False)
        fake_G2 = Variable(HalfTensor(np.zeros((real_A.size(0), *patch))), requires_grad=False)
                    
        """ 
        Stage 1: BicycleGAN - G1 
        Goal: Output S^
        """
        
        # -------------------------------
        #  Train Generator and Encoder
        # -------------------------------
        print("+ + + optimizer_E.zero_grad() + + +")
        optimizer_E.zero_grad()
        print("+ + + optimizer_G1.zero_grad() + + +")
        optimizer_G1.zero_grad()
                    
        with autocast(): 
            with torch.autograd.set_detect_anomaly(True):
                    
                # Produce output using encoding of B (cVAE-GAN)
                mu, logvar = encoder(real_B) # you still need these for the KL loss; ThermalGAN paper does not specify if it's B or S

                # Generate the fake segmentation mask        
                fake_S = generator_G1(real_A, TB)
                print("fake_S:", fake_S[0,:,:,:])

                # Pixelwise loss of translated image by VAE
                real_S = enlarge_and_bitwise_thermal(real_B)
                print("real_S size:", real_S.size())
                print("real_S:", real_S[0,:,:,:])
                
                loss_pixel_bic = bic_pixel_loss(fake_S, real_S)
 
                # Kullback-Leibler divergence of encoded B
                loss_kl = kl_loss(mu, logvar)

                D1_out = D_VAE(fake_S) # MSE loss
                print("----D1_out:", D1_out[0])
                loss_VAE_GAN = vae_loss(D1_out, valid_G1)

                TFB_ = vectorize_temps(fake_S).requires_grad_(requires_grad=True) # already [64, 1, 256, 256]
                TB = TB.reshape(TB.size(0), 1, TB.size(1), TB.size(2))
                loss_latent = bic_thermal_loss(TB, TFB_)
    
                # total G1 loss
                loss_GE = loss_VAE_GAN + opt.lambda_kl * loss_kl + opt.lambda_pixel * loss_pixel_bic + loss_latent

        # Not sure if ok for me to combine together 
        scaler.scale(loss_GE).backward(retain_graph=True) 
        scaler.step(optimizer_G1)
        scaler.step(optimizer_E)

        # ----------------------------------
        #  Train Discriminator (cVAE-GAN)
        # ----------------------------------

        print("+ + + optimizer_D_VAE.zero_grad() + + +")
        optimizer_D_VAE.zero_grad()
        with autocast():  
            vae_r = D_VAE(real_S)
            print("vae_r:", vae_r[0]) # these have -inf values
            """
            vae_r: tensor([[[-inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf],
             [-inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf],
             [-inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf],
             [-inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf],
             [-inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf],
             [-inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf],
             [-inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf],
             [-inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf],
             [-inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf],
             [-inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf],
             [-inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf],
             [-inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf],
             [-inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf],
             [-inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf],
             [-inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf]]],
           device='cuda:0', dtype=torch.float16, grad_fn=<SelectBackward>)

            I think this is happening because real_S values are very big: 
               tensor([[   5.0413,    5.0735,    5.2030,  ...,    5.2468,    5.2539,
                5.2539],
            [   5.1102,    5.0297,    5.2030,  ...,    5.1985,    5.2726,
                5.2610],
            [   5.1038,    5.0368,    5.1289,  ...,    5.1985,    5.2726,
                5.2610],
            ...,
            [   6.0142,    5.7843,    6.5531,  ...,    7.3919,    6.8198,
                6.4556],
            [   5.8454,    5.9684,    6.5344,  ...,   14.5705,    8.4611,
                7.1201],
            [   5.7578,    5.8506,    6.4258,  ...,  -75.6985, -109.7541,
               29.9986]])

           """
            vae_real = vae_loss(vae_r, valid_G1)
            
            vae_f = D_VAE(fake_S.detach())
            print("vae_f:", vae_f[0])
            vae_fake = vae_loss(vae_f, fake_G1)
            
            loss_D_VAE = vae_real + vae_fake
            print("===loss_D_VAE:", loss_D_VAE)

        scaler.scale(loss_D_VAE).backward() # retain_graph=True
        scaler.step(optimizer_D_VAE)
        print("+ + + optimizer_D_VAE.step() + + +")


        # ---------------------------------
        #  Train Generator
        # ---------------------------------
        print("+ + + optimizer_G2.zero_grad() + + +")
        optimizer_G2.zero_grad()         
        with autocast():
            
            with torch.autograd.set_detect_anomaly(True):
                    
                fake_B = generator_G2(fake_S.detach())
                print("fake_B:", fake_B.size())

                pred_fake = discriminator_pix(fake_B, real_A)
                print("pred_fake:", pred_fake.size())

                loss_GAN = criterion_GAN(pred_fake, valid_G2)
                print("loss_GAN:", loss_GAN)

                loss_pixel_pix = criterion_pixelwise(fake_B, real_B)
                print("loss_pixel_pix:", loss_pixel_pix)

                loss_G = loss_GAN + opt.lambda_pixel * loss_pixel_pix
       
        """Note: Because all of BicycleGAN has to kept, since .step will remove vars, 
        the previous losses were set to retain_graph=True, otherwise they are discarded
        and this loss cannot backprop"""
        
        scaler.scale(loss_G).backward(retain_graph=True)
        scaler.step(optimizer_G2)
        print("+ + + optimizer_G2.step() + + +")

        # ---------------------------------
        #  Train Discriminator
        # ---------------------------------
        print("+ + + optimizer_D_PIX.zero_grad() + + +")
        optimizer_D_PIX.zero_grad()
        with autocast():

            # Real loss
            pred_real_d = discriminator_pix(real_B, real_A)
            loss_real_d = criterion_GAN(pred_real_d, valid_G2.clone())

            # Fake loss
            pred_fake_d = discriminator_pix(fake_B.detach(), real_A)
            loss_fake_d = criterion_GAN(pred_fake_d, fake_G2.clone())

            # Total loss
            loss_D = 0.5 * (loss_real_d + loss_fake_d)

        scaler.scale(loss_D).backward()
        scaler.step(optimizer_D_PIX)
                    
                    
        #######################
        # Scaler Updated - Autograd()
        #######################
                    
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
            "\r[Epoch %d/%d] [Batch %d/%d] [loss_GE: %f, loss_latent: %f, loss_D_VAE: %f, loss_G: %f, loss_D: %f] ETA: %s"
            % (
                epoch, #%d
                opt.n_epochs, #%d
                i, #%d
                len(dataloader), #%d
                loss_GE.item(), #%f
                loss_latent.item(),
                loss_D_VAE.item(), #%f 
                loss_G.item(), #%f - adv G loss
                loss_D.item(),
                time_left, #%s
            )
        )
        
        f.write(
            "\r[Epoch %d/%d] [Batch %d/%d] [loss_GE: %f, loss_latent: %f, loss_D_VAE: %f, loss_G: %f, loss_D: %f] ETA: %s"
            % (
                epoch, #%d
                opt.n_epochs, #%d
                i, #%d
                len(dataloader), #%d
                loss_GE.item(), #%f
                loss_latent.item(),
                loss_D_VAE.item(), #%f 
                loss_G.item(), #%f - adv G loss
                loss_D.item(),
                time_left, #%s
            )
        )
        
        # If at sample interval save image
        if batches_done % opt.sample_interval == 0:
            sample_images(batches_done)

    if opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0:
        torch.save(generator_G1.state_dict(), "/home/local/AD/cordun1/experiments/TFC-GAN/saved_models/%s/generator_G1_%d.pth" % (opt.experiment, opt.epoch))
        torch.save(encoder.state_dict(), "/home/local/AD/cordun1/experiments/TFC-GAN/saved_models/%s/encoder_%d.pth" % (opt.experiment, opt.epoch))
        torch.save(D_VAE.state_dict(), "/home/local/AD/cordun1/experiments/TFC-GAN/saved_models/%s/D_VAE_%d.pth" % (opt.experiment, opt.epoch))
        #torch.save(D_LR.state_dict(), "/home/local/AD/cordun1/experiments/TFC-GAN/saved_models/%s/D_LR_%d.pth" % (opt.experiment, opt.epoch))
        torch.save(generator_G2.state_dict(), "/home/local/AD/cordun1/experiments/TFC-GAN/saved_models/%s/generator_G2_%d.pth" % (opt.experiment, epoch))
        torch.save(discriminator_pix.state_dict(), "/home/local/AD/cordun1/experiments/TFC-GAN/saved_models/%s/discriminator_pix_%d.pth" % (opt.experiment, epoch))
    