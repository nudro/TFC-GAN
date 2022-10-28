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
#from torch.cuda.amp import GradScaler, autocast
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
parser.add_argument("--sample_interval", type=int, default=100, help="interval between sampling of images from generators")
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

my_loss = torch.nn.L1Loss()
my_loss.cuda()
                    
                    
##############################
#  G1: BicycleGAN
##############################        

class UNetDown_bic(nn.Module):
    def __init__(self, in_size, out_size, normalize=True, dropout=0.0):
        super(UNetDown_bic, self).__init__()
        layers = [nn.Conv2d(in_size, out_size, 3, stride=2, padding=1, bias=False)]
        if normalize:
            layers.append(nn.BatchNorm2d(out_size)) # was Batch
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
            nn.BatchNorm2d(out_size),
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
               
        self.down1 = UNetDown_bic(channels + 1, 64, normalize=False) #takes A and T; normalize=False
        self.down2 = UNetDown_bic(64, 128)
        self.down3 = UNetDown_bic(128, 256)
        self.down4 = UNetDown_bic(256, 512)
        self.down5 = UNetDown_bic(512, 512)
        self.down6 = UNetDown_bic(512, 512)
        self.down7 = UNetDown_bic(512, 512)
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
        t = z.reshape(z.size(0), 1, z.size(1), z.size(2))
        print("t:", t[0, 0, :5, :5])
        print("x which is A:", x[0,0,:5,:5])
        # x: #[32, 3, 256, 256]
        # t needs to be [32, 1, 256, 256]
        d1 = self.down1(torch.cat((x, t), 1))
        print("d1:", d1[0, 0, :5, :5])
        d2 = self.down2(d1)
        print("d2:", d2[0, 0, :5, :5])
        d3 = self.down3(d2)
        print("d3:", d3[0, 0, :5, :5])
        d4 = self.down4(d3)
        print("d4:", d4[0, 0, :5, :5])
        d5 = self.down5(d4)
        print("d5:", d5[0, 0, :5, :5])
        d6 = self.down6(d5)
        print("d6:", d6[0, 0, :5, :5])
        d7 = self.down7(d6)
        print("d7:", d7[0, 0, :5, :5])
        u1 = self.up1(d7, d6)
        u2 = self.up2(u1, d5)
        u3 = self.up3(u2, d4)
        u4 = self.up4(u3, d3)
        u5 = self.up5(u4, d2)
        u6 = self.up6(u5, d1)
        print("u6:", u6[0, 0, :5, :5])

        output = self.final(u6)
        return output


# You still need this for the KL divergence loss
class Encoder(nn.Module):
    def __init__(self, latent_dim): #(8, 64, 256, 256)
        super(Encoder, self).__init__()
        
        resnet18_model = resnet18(pretrained=False)
        self.feature_extractor = nn.Sequential(*list(resnet18_model.children())[:-3])
        self.pooling = nn.AvgPool2d(kernel_size=8, stride=8, padding=0)
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

        out = self.feature_extractor(img)
        a = self.pooling(out)    
        b = a.view(a.size(0), -1)  
        # RuntimeError: mat1 and mat2 shapes cannot be multiplied (32x1024 and 256x8)
        mu = self.fc_mu(b).cuda() # 32,8 = (256 x 8) x (8 x 32)      
        logvar = self.fc_logvar(b).cuda() # 32,8

        return mu, logvar

class MultiDiscriminator(nn.Module):
    def __init__(self, img_shape):
        super(MultiDiscriminator, self).__init__()
        channels, self.h, self.w = img_shape

        def discriminator_block(in_filters, out_filters, normalize=True):
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters)) # Batch
            layers.append(nn.LeakyReLU(0.2))
            return layers

        # Extracts discriminator models
        self.models = nn.ModuleList()
        for i in range(3):
            self.models.add_module(
                "disc_%d" % i,
                nn.Sequential(
                    *discriminator_block(channels, 64, normalize=False),
                    *discriminator_block(64, 128),
                    *discriminator_block(128, 256),
                    *discriminator_block(256, 512),
                    nn.Conv2d(512, 1, 3, padding=1)
                ),
            )

        self.downsample = nn.AvgPool2d(channels, stride=2, padding=[1, 1], count_include_pad=False)

    def forward(self, x, gt):

        outputs = []
        for m in self.models:
            outputs.append(m(x))
            x = self.downsample(x)

        a = []
        for out in outputs:
            if gt==1: 
                gt_ = torch.ones(out.size(0), 1, out.size(2), out.size(3)).cuda()
            else:
                gt_ = torch.zeros(out.size(0), 1, out.size(2), out.size(3)).cuda()

            alt = my_loss(out,gt_)
            a.append(alt)
        
        at = (torch.tensor(a)).mean().requires_grad_(requires_grad=True).cuda()

        return at 

                          
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
        output = self.final(u7) 
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
        img_input = torch.cat((img_A, img_B), 1)
        out = self.model(img_input)             
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
          

# STOP - ALL THIS NEEDS TO BE FIXED FOR SAMPLING
def sample_images(batches_done):
    imgs = next(iter(test_dataloader))
    real_A = Variable(imgs["A"]).type(Tensor)
    real_B = Variable(imgs["B"]).type(Tensor)
    TB = Variable(imgs["T_B"]).type(Tensor)

    fake_S = generator_G1(real_A, TB).cuda()
    fake_B = generator_G2(fake_S).cuda() # this is actually R
    real_S = enlarge_and_bitwise_thermal(real_B).cuda()
    
    # Recall that fake_B is really S + R, and that you have to add it together

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
    normed = F.normalize(TFB_tensor, p=2.0, dim=2, eps=1e-12) # will this improve instablity? norm'd by height
    return normed

                    

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
    targ = [mask_tensor.size(0), 3, mask_tensor.size(1), mask_tensor.size(1)]
    mask_tensor_3D = mask_tensor[:, None, :, :].expand(targ).cuda()
    # must normalize otherwise leads to -inf values
    n = F.normalize(mask_tensor_3D, p=2.0, dim=2, eps=1e-12) 
    
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
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor                   
                                      
##############################
#       Training
##############################

prev_time = time.time()
f = open('/home/local/AD/cordun1/experiments/TFC-GAN/LOGS/{}.txt'.format(opt.out_file), 'a+')
                    
for epoch in range(opt.epoch, opt.n_epochs):
    for i, batch in enumerate(dataloader):
        real_A = Variable(batch["A"].type(Tensor)) # changed from HalfTensor back to Tensor
        real_B = Variable(batch["B"].type(Tensor))
        TB = Variable(batch["T_B"].type(Tensor)) # I will use my own temperature vector for this 
        TBn = F.normalize(TB, p=2.0, dim=1, eps=1e-12)
        
        # Adversarial ground truth: BicycleGAN
        valid_G1 = 1
        fake_G1 = 0
                             
        # Adversarial ground truths: pix2pix - keep these separate from BicyleGAN that computes it in a different way
        valid_G2 = Variable(Tensor(np.ones((real_A.size(0), *patch))), requires_grad=False)
        fake_G2 = Variable(Tensor(np.zeros((real_A.size(0), *patch))), requires_grad=False)
                        
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
         
        # Produce output using encoding of B (cVAE-GAN)
        mu, logvar = encoder(real_B) # you still need these for the KL loss; ThermalGAN paper does not specify if it's B or S

        # Generate the fake segmentation mask   
        #A_n = F.normalize(real_A, p=2.0, dim=2, eps=1e-12)
        fake_S = generator_G1(real_A, TBn) #(tensor,tensor)

        # Pixelwise loss of translated image by VAE
        real_S = enlarge_and_bitwise_thermal(real_B)
        loss_pixel_bic = bic_pixel_loss(fake_S, real_S)

        # Kullback-Leibler divergence of encoded B
        loss_kl = kl_loss(mu, logvar)

        loss_VAE_GAN = (D_VAE(fake_S, valid_G1)).mean()  
        # average across batches, since there are 2 separate batches by 2 GPUs in parallel 

        #TFB_ = vectorize_temps(fake_S).requires_grad_(requires_grad=True) # already [64, 1, 256, 256]
        TFB_ = vectorize_temps(fake_S)
        TBn = TBn.reshape(TBn.size(0), 1, TBn.size(1), TBn.size(2))
        loss_latent = bic_thermal_loss(TBn, TFB_)

        # total G1 loss
        loss_GE = loss_VAE_GAN + opt.lambda_kl * loss_kl + opt.lambda_pixel * loss_pixel_bic + loss_latent
                         
        loss_GE.backward(retain_graph=True)
        optimizer_G1.step()
        optimizer_E.step()
        print("====optimizer_G1.step()====")
        print("====optimizer_E.step()====")


        # ----------------------------------
        #  Train Discriminator (cVAE-GAN)
        # ----------------------------------

        print("+ + + optimizer_D_VAE.zero_grad() + + +")
        optimizer_D_VAE.zero_grad()

        vae_real = (D_VAE(real_S, valid_G1)).mean()
        vae_fake = (D_VAE(fake_S.detach(), fake_G1)).mean()
        loss_D_VAE = (vae_real + vae_fake).requires_grad_(requires_grad=True)

        loss_D_VAE.backward(retain_graph=True)
        optimizer_D_VAE.step()

        print("====optimizer_D_VAE.step()====")



        # ---------------------------------
        #  Train Generator
        # ---------------------------------
        print("+ + + optimizer_G2.zero_grad() + + +")
        optimizer_G2.zero_grad()         
   
        fake_B = generator_G2(fake_S.detach())
        pred_fake = discriminator_pix(fake_B, real_A)
        loss_GAN = criterion_GAN(pred_fake, valid_G2)
        loss_pixel_pix = criterion_pixelwise(fake_B, real_B)
        loss_G = loss_GAN + opt.lambda_pixel * loss_pixel_pix
       
        """Note: Because all of BicycleGAN has to kept, since .step will remove vars, 
        the previous losses were set to retain_graph=True, otherwise they are discarded
        and this loss cannot backprop"""
        
        loss_G.backward(retain_graph=True)
        optimizer_G2.step()
        print("+ + + optimizer_G2.step() + + +")

        # ---------------------------------
        #  Train Discriminator
        # ---------------------------------
        print("+ + + optimizer_D_PIX.zero_grad() + + +")
        optimizer_D_PIX.zero_grad()
        # Real loss
        pred_real_d = discriminator_pix(real_B, real_A)
        loss_real_d = criterion_GAN(pred_real_d, valid_G2)

        # Fake loss
        pred_fake_d = discriminator_pix(fake_B.detach(), real_A)
        loss_fake_d = criterion_GAN(pred_fake_d, fake_G2)

        # Total loss
        loss_D = 0.5 * (loss_real_d + loss_fake_d)

        loss_D.backward(retain_graph=True)
        optimizer_D_PIX.step()
       
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
            "\r[Epoch %d/%d] [Batch %d/%d] [loss_VAE_GAN: %f, loss_D_VAE: %f, loss_kl: %f, loss_pixel_bic: %f, loss_latent: %f, loss_GAN: %f, loss_pixel_pix: %f, loss_G: %f, loss_D: %f, loss_real_d: %f, loss_fake_d: %f] ETA: %s"
            % (
                epoch, #%d
                opt.n_epochs, #%d
                i, #%d
                len(dataloader), #%d
                loss_VAE_GAN.item(), #%f
                loss_D_VAE.item(),
                loss_kl.item(),
                loss_pixel_bic.item(), #%f 
                loss_latent.item(),
                loss_GAN.item(),
                loss_pixel_pix.item(),
                loss_G.item(), #%f - adv G loss
                loss_D.item(),
                loss_real_d.item(),
                loss_fake_d.item(),
                time_left, #%s
            )
        )
        # loss_VAE_GAN + opt.lambda_kl * loss_kl + opt.lambda_pixel * loss_pixel_bic + loss_latent
        #  loss_G = loss_GAN + opt.lambda_pixel * loss_pixel_pix
        #  loss_D = 0.5 * (loss_real_d + loss_fake_d)
        
        f.write(
            "\r[Epoch %d/%d] [Batch %d/%d] [loss_VAE_GAN: %f, loss_D_VAE: %f, loss_kl: %f, loss_pixel_bic: %f, loss_latent: %f, loss_GAN: %f, loss_pixel_pix: %f, loss_G: %f, loss_D: %f, loss_real_d: %f, loss_fake_d: %f] ETA: %s"
            % (
                epoch, #%d
                opt.n_epochs, #%d
                i, #%d
                len(dataloader), #%d
                loss_VAE_GAN.item(), #%f
                loss_D_VAE.item(),
                loss_kl.item(),
                loss_pixel_bic.item(), #%f 
                loss_latent.item(),
                loss_GAN.item(),
                loss_pixel_pix.item(),
                loss_G.item(), #%f - adv G loss
                loss_D.item(),
                loss_real_d.item(),
                loss_fake_d.item(),
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
        torch.save(generator_G2.state_dict(), "/home/local/AD/cordun1/experiments/TFC-GAN/saved_models/%s/generator_G2_%d.pth" % (opt.experiment, epoch))
        torch.save(discriminator_pix.state_dict(), "/home/local/AD/cordun1/experiments/TFC-GAN/saved_models/%s/discriminator_pix_%d.pth" % (opt.experiment, epoch))
    