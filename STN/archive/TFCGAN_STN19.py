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
import mediapipe as mp
from protobuf_to_dict import protobuf_to_dict 
from datasets_stn import * # only A and B
import kornia
from lpips_pytorch import LPIPS, lpips
import torch.nn as nn
import kornia.contrib as K
from medpy.filter.smoothing import anisotropic_diffusion

"""

Follows Arar using a ViT. 
The results at 30 epochs are very, very sheared visible images. 

"""

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
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=50, help="interval between sampling of images from generators")
parser.add_argument("--checkpoint_interval", type=int, default=50, help="interval between model checkpoints")
parser.add_argument("--gpu_num", type=int, default=0, help="gpu card")
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
criterion_GAN = torch.nn.BCEWithLogitsLoss() # Relativitic
criterion_lpips = LPIPS(
    net_type='vgg',  # choose a network type from ['alex', 'squeeze', 'vgg']
    version='0.1'  # Currently, v0.1 is supported
)
criterion_L1 = nn.L1Loss()
criterion_amp = nn.MSELoss()
criterion_phase = nn.MSELoss()

############################
#  Utils
############################

# Calculate output of image discriminator (PatchGAN)
patch = (1, opt.img_height // 2 ** 4, opt.img_width // 2 ** 4)
#patch = (1, opt.patch_height // 2 ** 4, opt.patch_width // 2 ** 4) # < using patches, not global wh
print("discriminator patch:", patch)

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

def expand(tensor):
    # to add 3Channels from 1 channel (handy)
    t = torch.Tensor(tensor).cuda()
    t = t.reshape(t.size(0), 1, t.size(1), t.size(2))
    t = t.expand(-1, 3, -1, -1) # needs to be [1, 3, 256, 256]
    return t
    
def sample_images(batches_done):
    imgs = next(iter(test_dataloader)) # batch_size = 1
    real_A = Variable(imgs["A"].type(HalfTensor)) # torch.Size([1, 3, 256, 256])
    real_B = Variable(imgs["B"].type(HalfTensor))

    # dummy
    dummy_grid = torch.rand(real_A.size(0), real_A.size(1), real_A.size(2), 2) # torch.Size([1, 256, 256, 2])

    # Part (a) Deformation Grid 
    init_grid = model(img_A=real_A, img_B=real_B, src=real_A, grid=dummy_grid, mode='init') # torch.Size([1, 256, 256, 2])

    # Flow 1 - A is deformed by theta, which was already learned above 
    warped_A = model(img_A=real_A, img_B=real_B, src=real_A, grid=init_grid.detach(), mode='flow') 
    fake_WB = generator(warped_A)

    # Flow 2 - Fake_2B is deformed by theta
    fake_2B = generator(real_A)
    warped_2B = model(img_A=real_A, img_B=real_B, src=fake_2B, grid=init_grid.detach(), mode='flow')

    # Next time - print out the deformation grid 
    #plot_init = init_grid.reshape(init_grid.size(0), 2, init_grid.size(1), init_grid.size(2))
    #plot_init = init_grid.squeeze(0).permute(1,2,0)/255
    #print("plot_init:", plot_init.size())
    
    img_sample_global = torch.cat((real_A.data, warped_A.data, fake_WB.data, fake_2B.data, warped_2B.data), -1)
    save_image(img_sample_global, "images/%s/%s_P1.png" % (opt.experiment, batches_done), nrow=4, normalize=True)

    
def print_network(model, name):
    """Print out the network information."""
    num_params = 0
    for p in model.parameters():
        num_params += p.numel()
    print(model)
    print(name)
    print("The number of parameters: {}".format(num_params))


#################    
# ViT for STN
################

class LocalizerVIT(nn.Module):
    def __init__(self, img_shape):
        super(LocalizerVIT, self).__init__()
        channels, self.h, self.w = img_shape
        self.vit = nn.Sequential(
            K.VisionTransformer(image_size=self.h, patch_size=16, in_channels=channels*2) # (A,B), changed from 16-> 32 patch
        )
        
    def forward(self, x):
        with autocast():
            out = self.vit(x).type(HalfTensor) # returns torch.Size([batch, 257, 768])
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

        self.fc_loc = nn.Sequential(
            nn.Linear(1*257*768, 256), # (hard-coded in, 257, 768 based on ViT output)
            nn.ReLU(True),
            nn.Linear(256, 3*2))

        self.fc_loc[2].bias.data.zero_() # DO NOT CHANGE! the problem for everything is this, don't change it

       
    def stn_phi(self, x):
        xs = self.localization(x) # (A,B), 6 ch
        xs = xs.view(-1, 1 * xs.size(1) * xs.size(2)) 
        theta = self.fc_loc(xs)  
        theta = theta.view(-1, 2, 3)
        return theta
    

    def forward(self, img_A, img_B, src, grid, mode):
        identity_matrix = [1, 0, 0, 0, 1, 0] #identity_theta = torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float).cuda()
        
        with autocast():  
            if mode=='flow':
                # just the tensors of the source image to be aligned
                src_tensors = []
                for img in src: 
                    this_img_src = img.reshape(1, img.size(0), img.size(1), img.size(2)) 
                    src_tensors.append(this_img_src)

                # use the just passed deformation grid from init
                # have to unpack from torch.Size([batch/3, 256, 256, 2]) -> grid[i]
                my_grids = []
                for g in grid: # torch.Size([12, 256, 256, 2])
                    #print("g:", g.size()) # torch.Size([256, 256, 2])
                    this_grid = g.reshape(1, g.size(0), g.size(1), g.size(2)) # (1, 256, 256, 2)
                    #print("this_grid:", this_grid.size())
                    my_grids.append(this_grid)
                    
                warped = []
                for i in range(len(src_tensors)):
                    Rs = F.grid_sample(src_tensors[i], grid=my_grids[i],  mode='bicubic', padding_mode='zeros', align_corners=True)  # changed from border -> zeros
                    warped.append(Rs.type(HalfTensor))

                return torch.cat(warped)
            
            elif mode=='init':
                img_input = torch.cat((img_A, img_B), 1)
                dtheta = self.stn_phi(img_input) 
                identity_theta = torch.tensor(identity_matrix, dtype=torch.float).cuda()
                dtheta = dtheta.reshape(img_A.size(0), 2*3)
                theta = dtheta + identity_theta.unsqueeze(0).repeat(img_A.size(0), 1)
                
                theta_batches = []
                for t in theta:
                    this_theta = (t.view(-1, 2, 3)).reshape(1,2,3)
                    theta_batches.append(this_theta)

                # just the tensors of the source image to be aligned
                src_tensors = []
                for img in src: 
                    this_img_src = img.reshape(1, img.size(0), img.size(1), img.size(2)) 
                    src_tensors.append(this_img_src)

                # result
                grids = []
                for i in range(len(src_tensors)):
                    rs_grid = F.affine_grid(theta_batches[i], src_tensors[i].size(), align_corners=True) # (1, 256, 256, 2)
                    grids.append(rs_grid)

                return torch.cat(grids)


##############################
#     Generator  U-NET
# These have BlurPool b/c it's TFC-GAN
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
    
class UNetDownNOBP(nn.Module):
    def __init__(self, in_size, out_size, normalize=True, dropout=0.0):
        super(UNetDownLoc, self).__init__()
        layers = [nn.Conv2d(in_size, out_size, 4, 2, 1, bias=False)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_size))
        layers.append(nn.LeakyReLU(0.2))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.model(x)


class UNetUpNOBP(nn.Module):
    def __init__(self, in_size, out_size, dropout=0.0):
        super(UNetUpLoc, self).__init__()
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

    
##############################
#        Discriminator
##############################

class Discriminator(nn.Module):

    def __init__(self, img_shape):
        super(Discriminator, self).__init__()
        channels, self.h, self.w = img_shape

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


#####################
# LOSSES
#####################

def iou_global_loss(real_A, B_R, smooth=1):
    #print("iou_global_loss A input:", real_A.size())
    #print("iou_global_loss B_R input:", B_R.size())
    
    # takes tensor, converts to grayscale, converts to numpy 
    gray_A = transforms.Grayscale()(real_A)
    gray_R = transforms.Grayscale()(B_R) # says it can take a tensor
    
    #print("grayscale A: {}, grayscale R: {}".format(gray_A.size(), gray_R.size()))
    
    # convert to numpy for medpy package
    A = gray_A.cpu().detach().numpy()
    R = gray_R.cpu().detach().numpy()
    
    # runs anisotropic, must be graysclae
    aniso_A = anisotropic_diffusion(A, niter=1, kappa=20.5, gamma=100, option=3)
    aniso_R = anisotropic_diffusion(R, niter=1, kappa=20.5, gamma=100, option=3)
          
    # convert back to tensor
    aniso_A = torch.from_numpy(aniso_A).cuda()  
    aniso_R = torch.from_numpy(aniso_R).cuda()  
          
    #intersection = (aniso_R * aniso_A).sum()
    #total = (aniso_R + aniso_A).sum()
    #union = total - intersection 

    #IoU = (intersection + smooth)/(union + smooth)

    # 1 - b/c you want to maximize this IoU, 90% IoU is 0.10 Loss (min it)        
    #loss = 1 - IoU
    
    mse_loss = nn.MSELoss()
    
    loss = mse_loss(aniso_A, aniso_R)
    return loss


def global_pixel_loss(real_B, fake_B):
    loss_pix = criterion_lpips(fake_B, real_B)
    return loss_pix

def global_fourier_loss(BR, fake_B): 
    Af, Pf = fft_components(fake_B, patch=False)
    Ar, Pr = fft_components(BR, patch=False)
    loss_Pha = criterion_phase(Pf, Pr)
    loss_Amp = criterion_amp(Af, Ar)
    loss_FFT = (0.5*(loss_Pha + loss_Amp))*0.01 # 0.01 lambda
    return loss_FFT

def global_gen_loss(real_A, real_B, fake_B):
    pred_fake_B = discriminator(fake_B, real_A)
    real_pred_B = discriminator(real_B, real_A)
    loss_GAN = criterion_GAN(pred_fake_B - real_pred_B.detach(), valid)
    return loss_GAN

def global_disc_loss(real_A, real_B, fake_B):
    pred_real_B = discriminator(real_B, real_A)
    pred_fake_B = discriminator(fake_B.detach(), real_A)
    loss_real_B = criterion_GAN(pred_real_B - pred_fake_B, valid)
    loss_fake_B = criterion_GAN(pred_fake_B - pred_real_B, fake)
    loss_D = 0.25*(loss_real_B + loss_fake_B)
    return loss_D
    

# ===========================================================
# Initialize generator and discriminator
input_shape_patch = (opt.channels, opt.img_height, opt.img_width)
generator = GeneratorUNet(input_shape_patch) 
discriminator = Discriminator(input_shape_patch)

if cuda:
    generator = generator.cuda()
    discriminator = discriminator.cuda()
    model = Net().cuda()
    
    criterion_GAN.cuda()
    criterion_lpips.cuda()
    criterion_L1.cuda()
    criterion_amp = nn.L1Loss()
    criterion_phase = nn.L1Loss()
    
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
    model.apply(weights_init_normal) 

# Optimizers - joint model, fusion, and G together
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
f = open('/home/local/AD/cordun1/experiments/TFC-GAN/LOGS/{}.txt'.format(opt.experiment), 'a+')

# Try AMP
scaler = GradScaler()

for epoch in range(opt.epoch, opt.n_epochs):
    for i, batch in enumerate(dataloader):
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
        print("+ + + optimizer_G.zero_grad() + + + ")   
        with autocast(): 
            
            # dummy
            dummy_grid = torch.rand(real_A.size(0), real_A.size(1), real_A.size(2), 2) # torch.Size([36, 256, 256, 2])
            
            # Part (a) Deformation Grid 
            init_grid = model(img_A=real_A, img_B=real_B, src=real_A, grid=dummy_grid, mode='init') # torch.Size([36, 256, 256, 2])
        
            # Flow 1 - A is deformed by theta, which was already learned above
            warped_A = model(img_A=real_A, img_B=real_B, src=real_A, grid=init_grid.detach(), mode='flow') # You have to detach the grid, otherwise gradients will flow backwards and it will change!!!
            fake_WB = generator(warped_A)
            recon_loss1 = global_pixel_loss(real_B, fake_WB)
            
            # Flow 2 - Fake_2B is deformed by theta
            fake_2B = generator(real_A)
            warped_2B = model(img_A=real_A, img_B=real_B, src=fake_2B, grid=init_grid.detach(), mode='flow') #torch.Size([36, 3, 256, 256])
            recon_loss2 = global_pixel_loss(real_B, warped_2B)
            
            loss_pixel = 0.5*(recon_loss1 + recon_loss2)
            
            """
            if i % 20 == 0: # something random
                # I want to see what it looks like ^^
                training_sample = torch.cat((real_A.data, warped_A.data, fake_WB.data, fake_2B.data, warped_2B.data), -1)
                save_image(training_sample, "images/%s/%s_%s_TS.png" % (opt.experiment, i, epoch), nrow=4, normalize=True)
            """
  
            # LPIPS patch loss, reconstruction using the ground truth as the fake 
            #loss_pixel = 0.5*(global_pixel_loss(real_B, fake_B) + global_pixel_loss(real_A, fake_A))
            
            # Fourier Transform Loss for Each Patch
            #loss_FFT = 0.5*(global_fourier_loss(real_B, fake_WB) + global_fourier_loss(real_B, warped_2B))
 
            # Adverarial - How fake and how real
            loss_GAN1 = global_gen_loss(real_A, real_B, fake_WB)
            loss_GAN2 = global_gen_loss(real_A, real_B, warped_2B)
            loss_GAN = 0.5*(loss_GAN1 + loss_GAN2)
            
            # Total Loss
            alpha1 = 0.001
            alpha2 = 0.01
            loss_G = loss_GAN + alpha2*loss_pixel

        scaler.scale(loss_G).backward()
        scaler.step(optimizer_G)
        print("+ + + optimizer_G.step() + + + ")
        
        # -----------------------
        #  Train Discriminator 
        # -----------------------

        optimizer_D.zero_grad()
        print("+ + + optimizer_D.zero_grad() + + +")
        with autocast():
            
            loss_D1 = global_disc_loss(real_A, real_B, fake_WB)
            loss_D2 = global_disc_loss(real_A, real_B, warped_2B)
            loss_D = 0.5*(loss_D1 + loss_D2)

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
            "\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, pix: %f ] ETA: %s"
            % (
                epoch, #%d
                opt.n_epochs, #%d
                i, #%d
                len(dataloader), #%d
                loss_D.item(), #%f
                loss_G.item(), #%f - total G loss
                loss_pixel.item(),
                time_left, #%s
            )
        )
        
        f.write(
            "\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, pix: %f] ETA: %s"
            % (
                epoch, #%d
                opt.n_epochs, #%d
                i, #%d
                len(dataloader), #%d
                loss_D.item(), #%f
                loss_G.item(), #%f - total G loss
                loss_pixel.item(),
                time_left, #%s
            )
        )
        
        # If at sample interval save image
        if batches_done % opt.sample_interval == 0:
            sample_images(batches_done) 


    if opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0:
        # Save model checkpoints
        torch.save(generator.state_dict(), "/home/local/AD/cordun1/experiments/TFC-GAN/saved_models/%s/generator_%d.pth" % (opt.experiment, epoch))
        torch.save(discriminator.state_dict(), "/home/local/AD/cordun1/experiments/TFC-GAN/saved_models/%s/discriminator_%d.pth" % (opt.experiment, epoch))
        torch.save(model.state_dict(), "/home/local/AD/cordun1/experiments/TFC-GAN/saved_models/%s/stn_%d.pth" % (opt.experiment, epoch))


                   