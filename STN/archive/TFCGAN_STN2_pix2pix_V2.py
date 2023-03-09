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
#from lpips_pytorch import LPIPS, lpips
import cv2
from torch.distributed import Backend
#from torch.nn.parallel.distributed import DistributedDataParallel
#import antialiased_cnns
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from datasets_stn import *

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=210, help="number of epochs of training")
parser.add_argument("--dataset_name", type=str, default="eurecom_warped_pairs", help="name of the dataset")
parser.add_argument("--batch_size", type=int, default=12, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
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

#----
# Loss functions
criterion_GAN = torch.nn.BCEWithLogitsLoss() 
criterion_L1= torch.nn.L1Loss()

# Loss weight of L1 pixel-wise loss between translated image and real image
lambda_pixel = 100

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
    imgs = next(iter(test_dataloader))
    real_A = Variable(imgs["A"].type(HalfTensor))
    real_B = Variable(imgs["B"].type(HalfTensor))

    fake_A = generator(real_B) # translate: OT = T (Ia)
    warped_B, warped_fA, reg_term = model(img_A=real_A, img_B=real_B, src=[real_B, fake_A]) 

    registered_B = torch.cat(warped_B) # should be the grid applied to B <- THIS IS WHAT WE WANT AT TEST TIME
    print("---sample---")
    print("real_A size():", real_A.size())
    print("registered_B.size():", registered_B.size())
    fake_TR_A = generator(registered_B)

    # => Flow 2 - Translate First ---------
    fake_RT_A = torch.cat(warped_fA) # fake_A deformations
    
    # GLOBAL
    img_sample_global = torch.cat((real_A.data, registered_B.data, fake_TR_A.data, fake_RT_A.data, real_B.data), -2)
    save_image(img_sample_global, "images/%s/%s_g.png" % (opt.experiment, batches_done), nrow=5, normalize=True)

    
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
        self.up5 = UNetUp(256, 64)

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
            out = u5.type(HalfTensor)
            #out = self.final(u5).type(HalfTensor)
        return out
    
#################
# STN
#################
                          
class Net(nn.Module):
    """simplified from official: 
    https://github.com/moabarar/nemar/blob/nemar_deploy/models/stn/affine_stn.py"""
    def __init__(self):
        super(Net, self).__init__()

        # Spatial transformer localization-network
        # ORIGINAL
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
        """
        # SECOND MODEL - outputted gray after small padded images
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
            nn.Linear(512*64*64, 120), # <- change this
            nn.ReLU(True),
            nn.Linear(120, 3 * 2)
        )
        
        # UNET
        input_shape = (opt.channels, opt.img_height, opt.img_width)
        self.localization = LocalizerUNet(input_shape)
        
         # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(128*128*128, 256), # <- change this
            nn.ReLU(True),
            nn.Linear(256, 3 * 2)
        )
        """

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.normal_(mean=0.0, std=5e-4)
        self.fc_loc[2].bias.data.zero_() # OMG DON'T CHANGE THIS! It rescales the image and pads it!!!
        
    # Spatial transformer network forward function
    def stn_phi(self, x):
        xs = self.localization(x) # convolves the cat channel=6
        print("stn_phi xs:", xs.size()) # stn_phi xs: torch.Size([4, 128, 128, 128])
        xs = xs.view(-1, 10*60*60) 
        #xs = xs.view(-1, 512*64*64) 
        print("xs size:", xs.size()) # xs size: torch.Size([4, 131072])
        
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)
        return theta
    
    def get_grid(self, theta, src):
        #Use for a single image where src = real_B
        # we don't call get_grid until prediction time for a single image
        grid = F.affine_grid(theta, src.size())
        print("grid size:", grid.size())
        return grid
                    
    def stn_resample(self, src, rs_grid):
        #Use for a single image where src = real_B
        Rs = F.grid_sample(src, grid)
        return Rs

    def forward(self, img_A, img_B, src):
        # src = [real_B, fake_A] a list of tensors
        with autocast():
            
            #print("src: {} | a: {} | b: {}".format(src.size(), img_A.size(), img_B.size()))
            """
            b/c 3 GPUs: 
            src: torch.Size([4, 3, 256, 256]) | a: torch.Size([4, 3, 256, 256]) | b: torch.Size([4, 3, 256, 256])
            src: torch.Size([4, 3, 256, 256]) | a: torch.Size([4, 3, 256, 256]) | b: torch.Size([4, 3, 256, 256])
            src: torch.Size([4, 3, 256, 256]) | a: torch.Size([4, 3, 256, 256]) | b: torch.Size([4, 3, 256, 256])
            
            """
            
            print("src is: {}, {}".format(src[0].size(), src[1].size()))
            # src is: torch.Size([4, 3, 256, 256]), torch.Size([4, 3, 256, 256])
            
            img_input = torch.cat((img_A, img_B), 1)
            dtheta = self.stn_phi(img_input) # deformation field for real A and real B
            identity_theta = torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float).cuda()
            
            print("dtheta size: {} | img_B size: {}".format(dtheta.size(), img_B.size()))
            # dtheta size: torch.Size([4, 2, 3]) | img_B size: torch.Size([4, 3, 256, 256])
            
            dtheta = dtheta.reshape(img_B.size(0), 2*3)
            print("img_B: {} | dtheta: {}  ".format(img_B.get_device(), dtheta.get_device()))
            #theta = dtheta + self.identity_theta.unsqueeze(0).repeat(img_B.size(0), 1)
            theta = dtheta + identity_theta.unsqueeze(0).repeat(img_B.size(0), 1)
            print("theta size:", theta.size())
            
            # The theta is what's being learned by the NN - grid and grid_sample are just functions
            theta_batches = []

            for t in theta:
                # torch.Size([1, 2, 3]) for theta0, theta1, theta2, theta3, since there are 4 batches
                #this_theta = (theta.view(-1, 2, 3)[t]).reshape(1,2,3)
                this_theta = (t.view(-1, 2, 3)).reshape(1,2,3)
                theta_batches.append(this_theta)
                
                
            """
            I have confirmed they are different
            
            -t-: tensor([ 2.0000e+00, -1.3793e-04,  4.1127e-06,  9.5367e-07,  2.0000e+00,
                    -6.8784e-05], device='cuda:1', grad_fn=<UnbindBackward>)
            
            -t-: tensor([ 2.0000e+00, -1.1772e-04,  2.0027e-05,  7.0930e-06,  2.0000e+00,
                    -3.9876e-05], device='cuda:2', grad_fn=<UnbindBackward>)
            
            -t-: tensor([ 2.0000e+00, -1.1736e-04,  2.8431e-05, -3.0994e-06,  2.0000e+00,
                    -2.1756e-05], device='cuda:0', grad_fn=<UnbindBackward>)
            
            -t-: tensor([ 2.0000e+00, -1.3494e-04,  5.5671e-05,  2.2948e-05,  2.0000e+00,
                    -4.9174e-05], device='cuda:1', grad_fn=<UnbindBackward>)
            """
                
            print("---> len(theta_batches):", len(theta_batches))
            
            images_B = []
            for img in src[0]: #real_B
                #print("img:", img.size())
                this_img_B = img.reshape(1, img.size(0), img.size(1), img.size(2)) # affine_grid only takes 4D - was [3,256,256]-> make it [1,3,256,256]
                #print("img size:", img.size())
                images_B.append(this_img_B)
                
            print("---> len(images_B):", len(images_B))
            
            images_fA = []
            for img_ in src[1]: #fake_A
                #print("img:", img.size())
                this_img_fA = img_.reshape(1, img_.size(0), img_.size(1), img_.size(2)) # affine_grid only takes 4D - was [3,256,256]-> make it [1,3,256,256]
                #print("img size:", img.size())
                images_fA.append(this_img_fA)
                
            print("---> len(images_fA):", len(images_fA))
            
            warped_B = []
            # match with src = real_B
            for i in range(len(images_B)): #1:1 match with theta, matching with theta is important
                print("Each image in collection images_B is {} size".format(images_B[i].size())) # torch.Size([1, 3, 256, 256]) size
                rs_grid_B = F.affine_grid(theta_batches[i], images_B[i].size())
                
                Rs_B = F.grid_sample(images_B[i], rs_grid_B,  mode='bilinear', padding_mode='zeros', align_corners=False)
                
                warped_B.append(Rs_B.type(HalfTensor))
                
            print("---> len(warped_B):{}".format(len(warped_B))) 
                
            # match with src - fake_A
            warped_fA = []
            for i in range(len(images_fA)): #1:1 match with theta, matching with theta is important
                rs_grid_fA = F.affine_grid(theta_batches[i], images_fA[i].size())
                Rs_fA = F.grid_sample(images_fA[i], rs_grid_fA,  mode='bilinear', padding_mode='zeros', align_corners=False)
                warped_fA.append(Rs_fA.type(HalfTensor))
                
            print("---> len(warped_fA):{}".format(len(warped_fA))) 
            
           
            # for training, you need all the images in the batch
            # grid_sample has to be run for each for them
            #warped_images = []
            
            #for img in src: # <- apply on the fake_B which is src
                #img = img.reshape(1, img.size(0), img.size(1), img.size(2)) # affine_grid only takes 4D - was [3,256,256]-> make it [1,3,256,256]
                #print("img size:", img.size())
                #rs_grid = F.affine_grid(batch_theta, img.size())
                #Rs = F.grid_sample(img, rs_grid, mode='bilinear', padding_mode='zeros', align_corners=False)
                #warped_images.append(Rs.type(HalfTensor))
                

                # there is a theta for each of the batches - you have to pass each one separately
                #print("-----")
                #print(img.size())
                #print("theta.view(-1, 2, 3):", theta.view(-1, 2, 3).size()) # torch.Size([4, 2, 3])
                #for t in range(theta.size(0)):
                    #foo = theta.view(-1, 2, 3)[t]
                    #foo = foo.reshape(1, foo.size(0), foo.size(1))
                    #print(foo.size())
                    #woo = foo.reshape(1, 2, 3)
                    #print(woo.size())
                # has to be theta (Tensor) – input batch of affine matrices with shape (N×2×3)

                #for t in range(theta.size(0)): 
                    #print("-t-:", t)
                    # torch.Size([1, 2, 3]) for theta0, theta1, theta2, theta3, since there are 4 batches
                    #batch_theta = (theta.view(-1, 2, 3)[t]).reshape(1,2,3)
                    #print("-----")
                    #print(batch_theta.size())
                    #print(img.size())
                    #print("-----")
                    #rs_grid = F.affine_grid(batch_theta, img.size())
    
                    #Rs = F.grid_sample(img, rs_grid, mode='bilinear', padding_mode='zeros', align_corners=False)
                    
                    #warped_images.append(Rs.type(HalfTensor))
                       
            # Calculate STN regularization term - for affine transformation, the predicted affine transformation should not
            # largely deviate from the identity transformation.
            
            reg_term = torch.mean(torch.abs(dtheta))
            
        return warped_B, warped_fA, reg_term

                    
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
            print("G out:", out.size()) # torch.Size([4, 3, 256, 256])
        return out

    

                    
##############################
#        Discriminator
##############################

class Discriminator(nn.Module):

    def __init__(self, img_shape):
        super(Discriminator, self).__init__()
        channels, self.h, self.w = img_shape
        def discriminator_block(in_filters, out_filters, normalization=True):
            """Returns downsampling layers of each discriminator block"""
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
       
    def forward(self, img_A, img_B):
        with autocast():
            img_input = torch.cat((img_A, img_B), 1)
            d_in = img_input
            output = self.model(d_in)
            print("D output:", output.size())
        return output.type(HalfTensor)
    
# ===========================================================
# Initialize generator and discriminator
input_shape = (opt.channels, opt.img_height, opt.img_width)

generator = GeneratorUNet(input_shape)
discriminator = Discriminator(input_shape)
#localizer = LocalizerUNet(input_shape)

if cuda:
    #localizer = localizer.cuda()
    generator = generator.cuda()
    discriminator = discriminator.cuda()
    criterion_GAN.cuda()
    criterion_L1.cuda()
    model = Net().cuda()

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
    # stn has its own initialized weights and uses the identity matrix

# Optimizers
#optimizer_G = torch.optim.Adam(itertools.chain(generator.parameters(), model.parameters()), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_M = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

# Tensor type - only use HalfTensor in this AMP script                    
Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor
HalfTensor = torch.cuda.HalfTensor if cuda else torch.HalfTensor

def print_network(model, name):
    """Print out the network information."""
    num_params = 0
    for p in model.parameters():
        num_params += p.numel()
    print(model)
    print(name)
    print("The number of parameters: {}".format(num_params))

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

#f = open('/home/local/AD/cordun1/experiments/TFC-GAN/LOGS/{}.txt'.format(opt.out_file), 'a+')

#print_network(generator, 'generator')
#print_network(discriminator1, 'discriminator1')

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
        optimizer_M.zero_grad()
        print("+ + + optimizer_G.zero_grad() + + +")   
        print("+ + + optimizer_M.zero_grad() + + +")
        with autocast(): 
            
            # => Flow 1 - Register First ---------
            print("Flow 1 - Register First...")
            # Apply STN Sampler on the generated output, apply transform on B (grid_sample is not a NN)
            # STN model does the grid and the resampling at once, outputs the sampled
            # src is real_B b/c this is the one I want to be registered/deformed to match A
            
            fake_A = generator(real_B) # translate: OT = T (Ia)
            warped_B, warped_fA, reg_term = model(img_A=real_A, img_B=real_B, src=[real_B, fake_A]) 

            print("len(warped_B):", len(warped_B)) # should be batch
            print("len(warped_fB):", len(warped_fA)) # should be batch
        
            registered_B = torch.cat(warped_B) # should be the grid applied to B
            print("registered_B:", registered_B.size())
            
            # Now translate - pass the OR with its deformations to the generator 
            fake_TR_A = generator(registered_B)
            print("fake_TR_A:", fake_TR_A.size())
         
            # => Flow 2 - Translate First ---------
            print("Flow 2: Translate First...")
            """This has actually already happened in the beginning
            when we generated fake_B (translation happened 1st), the 
            fake_B was then passed to the STN as src; the warped_images list
            was the output, where the [1] slice is the O_T."""
            #O_RonT = warped_images # OR = R(Ia , φ): registered outputs
            fake_RT_A = torch.cat(warped_fA) # fake_A deformations
            print("fake_RT_A:", fake_RT_A.size()) 
        
            # Rconstruction error - Lrecon(T, R) = ∥ORT − Ib∥1 + ∥OT R − Ib∥1
            reconstruction_loss = criterion_L1(fake_TR_A, real_A) + criterion_L1(fake_RT_A, real_A)
                    
            # Adv 
            pred_fake_TR = discriminator(fake_TR_A.cuda(), real_B)
            pred_fake_RT = discriminator(fake_RT_A.cuda(), real_B)
            
            #print(pred_fake_RT.get_device())
            #print(pred_fake_TR.get_device())
            #print(real_B.get_device())

            loss_GAN = criterion_GAN(pred_fake_TR, valid) + criterion_GAN(pred_fake_RT, valid)
            #print("valid:", valid.size()) 
            #print(criterion_GAN(pred_fake_TR, valid))
            #print(criterion_GAN(pred_fake_RT, valid))
            
            # Total Loss
            loss_G = 0.5*(loss_GAN + reconstruction_loss)
            

        scaler.scale(loss_G).backward()
        scaler.step(optimizer_G)
        scaler.step(optimizer_M)
        print("+ + + optimizer_G.step() + + +")
        print("+ + + optimizer_M.step() + + +")
        # -----------------------
        #  Train Discriminator 
        # -----------------------

        optimizer_D.zero_grad()
        print("+ + + optimizer_D.zero_grad() + + +")
        with autocast():

            # Real loss
            pred_real = discriminator(real_A, real_B)
            loss_real = criterion_GAN(pred_real, valid)

            # Fake loss of O_TonR (translated on the registered)
            pred_fake_TR_d = discriminator(fake_TR_A.detach(), real_B)
            loss_fake_TR_d = criterion_GAN(pred_fake_TR_d, fake)
                    
            # Fake loss of O_RonT (registered the translated image)
            pred_fake_RT_d = discriminator(fake_RT_A.detach(), real_B)
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
        batches_left = opt.n_epochs * len(dataloader) - batches_done
        time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
        prev_time = time.time()

        # Print log
        sys.stdout.write(
            "\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, L1: %f, GAN: %f] ETA: %s"
            % (
                epoch, #%d
                opt.n_epochs, #%d
                i, #%d
                len(dataloader), #%d
                loss_D.item(), #%f
                loss_G.item(), #%f - total G loss
                reconstruction_loss.item(), #%f
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

                   