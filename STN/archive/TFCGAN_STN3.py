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
#import mediapipe as mp
#from protobuf_to_dict import protobuf_to_dict 
from datasets_stn import *
from torchmetrics import JaccardIndex
import kornia

"""
based on TFCGAN_STN2_pix2pix_V8_fBA_v4.PY
"""

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=210, help="number of epochs of training")
parser.add_argument("--dataset_name", type=str, default="eurecom_warped_pairs", help="name of the dataset")
parser.add_argument("--batch_size", type=int, default=12, help="size of the batches")
parser.add_argument("--lr", type=float, default=1e-3, help="adam: learning rate") # changed this to faster LR 
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_height", type=int, default=256, help="size of image height")
parser.add_argument("--img_width", type=int, default=256, help="size of image width")
parser.add_argument("--patch_height", type=int, default=128, help="size of patch height")
parser.add_argument("--patch_width", type=int, default=128, help="size of patch width")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=100, help="interval between sampling of images from generators")
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
criterion_MSE = torch.nn.MSELoss()

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

def expand(tensor):
    t = torch.Tensor(tensor).cuda()
    t = t.reshape(t.size(0), 1, t.size(1), t.size(2))
    t = t.expand(-1, 3, -1, -1) # needs to be [1, 3, 256, 256]
    return t
    
def sample_images(batches_done):
    imgs = next(iter(test_dataloader)) # batch_size = 1
    real_A = Variable(imgs["A"].type(HalfTensor))
    real_B = Variable(imgs["B"].type(HalfTensor))

    fake_B = generator(real_A) 
    
    """
    thresh_gray_V, Vgrays = torch_to_mask_test(real_A)
    tv = expand(thresh_gray_V)
    print("tv.size:", tv.size())
    tvg = expand(Vgrays)
    print("tvg.size:", tvg.size())

    thresh_gray_T, Tgrays = torch_to_mask_test(fake_B) 
    tt = expand(thresh_gray_T)
    ttg = expand(Tgrays)
    """
     
    warped_B = model(img_A=real_A, img_B=fake_B, src=real_B)
    R_ = torch.cat(warped_B)
    
    masks_A = torch.where(real_A > 0, 1.0, 0.) # torch.Size([b, 3, 256, 256])
    masks_RB = torch.where(R_ > 0, 1.0, 0.) # torch.Size([b, 3, 256, 256])
   
    lap_A = canny_edges_test(masks_A) # torch.Size([1, 1, 256, 256])
    lap_RB = canny_edges_test(masks_RB)

    img_sample_global = torch.cat((real_A.data, real_B.data, R_.data, masks_A.data, lap_A.data), -2)
    save_image(img_sample_global, "images/%s/%s_g.png" % (opt.experiment, batches_done), nrow=5, normalize=True)

    
def print_network(model, name):
    """Print out the network information."""
    num_params = 0
    for p in model.parameters():
        num_params += p.numel()
    print(model)
    print(name)
    print("The number of parameters: {}".format(num_params))


#################    
# UNET for STN
################

# I think The Localizer UNET needs to be BlurPool-free
class UNetDownLoc(nn.Module):
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


class UNetUpLoc(nn.Module):
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


class LocalizerUNet(nn.Module):
    def __init__(self, img_shape):
        super(LocalizerUNet, self).__init__()
        channels, self.h, self.w = img_shape
        self.down1 = UNetDownLoc(channels*2, 64)
        self.down2 = UNetDownLoc(64, 128)
        self.down3 = UNetDownLoc(128, 256)
        self.down4 = UNetDownLoc(256, 512)
        self.down5 = UNetDownLoc(512, 512)
        self.down6 = UNetDownLoc(512, 512)

        self.up1 = UNetUpLoc(512, 512)
        self.up2 = UNetUpLoc(1024, 512)        
        self.up3 = UNetUpLoc(1024, 256)
        self.up4 = UNetUpLoc(512, 128)
        self.up5 = UNetUpLoc(256, 64) # Then it'll only be 64 x 64

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
            out = self.final(u5).type(HalfTensor)
        return out

#################
# STN
#################
                          
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        input_shape = (opt.channels, opt.img_height, opt.img_width)
        self.localization = LocalizerUNet(input_shape)

        self.fc_loc = nn.Sequential(
            nn.Linear(3*256*256, 256), 
            nn.ReLU(True),
            nn.Linear(256, 3 * 2)
        )

        self.fc_loc[2].bias.data.zero_() # DO NOT CHANGE! the problem for everything is this, don't change it
       
    def stn_phi(self, x):
        xs = self.localization(x) # convolves the cat channel=6
        xs = xs.view(-1, 3*256*256) 
        theta = self.fc_loc(xs)       
        theta = theta.view(-1, 2, 3)
        return theta

    def forward(self, img_A, img_B, src):
        with autocast():
            img_input = torch.cat((img_A, img_B), 1) # send concatenated A&B to LocalizerUNET
            dtheta = self.stn_phi(img_input) # deformation field for real A and real B
            identity_theta = torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float).cuda()
            dtheta = dtheta.reshape(img_B.size(0), 2*3)
            theta = dtheta + identity_theta.unsqueeze(0).repeat(img_B.size(0), 1)
    
            # get each theta for the batch
            theta_batches = []
            for t in theta:
                this_theta = (t.view(-1, 2, 3)).reshape(1,2,3)
                theta_batches.append(this_theta)

            # just the tensors of the source image to be aligned
            src_tensors = []
            for img in src: 
                #print("src img size:", img.size())
                this_img_src = img.reshape(1, img.size(0), img.size(1), img.size(2)) 
                src_tensors.append(this_img_src)
                
            # result
            warped = []
            for i in range(len(src_tensors)): #1:1 match with theta, matching with theta is important
                # says: apply the theta deformation grid on the source based on the learned matrix
                rs_grid = F.affine_grid(theta_batches[i], src_tensors[i].size(), align_corners=True)
                # Do not change from nearest
                Rs = F.grid_sample(src_tensors[i], rs_grid,  mode='nearest', padding_mode='border', align_corners=True)
                warped.append(Rs.type(HalfTensor))

        return warped

    
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
# MASKS
#####################

"""
def torch_to_mask_test(tensor):
    # converts A and B Tensors -> Numpy Arr for cv2
    masks = []
    grays = []
    
    for m in range(0, tensor.size(0)):
        # takes pytorch tensor and converts to grayscale array for cv2
        gr = tensor[m, :, :, :].data.view(opt.img_width, opt.img_width, 3) # get the data in cv2 format
        #print("torch_to_mask:", gr.size())
        
        gr = gr.detach().cpu().numpy() #turn into array 
        gr = gr.astype(np.uint8) # convert to UINT
        gr = cv2.cvtColor(gr, cv2.COLOR_BGR2GRAY)
        print("gr test shape:", gr.shape)
        
        grays.append(gr)
        
        #make mask
        retval, thresh = cv2.threshold(gr, thresh=100, maxval=255, type=cv2.THRESH_BINARY)
        masks.append(thresh)
    
    return masks,grays
"""

"""
def pytorch_mask_maker(tensor):
    
    MASKS = []
    
    for m in range(0, tensor.size(0)):   
        mask = torch.where(tensor > 0, 1.0, 0.)
        print("mask.size:", mask.size())
        MASKS.append(mask)
        
    return MASKS 
"""

def prep(img_A, img_B):
 
    masks_V = Variable(torch.where(img_A > 0, 1.0, 0.), requires_grad=True).cuda()
    masks_T = Variable(torch.where(img_B > 0, 1.0, 0.), requires_grad=True).cuda() # torch.Size([b, 3, 256, 256])
   
    lap1 = Variable(canny_edges(masks_V), requires_grad=True).cuda() # torch.Size([b, 1, 256, 256])
    lap2 = Variable(canny_edges(masks_T), requires_grad=True).cuda()
    
    print("----------prep----------")
    print(masks_V.size())
    print(masks_T.size())
    print(lap1.size())
    print(lap2.size())
 
    return masks_V, masks_T, lap1, lap2


def canny_edges(tensor):
    # tensor is [256,256] binary mask image
    x_rgb: torch.cuda.HalfTensor = kornia.color.bgr_to_rgb(tensor) # will add 3 channels
    tensor = tensor.reshape(x_rgb.size(0), 3, opt.img_width, opt.img_width) #[batch, 3, W, H)
    x_gray = (kornia.color.rgb_to_grayscale(tensor)) 
    x_laplacian: torch.cuda.HalfTensor = kornia.filters.canny(tensor)[0]    
    return x_laplacian


def canny_edges_test(tensor):
    x_rgb: torch.cuda.HalfTensor = kornia.color.bgr_to_rgb(tensor) # will add 3 channels
    tensor = tensor.reshape(1,3,opt.img_width, opt.img_width) # notice batch size 1
    x_gray = (kornia.color.rgb_to_grayscale(tensor)) 
    x_laplacian: torch.cuda.HalfTensor = kornia.filters.canny(tensor)[0]   
    
    # convert to rgb for images
    x_lap_rgb = kornia.color.grayscale_to_rgb(x_laplacian)
    rgb_img = (1. - x_lap_rgb.clamp(0., 1.)) # only for plotting
    
    return rgb_img


"""
def torch_to_mask(tensor):
    # converts A and B Tensors -> Numpy Arr for cv2
    masks = []
    
    for m in range(0, tensor.size(0)):
        # takes pytorch tensor and converts to grayscale array for cv2
        gr = tensor[m, :, :, :].data.view(opt.img_width, opt.img_width, 3) # get the data in cv2 format

        gr = gr.detach().cpu().numpy() #turn into array 
        gr = gr.astype(np.uint8) # convert to UINT
        gr = cv2.cvtColor(gr, cv2.COLOR_BGR2GRAY)
        
        #make mask
        retval, thresh = cv2.threshold(gr, thresh=100, maxval=255, type=cv2.THRESH_BINARY)
        masks.append(thresh)
    
    return masks
    

def prep(img_A, img_B):

    thresh_gray_V = torch_to_mask(img_A)
    thresh_gray_T = torch_to_mask(img_B) # returns a list of arrays

    M1_Tensors = []
    M2_Tensors = []
    E1_Tensors = []
    E2_Tensors = []

    for i in range(0, len(thresh_gray_T)):
        
        # Calculating edges from masks for each array 
        e1 = cv2.Canny(thresh_gray_V[i],0,255)
        e2 = cv2.Canny(thresh_gray_V[i],0,255)

        # flatten edges
        ed1 = np.reshape(e1 > .5, (-1, e1.shape[-1])).astype(np.float32) # visible
        ed2 = np.reshape(e2 > .5, (-1, e2.shape[-1])).astype(np.float32) # thermal

        # flatten masks 
        masks1 = np.reshape(thresh_gray_V[i] > .5, (-1, thresh_gray_V[i].shape[-1])).astype(np.float32) # visible
        masks2 = np.reshape(thresh_gray_T[i] > .5, (-1, thresh_gray_T[i].shape[-1])).astype(np.float32) # thermal
    
        # If either set of masks is empty return empty result
        if masks1.shape[-1] == 0 or masks2.shape[-1] == 0:
            return np.zeros((masks1.shape[-1], masks2.shape[-1]))


       # Jaccard: convert back to PyTorch Tensor

        m1_tens = Variable(torch.from_numpy(masks1) ,requires_grad=True).cuda()
        m2_tens = Variable(torch.from_numpy(masks2), requires_grad=True).cuda()
        
        ed1_tens = Variable(torch.from_numpy(ed1), requires_grad=True).cuda()
        ed2_tens = Variable(torch.from_numpy(ed2), requires_grad=True).cuda()
        
        M1_Tensors.append(m1_tens)
        M2_Tensors.append(m2_tens)
        E1_Tensors.append(ed1_tens)
        E2_Tensors.append(ed2_tens)
              

    # return a tensor converted from a list of tensors
    M1 = torch.stack(M1_Tensors).cuda()
    M2 = torch.stack(M2_Tensors).cuda()
    E1 = torch.stack(E1_Tensors).cuda()
    E2 = torch.stack(E2_Tensors).cuda()
 
    return M1, M2, E1, E2
"""


def my_custom_iou(inputs, targets, smooth=1):
    #intersection is equivalent to True Positive count
    #union is the mutually inclusive area of all labels & predictions 
    intersection = (inputs * targets).sum()
    total = (inputs + targets).sum()
    union = total - intersection 

    IoU = (intersection + smooth)/(union + smooth)

    # 1 - b/c you want to maximize this IoU, 90% IoU is 0.10 Loss (min it)        
    loss = 1 - IoU
    return loss




# ===========================================================
# Initialize generator and discriminator
input_shape = (opt.channels, opt.img_height, opt.img_width)
generator = GeneratorUNet(input_shape)
discriminator = Discriminator(input_shape)

if cuda:
    generator = generator.cuda()
    discriminator = discriminator.cuda()
    model = Net().cuda()
    criterion_GAN.cuda()
    criterion_MSE.cuda()

    
generator = torch.nn.DataParallel(generator, device_ids=[0, 1])
discriminator = torch.nn.DataParallel(discriminator, device_ids=[0, 1])
model = torch.nn.DataParallel(model, device_ids=[0, 1])                       

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
#optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
#optimizer_M = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

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
        #optimizer_M.zero_grad()
        print("+ + + optimizer_G.zero_grad() + + +")   
        with autocast(): 
            # generate a fake
            fake_B = generator(real_A) # fake_B scale to A, use this fake_B as the concat pair to learn the deformation grid
            
            # sample the deformation grid on the src, real_B
            warped_B = model(img_A=real_A, img_B=fake_B, src=real_B) # requires the STN to output a well-aligned thermal (warped_B) to send to jaccard
            R_ = torch.cat(warped_B) # just converting to tensor: Registered
            
            # min MSE(XOR, EDGE_IOU)
            M1, M2, LAP1, LAP2 = prep(real_A, R_) # using the Registered B - need to max the IOU between the thermal and visible based on the registered thermal
            
            #print("M1:\n", M1[0])
            #print("E1:\n", E1[0])
            
            criterion_iou_masks = my_custom_iou(M1,M2)
            criterion_iou_edges = my_custom_iou(LAP1,LAP2)
            #criterion_iou_masks = criterion_MSE(M1,M2)
            #criterion_iou_edges = criterion_MSE(LAP1,LAP2)
   
            loss_iou = 0.5*(criterion_iou_masks + criterion_iou_edges) 
            
            print("loss_iou:", loss_iou)
        
            # Adverarial - How fake and how real
            # relativistic loss
            pred_real = discriminator(real_B, real_A) 
            pred_fake = discriminator(fake_B, real_A) 
            loss_GAN = criterion_GAN(pred_fake - pred_real.detach(), valid)
        
  
        scaler.scale(loss_GAN).backward() 
        scaler.scale(loss_iou).backward(retain_graph=True)  
        scaler.step(optimizer_G)
        #scaler.step(optimizer_M)
        print("+ + + optimizer_G.step() + + +")
        print("+ + + optimizer_M.step() + + +")
        
        # -----------------------
        #  Train Discriminator 
        # -----------------------

        optimizer_D.zero_grad()
        print("+ + + optimizer_D.zero_grad() + + +")
        with autocast():
            # real
            yhat_real = discriminator(real_B, real_A) 
            #fake 
            yhat_fake = discriminator(fake_B.detach(), real_A)
            #adv loss
            loss_real = criterion_GAN(yhat_real - yhat_fake, valid)
            loss_fake = criterion_GAN(yhat_fake - yhat_real, fake)
            
            loss_D = 0.5*(loss_real + loss_fake)

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
            "\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [GAN loss: %f, IOU: %f] ETA: %s"
            % (
                epoch, #%d
                opt.n_epochs, #%d
                i, #%d
                len(dataloader), #%d
                loss_D.item(), #%f
                loss_GAN.item(), #%f - total G loss
                loss_iou.item(),
                time_left, #%s
            )
        )
        
        f.write(
            "\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [GAN loss: %f, IOU: %f] ETA: %s"
            % (
                epoch, #%d
                opt.n_epochs, #%d
                i, #%d
                len(dataloader), #%d
                loss_D.item(), #%f
                loss_GAN.item(), #%f - total G loss
                loss_iou.item(),
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

                   