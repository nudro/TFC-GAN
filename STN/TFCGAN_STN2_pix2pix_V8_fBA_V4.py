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
from datasets_temp_B import * # Loads A1:A4 & B1:B4 patches

"""
Corrected version of V3

Mapping for the registered B aligns with a theta generated from 
img_A = fake_B
img_B = real_A

Swapped from the original V8 experiment 

Does not generate a fake_A - doesn't need it
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
criterion_mesh = nn.TripletMarginLoss(margin=1.0, p=2)
# LPIPS for reconstruction loss
criterion_lpips = LPIPS(
    net_type='vgg',  # choose a network type from ['alex', 'squeeze', 'vgg']
    version='0.1'  # Currently, v0.1 is supported
)

triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2)
# Fourier Amplitude and Phase Losses
#criterion_amp = nn.TripletMarginLoss(margin=1.0, p=2)
#criterion_pha = nn.TripletMarginLoss(margin=1.0, p=2)

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

    fake_B = generator(real_A) 
    warped_B = model(img_A=fake_B, img_B=real_A, src=real_B)
    R_ = torch.cat(warped_B)

    img_sample_global = torch.cat((real_A.data, real_B.data, fake_B.data, R_.data), -2)
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
            img_input = torch.cat((img_A, img_B), 1)
            dtheta = self.stn_phi(img_input) # deformation field for real A and real B
            identity_theta = torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float).cuda()
            dtheta = dtheta.reshape(img_B.size(0), 2*3)
            theta = dtheta + identity_theta.unsqueeze(0).repeat(img_B.size(0), 1)
    
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
    
    
    
# ===========================================================
# Initialize generator and discriminator
input_shape = (opt.channels, opt.img_height, opt.img_width)

generator = GeneratorUNet(input_shape)
discriminator = Discriminator(input_shape)

if cuda:
    generator = generator.cuda()
    discriminator = discriminator.cuda()
    criterion_GAN.cuda()
    criterion_lpips.cuda()
    triplet_loss.cuda()
    #criterion_amp.cuda()
    #criterion_pha.cuda()
    criterion_mesh.cuda()
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
    model.apply(weights_init_normal) # Need this for the STN UNET and Affine

# Optimizers
optimizer_G = torch.optim.Adam(itertools.chain(generator.parameters(), model.parameters()), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

# Tensor type - only use HalfTensor in this AMP script                    
Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor
HalfTensor = torch.cuda.HalfTensor if cuda else torch.HalfTensor

#+++++++ Initialize face mesh
mp_face_mesh = mp.solutions.face_mesh
# Initialize MediaPipe Face Mesh.
my_face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.3)
 
    
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

######################
# Calc Losses per generator pass
######################
 
"""
class FFT_Components(object):

    def __init__(self, image):
        self.image = image
   
    def make_components(self):
        img = np.array(self.image) #turn into numpy
        f_result = np.fft.rfft2(img) 
        fshift = np.fft.fftshift(f_result)
        amp = np.abs(fshift)
        phase = np.arctan2(fshift.imag,fshift.real)
        return amp, phase
    
    def make_spectra(self):
        img = np.array(self.image) #turn into numpy
        f_result = np.fft.fft2(img) # setting this to regular FFT2 to make magnitude spectra
        fshift = np.fft.fftshift(f_result)
        magnitude_spectrum = np.log(np.abs(fshift)) 
        return magnitude_spectrum
        
    
    
def fft_components(thermal_tensor, patch=True):
    # thermal_tensor can be fake_B or real_B
    #print("thermal tensor shape:", thermal_tensor.size())
    AMP = []
    PHA = []
    for t in range(0, opt.batch_size): 
        # thermal images must be in grayscale (1 channel)
        b = transforms.ToPILImage()(thermal_tensor[t, :, :, :]).convert("L")
        fft_space = FFT_Components(b)
        amp, phase = torch.Tensor(fft_space.make_components()).cuda() # convert them into torch tensors
        #print("Shape of amp tensor:", amp.size())
        AMP.append(amp)
        PHA.append(phase)

    # reshape each amplitude and phase to Torch tensors <- Torch says I can do this faster?
    # For RFFT2 the dims is 256 x 129 for half of all real values + 1 col
    # ^^ above was for the full 256 x 256 image - the following is for the patch 128/2 + 1 col
    AMP_tensor = torch.cat(AMP).reshape(opt.batch_size, 1, opt.patch_height, 65)    
    PHA_tensor = torch.cat(PHA).reshape(opt.batch_size, 1, opt.patch_height, 65)    
    
    # not patch only for getting global 
    if not patch:
        AMP_tensor = torch.cat(AMP).reshape(opt.batch_size, 1, opt.img_height, 129)    
        PHA_tensor = torch.cat(PHA).reshape(opt.batch_size, 1, opt.img_height, 129)
    
    return AMP_tensor, PHA_tensor
"""    
    
##############################    
def triplet_patches(fake_img, P1, P2, P3, P4):
    """
    Calculate a triplet structural loss and triplet Fourier loss
    using either fake_A or fake_B and its associated patches
    """

    # triplet loss on the patches of fake_img which can be fake_A or fake_B
    fake_P1 = fake_img[:, :, 0:0+opt.img_width//2, 0:0+opt.img_height//2] #(x,y) = (0,0)
    fake_P2 = fake_img[:, :, 0:0+opt.img_width//2, 128:128+opt.img_height//2] #(x,y) = (0, 128)
    fake_P3 = fake_img[:, :, 128:128+opt.img_width//2, 0:0+opt.img_height//2] #(x,y)=(128,0)
    fake_P4 = fake_img[:, :, 128:128+opt.img_width//2, 128:128+opt.img_height//2] #(x,y) = (128,128)

    """
    #>>Fourier Transform Loss for Each Patch
    A1f, P1f = fft_components(fake_P1) 
    A2f, P2f = fft_components(fake_P2)
    A3f, P3f = fft_components(fake_P3)
    A4f, P4f = fft_components(fake_P4)

    A1r, P1r = fft_components(P1)
    A2r, P2r = fft_components(P2)
    A3r, P3r = fft_components(P3)
    A4r, P4r = fft_components(P4)
    """

    # Here I randomize the negatives
    random_patches = torch.stack([P1, P2, P3, P4])
    patch_num = 4
    K1 = random_patches[np.random.randint(patch_num, size=1).item()]
    K2 = random_patches[np.random.randint(patch_num, size=1).item()]
    K3 = random_patches[np.random.randint(patch_num, size=1).item()]
    K4 = random_patches[np.random.randint(patch_num, size=1).item()]
    
    """
    A1K, P1K = fft_components(K1)
    A2K, P2K = fft_components(K2)
    A3K, P3K = fft_components(K3)
    A4K, P4K = fft_components(K4)

    P1_trip_amp = criterion_amp(A1f,A1r,A1K)
    P2_trip_amp = criterion_amp(A2f,A2r,A2K)
    P3_trip_amp = criterion_amp(A3f,A3r,A3K)
    P4_trip_amp = criterion_amp(A4f,A4r,A4K)
    Amp_loss = 1/4*(P1_trip_amp + P2_trip_amp + P3_trip_amp + P4_trip_amp)

    P1_trip_pha = criterion_pha(P1f,P1r,P1K)
    P2_trip_pha = criterion_pha(P2f,P2r,P2K)
    P3_trip_pha = criterion_pha(P3f,P3r,P3K)
    P4_trip_pha = criterion_pha(P4f,P4r,P4K)
    Pha_loss = 1/4*(P1_trip_pha + P2_trip_pha + P3_trip_pha + P4_trip_pha)
    
    FFT_loss = 1/2*(Amp_loss+Pha_loss)
    """

    # randomize the negatives
    P1_trip_loss = triplet_loss(fake_P1, P1, K1)
    P2_trip_loss = triplet_loss(fake_P2, P2, K2)
    P3_trip_loss = triplet_loss(fake_P3, P3, K3)
    P4_trip_loss = triplet_loss(fake_P4, P4, K4)
    
    Patch_loss = 0.25*(P1_trip_loss + P2_trip_loss + P3_trip_loss + P4_trip_loss)

    #return FFT_loss, Patch_loss
    return Patch_loss


################### Face Mesh ##################

def key_maker(keys_list):
    """
    [{'x': 0.8356158137321472, 'y': 0.5726576447486877, 'z': -0.04518777132034302}
     {'x': 0.8393599390983582, 'y': 0.526944637298584, 'z': -0.08034992218017578}
     {'x': 0.8377288579940796, 'y': 0.5413321852684021, 'z': -0.04345175251364708}
     {'x': 0.8282697796821594, 'y': 0.48486924171447754, 'z': -0.05877460911870003}
     {'x': 0.8400776982307434, 'y': 0.5136213302612305, 'z': -0.0849500373005867}
     {'x': 0.8407394886016846, 'y': 0.49684014916419983, 'z': -0.07834245264530182}
    """
   
    outside = []
    for i in range(0, len(keys_list)):
        inside = []
        foo = keys_list[i][0]
        for i in range(0, len(foo)):
            inside.append(list(foo[i].values()))
        outside.append(inside)
    arr = np.array(outside)
    print(arr.shape)
    return arr
    
    
def face_mesh(face_tensor, face_mesh=my_face_mesh):
    all_keypoints = []
    indices = []
  
    for m in range(0, face_tensor.size(0)): # use face_tensor.size(0) b/c batch size is spread over multiple GPUs
        # save torch tensor to image
        face = face_tensor[m, :, :, :].data.view(256,256,3) # get the data in cv2 format
        face = face.detach().cpu().numpy() #turn into array 
        face = face.astype(np.float32) # if you don't cast to fp32, norm won't work
        # ensure dtype is the same as array going in
        face_norm = cv2.normalize(face, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F) 
        face_norm = face_norm.astype(np.uint8) # convert to UINT
        results = face_mesh.process(cv2.cvtColor(face_norm, cv2.COLOR_BGR2RGB))
        
        if not results.multi_face_landmarks:
            #print("No Mesh Found.")
            keypoints = np.zeros([468,3]) # dummy variable
            my_keypoints = np.array(list(keypoints))
            all_keypoints.append(my_keypoints)
            indices.append(m)
            # indices keep track of the images that have no faces detected, we will throw those out 
            # and not count it towards the loss
            
        else:
            #print("Mesh found.")
            #face_files.append(os.path.basename(item))
            landmarks = results.multi_face_landmarks
  
            # we want the dict values x, y, z into an array
            keypoints = protobuf_to_dict(landmarks[0]) ##iterate for a dictionary <- # Ref: https://github.com/google/mediapipe/issues/1020
            my_keypoints = np.array(list(keypoints.values()))  #grab the values and convert to an array      
            all_keypoints.append(my_keypoints)
            
    return all_keypoints, indices


def mesh_loss(keysA, idA, keysRB, idB, keys_realB, id_realB):
    """
    Takes the keys and indices where there were missing faces. 
    Doesn't count those towards the loss.
    
    Drop the indices where there are no faces, for both A and B. 
    # [0, 3, 4, 5, 6, 8, 10, 11, 12, 13, 18, 21, 24, 27, 28, 29, 31, 32, 34, 35]
    # gets all of them in one sweep
    # for each of these, drop them out because these are duds
    # have to pop off the same index for each, otherwise it can't be compared
    
    keysA, idA = face_mesh(real_A)
    loss_mesh = mesh_loss(keysA, idA, keysRB, idB)
    
    You have to get the union between A, RB, and RealB, otherwise:
    A and RB Indices: [0, 1, 3, 4, 5, 6, 8, 10, 11, 12, 18, 20, 21, 22, 23, 24, 26, 27, 32, 39, 40, 41, 42, 43, 44, 45, 46, 48, 49, 51, 52, 53, 58, 59]
    A and real B Indices: [1, 2, 4, 5, 6, 7, 8, 9, 13, 15, 16, 17, 21, 23, 24, 27, 33, 34, 35, 37, 39, 41, 42, 43, 44, 46, 47, 48, 51, 57, 59]
    
    """

    indices = list(set(idA).union(idB)) #Assumption - only do this for the registered B id b/c the thermal face will be found on either
    indices2 = list(set(indices).union(id_realB)) # The union of A+RB no faces + real_B no faces

    
    # have to do all at once
    keysA_f = [i for j, i in enumerate(keysA) if j not in indices2] # list
    keysRB_f = [i for j, i in enumerate(keysRB) if j not in indices2] # list
    keys_realB_f = [i for j, i in enumerate(keys_realB) if j not in indices2] # list
    
    # convert list of arrays to torch tensors 
    RB = key_maker(keysRB_f) # extract all values
    RB_tensor = torch.from_numpy(RB)
    
    AK = key_maker(keysA_f)
    A_tensor = torch.from_numpy(AK)
    
    GTB = key_maker(keys_realB_f)
    GTB_tensor = torch.from_numpy(GTB)
    
    #9/26 changed to a triplet loss
    # anchor - registered B face mesh
    # positive - make reg B face mesh match visible mesh
    # negative - make it far away from ground truth B mesh 
    loss = criterion_mesh(RB_tensor, A_tensor, GTB_tensor) 
    
    return loss
    
    
##############################
#       Training
##############################

prev_time = time.time()
f = open('/home/local/AD/cordun1/experiments/TFC-GAN/LOGS/{}.txt'.format(opt.experiment), 'a+')

# Try AMP
scaler = GradScaler()

for epoch in range(opt.epoch, opt.n_epochs):
    for i, batch in enumerate(dataloader):
        print("****** {} ******\n".format(i))
        real_A = Variable(batch["A"].type(HalfTensor))
        real_B = Variable(batch["B"].type(HalfTensor))

        B1 = Variable(batch["B1"].type(HalfTensor))
        B2 = Variable(batch["B2"].type(HalfTensor))
        B3 = Variable(batch["B3"].type(HalfTensor))
        B4 = Variable(batch["B4"].type(HalfTensor))

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
            
            fake_B = generator(real_A) 
            loss_triplet_patch = triplet_patches(fake_B, B1, B2, B3, B4)
            # calc LPIPS
            loss_pixelB = criterion_lpips(fake_B, real_B)
            
            warped_B = model(img_A=fake_B, img_B=real_A, src=real_B)
            R_ = torch.cat(warped_B) # tensor of the registered real_B
            
            #Face Mesh Loss
            print("========A=======")
            keysA, idA = face_mesh(real_A) # keypoints A     
            print("=======Reg B=======")
            keysRB, idB = face_mesh(R_) # keypoints registered fake_B
            print("========B=======")
            keys_realB, id_realB = face_mesh(real_B) # keypoints B
            
            loss_mesh = mesh_loss(keysA, idA, 
                                  keysRB, idB, 
                                  keys_realB, id_realB) # calc loss between both landmarks
            
            # Adverarial - How fake and how real
            # relativistic loss
            pred_real = discriminator(real_B, real_A) 
            pred_fake = discriminator(fake_B, real_A)
            
            loss_GAN = criterion_GAN(pred_fake - pred_real.detach(), valid)
        
            # Total Loss
            loss_G = 0.5*(loss_GAN + loss_triplet_patch + 1/10*loss_pixelB + 100*loss_mesh)

        scaler.scale(loss_G).backward()
        scaler.step(optimizer_G)
        print("+ + + optimizer_G.step() + + +")
        
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
            "\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, trip: %f, GAN: %f, pix: %f, mesh: %f] ETA: %s"
            % (
                epoch, #%d
                opt.n_epochs, #%d
                i, #%d
                len(dataloader), #%d
                loss_D.item(), #%f
                loss_G.item(), #%f - total G loss
                #loss_fft.item(),
                loss_triplet_patch.item(),
                loss_GAN.item(), #%f - adv G loss
                loss_pixelB.item(),
                loss_mesh.item(),
                time_left, #%s
            )
        )
        
        f.write(
            "\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, trip: %f, GAN: %f, pix: %f, mesh: %f] ETA: %s"
            % (
                epoch, #%d
                opt.n_epochs, #%d
                i, #%d
                len(dataloader), #%d
                loss_D.item(), #%f
                loss_G.item(), #%f - total G loss
                #loss_fft.item(),
                loss_triplet_patch.item(),
                loss_GAN.item(), #%f - adv G loss
                loss_pixelB.item(),
                loss_mesh.item(),
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

                   