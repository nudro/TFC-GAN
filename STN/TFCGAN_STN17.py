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

Tries anisotropic loss from the medpy package of the real_A and registered_B, min the IOU 

only T = 5 steps 

Totally experimental - ViT as Localization Network 

No patches - global. 

More padding and adds back in the LPIPS loss. 

Tries circular padding and 'valid' https://rivesunder.github.io/SortaSota/ca/2021/01/05/toroid_universe.html

Uses the Fourier Transform as a L1 loss

Stacked STN Modules: like https://openaccess.thecvf.com/content_cvpr_2017/papers/Lin_Inverse_Compositional_Spatial_CVPR_2017_paper.pdf

Experiments of different modalities.

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
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=10, help="interval between sampling of images from generators")
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
    t = torch.Tensor(tensor).cuda()
    t = t.reshape(t.size(0), 1, t.size(1), t.size(2))
    t = t.expand(-1, 3, -1, -1) # needs to be [1, 3, 256, 256]
    return t
    
def sample_images(batches_done, t1_fu, t=100):
    imgs = next(iter(test_dataloader)) # batch_size = 1
    real_A = Variable(imgs["A"].type(HalfTensor))
    real_B = Variable(imgs["B"].type(HalfTensor))
    
    fake_B = generator(real_A)
    #AM, BM = prep(real_A, real_B) 

    # Register
    """ Given the theta, use it to warp the A and B? """
    B_R = model(img_A=real_A, img_B=fake_B, src=real_B, t=t, theta=t1_fu, mode='register')

    # MAGNITUDE SPECTRA
    #fb1_spec = sample_spectra(fake_B1.data)
    #b1_spec = sample_spectra(B1.data)
    #Rb1_spec = sample_spectra(B1_R.data)
    # cast from 1c -> 3 c RGB
    #fb1_spec, b1_spec, Rb1_spec = (fb1_spec.expand(-1, 3, -1, -1), b1_spec.expand(-1, 3, -1, -1), Rb1_spec.expand(-1, 3, -1, -1))
    #img_mag = torch.cat((fb1_spec.data, b1_spec.data, Rb1_spec.data), -1)
    #save_image(img_mag, "images/%s/%s_mag.png" % (opt.experiment, batches_done), nrow=5, normalize=True)
    
    # just trying upper right corner to see
    img_sample_global = torch.cat((real_A.data, real_B.data, B_R.data, fake_B.data), -1)
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

class LocalizerVIT1(nn.Module):
    def __init__(self, img_shape):
        super(LocalizerVIT1, self).__init__()
        channels, self.h, self.w = img_shape
        self.vit = nn.Sequential(
            K.VisionTransformer(image_size=self.h, patch_size=16, in_channels=channels*2) # (A,B), changed from 16-> 32 patch
        )
        
    def forward(self, x):
        with autocast():
            out = self.vit(x).type(HalfTensor) # returns torch.Size([batch, 257, 768])
        return out

class LocalizerVIT2(nn.Module):
    def __init__(self, img_shape):
        super(LocalizerVIT2, self).__init__()
        channels, self.h, self.w = img_shape
        self.vit = nn.Sequential(
            K.VisionTransformer(image_size=self.h, patch_size=16, in_channels=channels*2+1) # (A,B,THETA)
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

        self.localization1 = LocalizerVIT1(input_shape)
        self.localization2 = LocalizerVIT2(input_shape) # for t=1 to t=T, adds +1 channel
        
        self.theta_emb = nn.Linear(1, opt.img_height * opt.img_width)

        self.fc_loc = nn.Sequential(
            nn.Linear(1*257*768, 256), # (hard-coded in, 257, 768 based on ViT output)
            nn.ReLU(True),
            nn.Linear(256, 3*2))

        self.fc_loc[2].bias.data.zero_() # DO NOT CHANGE! the problem for everything is this, don't change it

       
    def stn_phi(self, x, mode):
        if mode=='init':
            xs = self.localization1(x) # (A,B), 6 ch
        elif mode=='rec':
            xs = self.localization2(x) #(A,B,theta), 7 ch
        
        xs = xs.view(-1, 1 * xs.size(1) * xs.size(2)) 
        theta = self.fc_loc(xs)  
        theta = theta.view(-1, 2, 3)
        return theta
    

    def forward(self, img_A, img_B, src, t, theta, mode='stack'):
        
        #angle = math.cos(math.radians(15))
        #angle = 1
        
        identity_matrix = [1, 0, 0, 0, 1, 0] #identity_theta = torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float).cuda()
        
        with autocast():  
            if mode=='stack':
                if t==0:
                    """ push it through the 1st time to the STN to output theta. will overwrite
                    a dummy theta that i pass through

                    The init theta is learned between the fake and the real vis"""
                    img_input = torch.cat((img_A, img_B), 1)
                    dtheta = self.stn_phi(img_input, mode='init') # deformation field for real A and real B

                    identity_theta = torch.tensor(identity_matrix, dtype=torch.float).cuda()
                    dtheta = dtheta.reshape(img_A.size(0), 2*3) #12,6 -> batch/3, 6
                    dtheta = dtheta + identity_theta.unsqueeze(0).repeat(img_A.size(0), 1) 
                    
                else: 
                    """ all other times, 1 - 9, do"""
                    # get each theta for the batch
                    # p has to be # torch.Size([12, 6]) -> batch/3, 6
                    theta_batches = []
                    for k in theta: # use the given theta for this timestep
                        this_theta = (k.view(-1, 2, 3)).reshape(1,2,3)
                        theta_batches.append(this_theta)

                    # just the tensors of the source image to be aligned
                    src_tensors = []
                    for img in src: 
                        this_img_src = img.reshape(1, img.size(0), img.size(1), img.size(2)) 
                        src_tensors.append(this_img_src)

                    # result
                    warped = []
                    for i in range(len(src_tensors)): #1:1 match with theta, matching with theta is important
                        rs_grid = F.affine_grid(theta_batches[i], src_tensors[i].size(), align_corners=True) 
                        # rs_grid is the attention mech of theta and A,B
                        """ Give Input Image real B (the image we want registered) and theta to the grid generator and sampler """
                        Rs = F.grid_sample(src_tensors[i], rs_grid,  mode='bicubic', padding_mode='border', align_corners=True) 
                        warped.append(Rs.type(HalfTensor))

                    # Now, send the registered B and real A to the STN (warped should be bs, 3, 128, 128)
                    theta = self.theta_emb(theta).view(theta.size(0), 1, img_A.size(2), img_A.size(3)) # the given theta, passed into the fxn (prev t-1)
                    img_input = torch.cat((img_A, img_B, theta), 1) # (12,3,128,128) + (12,1,128,128)
                    dtheta = self.stn_phi(img_input, mode='rec') # next t-step deformation field 
                    identity_theta = torch.tensor(identity_matrix, dtype=torch.float).cuda()
                    dtheta = dtheta.reshape(img_A.size(0), 2*3)
                    dtheta = dtheta + identity_theta.unsqueeze(0).repeat(img_A.size(0), 1) #12,6

                # is a tensor
                return dtheta

            elif mode=='register': # can be any t
                # get each theta for the batch
                # p has to be # torch.Size([12, 6]) -> batch/3, 6
                theta_batches = []
                for k in theta: # use the previous theta 
                    this_theta = (k.view(-1, 2, 3)).reshape(1,2,3)
                    theta_batches.append(this_theta)

                # just the tensors of the source image to be aligned
                src_tensors = []
                for img in src: 
                    this_img_src = img.reshape(1, img.size(0), img.size(1), img.size(2)) 
                    src_tensors.append(this_img_src)

                # result
                warped = []
                for i in range(len(src_tensors)): #1:1 match with theta, matching with theta is important
                    rs_grid = F.affine_grid(theta_batches[i], src_tensors[i].size(), align_corners=True) 
                    Rs = F.grid_sample(src_tensors[i], rs_grid,  mode='bicubic', padding_mode='border', align_corners=True) 
                    warped.append(Rs.type(HalfTensor))

                return torch.cat(warped) # converted to a tensor

            elif mode=='test':
                img_input = torch.cat((img_A, img_B), 1)
                dtheta = self.stn_phi(img_input, mode='init') 
                #identity_theta = torch.tensor(identity_matrix, dtype=torch.float).cuda()
                dtheta = dtheta.reshape(img_A.size(0), 2*3)
                #dtheta = dtheta + identity_theta.unsqueeze(0).repeat(img_A.size(0), 1)

                # get each theta for the batch
                theta_batches = []
                for t in dtheta:
                    this_theta = (t.view(-1, 2, 3)).reshape(1,2,3)
                    theta_batches.append(this_theta)

                # just the tensors of the source image to be aligned
                src_tensors = []
                for img in src: 
                    this_img_src = img.reshape(1, img.size(0), img.size(1), img.size(2)) 
                    src_tensors.append(this_img_src)

                # result
                warped = []
                for i in range(len(src_tensors)):
                    rs_grid = F.affine_grid(theta_batches[i], src_tensors[i].size(), align_corners=True) 
                    Rs = F.grid_sample(src_tensors[i], rs_grid,  mode='bicubic', padding_mode='border', align_corners=True)  
                    warped.append(Rs.type(HalfTensor))

                return torch.cat(warped)
                
    
##############################
# THETA FUSION
##############################

class Fusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(6, opt.batch_size) #(360,6) -> (6, 36) -> (36, 6)
        self.fc2 = nn.Linear(opt.batch_size, 6)

    def forward(self, x):
        # takes a series of thetas learned for a single patch, per batch
        # and learns the representation across the tensors
        x = self.fc1(x)
        out = self.fc2(x) #(120,6) Could it be that hte relu is messing up the affine values? 
        return out.type(HalfTensor)
    
    
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
# FFT
#####################

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

    if patch: 
        # reshape each amplitude and phase to Torch tensors <- Torch says I can do this faster?
        # For RFFT2 the dims is 256 x 129 for half of all real values + 1 col
        # ^^ above was for the full 256 x 256 image - the following is for the patch 128/2 + 1 col
        AMP_tensor = torch.cat(AMP).reshape(opt.batch_size, 1, opt.patch_height, 65)    
        PHA_tensor = torch.cat(PHA).reshape(opt.batch_size, 1, opt.patch_height, 65)    
    
    # not patch only for getting global 
    elif not patch:
        AMP_tensor = torch.cat(AMP).reshape(opt.batch_size, 1, opt.img_height, 129)    
        PHA_tensor = torch.cat(PHA).reshape(opt.batch_size, 1, opt.img_height, 129)
    
    return AMP_tensor, PHA_tensor
    
    
def sample_spectra(thermal_tensor):
    SPEC =[]
    for t in range(0, thermal_tensor.size(0)): 
            # thermal images must be in grayscale (1 channel)
            b = transforms.ToPILImage()(thermal_tensor[t, :, :, :]).convert("L")
            fft_space = FFT_Components(b)
            spectra = torch.Tensor(fft_space.make_spectra()).cuda() # convert to torch tensor
            SPEC.append(spectra)
            
    #SPEC_tensor = torch.cat(SPEC).reshape(thermal_tensor.size(0), 1, opt.img_height, opt.img_width)    
    SPEC_tensor = torch.cat(SPEC).reshape(thermal_tensor.size(0), 1, opt.patch_height, opt.patch_width)
    return SPEC_tensor

#####################
# MASKS
#####################

def prep(img_A, img_B):
    masks_V = Variable(torch.where(img_A > 0, 1.0, 0.), requires_grad=True).cuda()
    masks_T = Variable(torch.where(img_B > 0, 1.0, 0.), requires_grad=True).cuda() # torch.Size([b, 3, 256, 256])
    #lap1 = Variable(canny_edges(masks_V), requires_grad=True).cuda() # torch.Size([b, 1, 256, 256])
    #lap2 = Variable(canny_edges(masks_T), requires_grad=True).cuda()
    
    #return masks_V, masks_T, lap1, lap2
    return masks_V, masks_T

"""
def canny_edges(tensor):
    # tensor is [256,256] binary mask image
    x_rgb: torch.cuda.HalfTensor = kornia.color.bgr_to_rgb(tensor) # will add 3 channels
    tensor = tensor.reshape(x_rgb.size(0), 3, opt.patch_width, opt.patch_width) #[batch, 3, W, H)
    #tensor = tensor.reshape(x_rgb.size(0), 3, opt.img_width, opt.img_width) #[batch, 3, W, H)
    x_gray = (kornia.color.rgb_to_grayscale(tensor)) 
    x_laplacian: torch.cuda.HalfTensor = kornia.filters.canny(tensor)[0]    
    
    return x_laplacian


def canny_edges_test(tensor):
    x_rgb: torch.cuda.HalfTensor = kornia.color.bgr_to_rgb(tensor) # will add 3 channels
    #tensor = tensor.reshape(1,3,opt.img_width, opt.img_width) # notice batch size 1
    tensor = tensor.reshape(1,3,opt.patch_width, opt.patch_width) # notice batch size 1
    x_gray = (kornia.color.rgb_to_grayscale(tensor)) 
    x_laplacian: torch.cuda.HalfTensor = kornia.filters.canny(tensor)[0]   
    
    # convert to rgb for images
    x_lap_rgb = kornia.color.grayscale_to_rgb(x_laplacian)
    rgb_img = (1. - x_lap_rgb.clamp(0., 1.)) # only for plotting
    
    return rgb_img


def my_custom_iou(inputs, targets, smooth=1):
    #intersection is equivalent to True Positive count
    #union is the mutually inclusive area of all labels & predictions 
    # // not using this right now
    intersection = (inputs * targets).sum()
    total = (inputs + targets).sum()
    union = total - intersection 

    IoU = (intersection + smooth)/(union + smooth)

    # 1 - b/c you want to maximize this IoU, 90% IoU is 0.10 Loss (min it)        
    loss = 1 - IoU
    return loss

#def iou_patch_loss(A1, A2, A3, A4, B1_R, B2_R, B3_R, B4_R):
def iou_patch_loss(fake_B1, fake_B2, fake_B3, fake_B4, B1_R, B2_R, B3_R, B4_R):
    M1_1, M2_1, E1_1, E2_1 = prep(fake_B1, B1_R) 
    M1_2, M2_2, E1_2, E2_2 = prep(fake_B2, B2_R)
    M1_3, M2_3, E1_3, E2_3 = prep(fake_B3, B3_R)
    M1_4, M2_4, E1_4, E2_4 = prep(fake_B4, B4_R)

    loss_1 = criterion_L1(M1_1, M2_2)
    loss_2 = criterion_L1(M1_2, M2_2)
    loss_3 = criterion_L1(M1_3, M2_3)
    loss_4 = criterion_L1(M1_4, M2_4)
    
    # for now, return just the masks loss, not edges yet, let's see how this goes
    total_loss = (loss_1 + loss_2+ loss_3 + loss_4).mean() # is this the right way? 
    
    return total_loss
"""

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
          
    intersection = (aniso_R * aniso_A).sum()
    total = (aniso_R + aniso_A).sum()
    union = total - intersection 

    IoU = (intersection + smooth)/(union + smooth)

    # 1 - b/c you want to maximize this IoU, 90% IoU is 0.10 Loss (min it)        
    loss = 1 - IoU
    return loss


"""
def patch_fourier(B1_R, B2_R, B3_R, B4_R, fake_B1, fake_B2, fake_B3, fake_B4):
    A1f, P1f = fft_components(fake_B1) 
    A2f, P2f = fft_components(fake_B2)
    A3f, P3f = fft_components(fake_B3)
    A4f, P4f = fft_components(fake_B4)

    A1r, P1r = fft_components(B1_R)
    A2r, P2r = fft_components(B2_R)
    A3r, P3r = fft_components(B3_R)
    A4r, P4r = fft_components(B4_R)

    #loss_Amp = 0.25*(criterion_amp(A1f, A1r) + criterion_amp(A2f, A2r) + criterion_amp(A3f, A3r) + criterion_amp(A4f, A4r))
    loss_Pha = 0.25*(criterion_phase(P1f, P1r) + criterion_phase(P2f, P2r) + criterion_phase(P3f, P3r) + criterion_phase(P4f, P4r))
    #loss_FFT = 1/2*(loss_Amp + loss_Pha)

    #return loss_FFT
    return loss_Pha # typically contains more info than magnitude


def generator_patch_loss(A1, A2, A3, A4, B1, B2, B3, B4, fake_B1, fake_B2, fake_B3, fake_B4):
    pred_fake_B1 = discriminator(fake_B1, A1)
    real_pred_B1 = discriminator(A1, B1) 

    pred_fake_B2 = discriminator(fake_B2, A2)
    real_pred_B2 = discriminator(A2, B2) 

    pred_fake_B3 = discriminator(fake_B3, A3)
    real_pred_B3 = discriminator(A3, B3) 

    pred_fake_B4 = discriminator(fake_B4, A4)
    real_pred_B4 = discriminator(A4, B4) 

    loss_GAN_B1 = criterion_GAN(pred_fake_B1 - real_pred_B1.detach(), valid_p)
    loss_GAN_B2 = criterion_GAN(pred_fake_B2 - real_pred_B2.detach(), valid_p)
    loss_GAN_B3 = criterion_GAN(pred_fake_B3 - real_pred_B3.detach(), valid_p)
    loss_GAN_B4 = criterion_GAN(pred_fake_B4 - real_pred_B4.detach(), valid_p)
    loss_GAN_patch = 0.25*(loss_GAN_B1 + loss_GAN_B2 + loss_GAN_B3 + loss_GAN_B4)

    return loss_GAN_patch
    
    
def disc_patch_loss(A1, A2, A3, A4, B1, B2, B3, B4, fake_B1, fake_B2, fake_B3, fake_B4):
    # real
    pred_real_B1 = discriminator(B1, A1)
    pred_real_B2 = discriminator(B2, A2)
    pred_real_B3 = discriminator(B3, A3)
    pred_real_B4 = discriminator(B4, A4)

    # fake
    pred_fake_B1 = discriminator(fake_B1.detach(), A1)
    pred_fake_B2 = discriminator(fake_B2.detach(), A2)
    pred_fake_B3 = discriminator(fake_B3.detach(), A3)
    pred_fake_B4 = discriminator(fake_B4.detach(), A4)

    # adversarial losses
    loss_real_B1 = criterion_GAN(pred_real_B1 - pred_fake_B1, valid_p)
    loss_real_B2 = criterion_GAN(pred_real_B2 - pred_fake_B2, valid_p)
    loss_real_B3 = criterion_GAN(pred_real_B3 - pred_fake_B3, valid_p)
    loss_real_B4 = criterion_GAN(pred_real_B4 - pred_fake_B4, valid_p)
    loss_real_patch = 0.25*(loss_real_B1 + loss_real_B2 + loss_real_B3 + loss_real_B4)

    loss_fake_B1 = criterion_GAN(pred_fake_B1 - pred_real_B1, fake_p)
    loss_fake_B2 = criterion_GAN(pred_fake_B2 - pred_real_B2, fake_p)
    loss_fake_B3 = criterion_GAN(pred_fake_B3 - pred_real_B3, fake_p)
    loss_fake_B4 = criterion_GAN(pred_fake_B4 - pred_real_B4, fake_p)
    loss_fake_patch = 0.25*(loss_fake_B1 + loss_fake_B2 + loss_fake_B3 + loss_fake_B4)

    # total discriminator1 patch loss
    loss_D_patch = 0.5*(loss_real_patch + loss_fake_patch)

    return loss_D_patch
    
    
def pixel_patch_loss(B1, B2, B3, B4, fake_B1, fake_B2, fake_B3, fake_B4):
    loss_pix_B1 = criterion_lpips(fake_B1, B1)
    loss_pix_B2 = criterion_lpips(fake_B2, B2)
    loss_pix_B3 = criterion_lpips(fake_B3, B3)
    loss_pix_B4 = criterion_lpips(fake_B4, B4)
    loss_pix_patch = 0.25*(loss_pix_B1 + loss_pix_B2 + loss_pix_B3 + loss_pix_B4)

    return loss_pix_patch 
"""

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
    fusion = Fusion().cuda()
    
    criterion_GAN.cuda()
    criterion_lpips.cuda()
    criterion_L1.cuda()
    criterion_amp = nn.L1Loss()
    criterion_phase = nn.L1Loss()
    
generator = torch.nn.DataParallel(generator, device_ids=[0, 1, 2])
discriminator = torch.nn.DataParallel(discriminator, device_ids=[0, 1, 2])
model = torch.nn.DataParallel(model, device_ids=[0, 1, 2])  
fusion = torch.nn.DataParallel(fusion, device_ids=[0, 1, 2])

if opt.epoch != 0:
    # Load pretrained models
    generator.load_state_dict(torch.load("/home/local/AD/cordun1/experiments/TFC-GAN/saved_models/%s/generator_%d.pth" % (opt.experiment, opt.epoch)))
    discriminator.load_state_dict(torch.load("/home/local/AD/cordun1/experiments/TFC-GAN/saved_models/%s/discriminator_%d.pth" % (opt.experiment, opt.epoch)))
    model.load_state_dict(torch.load("/home/local/AD/cordun1/experiments/TFC-GAN/saved_models/%s/stn_%d.pth" % (opt.experiment, opt.epoch)))
    fusion.load_state_dict(torch.load("/home/local/AD/cordun1/experiments/TFC-GAN/saved_models/%s/fusion_%d.pth" % (opt.experiment, opt.epoch)))

else:
    # Initialize weights
    generator.apply(weights_init_normal)
    discriminator.apply(weights_init_normal)
    model.apply(weights_init_normal) 
    fusion.apply(weights_init_normal) 

# Optimizers - joint model, fusion, and G together
optimizer_G = torch.optim.Adam(itertools.chain(generator.parameters(), model.parameters(), fusion.parameters()), lr=opt.lr, betas=(opt.b1, opt.b2))
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
        #optimizer_M.zero_grad()
        print("+ + + optimizer_G.zero_grad() + + + ")   
        with autocast(): 
            # // Fake //
            fake_B = generator(real_A)
            
            # // Get Binary Masks //
            #AM, BM = prep(real_A, real_B) 

            # // Register Them //
            """ Essentially asking the STN to use the fake patch as the target, 
            so fake patch needs to get better in generation. For each batch of x samples,
            you will pass it through 10 stacks of STNs, passing the image + deformation 
            grid to the sampler which will then approximate the vlaues to shift the pixel
            values of the target (real B). This approximation vector and prior theta is
            sent to the UNET to then predict the next theta."""

            t=0
            # dummy vars to be overwritten at t=0
            theta1 = torch.rand(real_A.size(0), 6) # torch.Size([36, 6])
            th1_log = []
            
            T = 1
            for t in range(T):
                #print(t)
                # returns the theta for each patch 
                theta1 = model(img_A=real_A, img_B=fake_B, src=real_B, t=t, theta=theta1, mode='stack') 
                
                #print("------------ theta1 ------------")
                #print(theta1) # is this even changing?
                #print("--------------------------------")
                
                th1_log.append(theta1)
 
            # // Fusion //
            t1 = torch.cat(th1_log) # each is (360, 6): 10 x 36 (batch)
            t1_fu = fusion(t1)

            t=100 # dummy    
            # use the last learned theta learned from the prior timesteps:
            BR = model(img_A=real_A, img_B=fake_B, src=real_B, t=t, theta=t1_fu, mode='register')
            
            if i % 10 == 0: # something random
                # I want to see what it looks like ^^
                training_sample = torch.cat((real_A.data, real_B.data, BR.data, fake_B.data), -1)
                save_image(training_sample, "images/%s/%s_%s_TS.png" % (opt.experiment, i, epoch), nrow=4, normalize=True)
            
            # registration loss -> making the registered B get close to fake_B? 
            loss_IOU = iou_global_loss(real_A, BR)
            
            # LPIPS patch loss, reconstruction using the ground truth as the fake 
            loss_pixel = global_pixel_loss(real_B, fake_B)
            
            # Fourier Transform Loss for Each Patch
            #loss_FFT = global_fourier_loss(BR, fake_B) # registered and real
 
            # Adverarial - How fake and how real
            loss_GAN = global_gen_loss(real_A, real_B, fake_B)
            
            # Total Loss
            loss_G = loss_GAN + loss_IOU + loss_pixel

        scaler.scale(loss_G).backward()
        scaler.step(optimizer_G)
        #scaler.step(optimizer_M)
        print("+ + + optimizer_G.step() + + + ")
        
        # -----------------------
        #  Train Discriminator 
        # -----------------------

        optimizer_D.zero_grad()
        print("+ + + optimizer_D.zero_grad() + + +")
        with autocast():
            
            loss_D = global_disc_loss(real_A, real_B, fake_B)

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
            "\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, IOU: %f, pix: %f ] ETA: %s"
            % (
                epoch, #%d
                opt.n_epochs, #%d
                i, #%d
                len(dataloader), #%d
                loss_D.item(), #%f
                loss_G.item(), #%f - total G loss
                loss_IOU.item(),
                loss_pixel.item(),
                time_left, #%s
            )
        )
        
        f.write(
            "\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, IOU: %f, pix: %f] ETA: %s"
            % (
                epoch, #%d
                opt.n_epochs, #%d
                i, #%d
                len(dataloader), #%d
                loss_D.item(), #%f
                loss_G.item(), #%f - total G loss
                loss_IOU.item(),
                loss_pixel.item(),
                time_left, #%s
            )
        )
        
        # If at sample interval save image
        #if batches_done % opt.sample_interval == 0:
        #    sample_images(batches_done, t1_fu) # give the patch thetas to sample 


    if opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0:
        # Save model checkpoints
        torch.save(generator.state_dict(), "/home/local/AD/cordun1/experiments/TFC-GAN/saved_models/%s/generator_%d.pth" % (opt.experiment, epoch))
        torch.save(discriminator.state_dict(), "/home/local/AD/cordun1/experiments/TFC-GAN/saved_models/%s/discriminator_%d.pth" % (opt.experiment, epoch))
        torch.save(model.state_dict(), "/home/local/AD/cordun1/experiments/TFC-GAN/saved_models/%s/stn_%d.pth" % (opt.experiment, epoch))
        torch.save(fusion.state_dict(), "/home/local/AD/cordun1/experiments/TFC-GAN/saved_models/%s/fusion_%d.pth" % (opt.experiment, epoch))

                   