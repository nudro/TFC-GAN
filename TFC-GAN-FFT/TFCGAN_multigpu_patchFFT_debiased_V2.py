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
import antialiased_cnns
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from datasets_temp_Debias import *

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--dataset_name", type=str, default="facades", help="name of the dataset")
parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_height", type=int, default=256, help="size of image height")
parser.add_argument("--img_width", type=int, default=256, help="size of image width")
parser.add_argument("--patch_height", type=int, default=128, help="size of patch height")
parser.add_argument("--patch_width", type=int, default=128, help="size of patch width")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=500, help="interval between sampling of images from generators")
parser.add_argument("--checkpoint_interval", type=int, default=100, help="interval between model checkpoints")
parser.add_argument("--gpu_num", type=int, default=0, help="as of 1/8/22 never set this to 2")
parser.add_argument("--out_file", type=str, default="out", help="name of output log files")
parser.add_argument("--experiment", type=str, default="none", help="experiment name")
parser.add_argument("--annots_csv", type=str, default="none", help="csv file path for train labels")
parser.add_argument("--test_annots_csv", type=str, default="none", help="csv file path for test labels")
opt = parser.parse_args()

"""
Experiment: Adds Fourier GAN Loss
Experimental debiased version where I pass labels to the generator.
Discriminator outputs auxiliary classifiers for each gender, age, ethnicity label.

V2 - Simplified with fewer losses

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

# Adversarial
criterion_GAN = torch.nn.BCEWithLogitsLoss()

# Label Loss
criterion_label = torch.nn.CrossEntropyLoss()

# LPIPS
criterion_lpips = LPIPS(
    net_type='vgg',  # choose a network type from ['alex', 'squeeze', 'vgg']
    version='0.1'  # Currently, v0.1 is supported
)
# Patch Triplet Loss
triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2)
# scaling param for temp loss since value will be very small
lambda_t = 10

# Temp Triplet Loss
criterion_temp = nn.TripletMarginLoss(margin=1.0, p=2)

# Fourier Amplitude and Phase Losses
criterion_amp = nn.L1Loss()
criterion_phase = nn.L1Loss()

################
# Discr. Patches
#################

# Calculate output of image discriminator (PatchGAN)
# Not to be confused with the patches (the quadrants/crops) of the global 256 x 256 image
#patch_for_patches = (1, opt.patch_height // 2 ** 4, opt.patch_width // 2 ** 4)
#print("discriminator patch for patches:", patch_for_patches) #(1, 8, 8) = N X N

patch_for_g = (1, opt.img_height // 2 ** 4, opt.img_width // 2 ** 4)
print("discriminator patch for global:", patch_for_g) # (1, 16, 16) = N X N

##############################
#           U-NET
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

class GeneratorUNet(nn.Module):
    def __init__(self, img_shape):
        super(GeneratorUNet, self).__init__()
        channels, self.h, self.w = img_shape
        
        # Add a FC layer that will accept labels
        # matmul [batch, 3] input labels [3, 256*25]
        self.fc = nn.Linear(3, self.h * self.w)
        
        self.down1 = UNetDown(channels+1, 64, normalize=False) # 3RGB channels + labels
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

    def forward(self, x, labels):
        with autocast():
            labels = self.fc(labels).view(labels.size(0), 1, self.h, self.w)
            d1 = self.down1(torch.cat((x, labels), 1))
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
#        Discriminator1
# Uses Spectral Norm instead of Instance Norm
##############################

class Discriminator1(nn.Module):
    def __init__(self, img_shape):
        super(Discriminator1, self).__init__()
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
        
        # Auxiliary classifier layer for each label
        self.aux_gender = nn.Sequential(nn.Linear((channels * 2) * self.h * self.w, 2), nn.Softmax()) # 2 classes for gender
        self.aux_ethn = nn.Sequential(nn.Linear((channels * 2) * self.h * self.w, 4), nn.Softmax()) # 4 for ethnicity
        self.aux_age = nn.Sequential(nn.Linear((channels * 2) * self.h * self.w, 3), nn.Softmax()) # 3 for age

    def forward(self, img_A, img_B):
        with autocast():
            img_input = torch.cat((img_A, img_B), 1)
            output = self.model(img_input).type(HalfTensor)
            # Label Prediction 
            out = img_input.view(img_input.shape[0], -1) # flatten image to 2D
            
            gender_hat = self.aux_gender(out)
            ethn_hat = self.aux_ethn(out)
            age_hat = self.aux_age(out)

        return output, gender_hat, ethn_hat, age_hat


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


def label_formatter(labels):
    # for the real labels, not the predictions
    if opt.batch_size > 1:
        labels = labels.type(torch.LongTensor)
        labels = labels.squeeze_()
        labels = labels.to(device='cuda')
    elif opt.batch_size ==1:
        labels = labels.type(torch.LongTensor)
        labels = labels.squeeze_(dim=0) # batch size must be torch.size[1] by 0 dim
        labels = labels.to(device='cuda')
        
    #print("label formatter returns a label size of:", labels.size())
    return labels
    
        
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
    
    
def sample_spectra(thermal_tensor):
    SPEC =[]
    for t in range(0, thermal_tensor.size(0)): 
            # thermal images must be in grayscale (1 channel)
            b = transforms.ToPILImage()(thermal_tensor[t, :, :, :]).convert("L")
            fft_space = FFT_Components(b)
            spectra = torch.Tensor(fft_space.make_spectra()).cuda() # convert to torch tensor
            SPEC.append(spectra)
            
    SPEC_tensor = torch.cat(SPEC).reshape(thermal_tensor.size(0), 1, opt.img_height, opt.img_width)    
    
    return SPEC_tensor
    
    
def sample_images(batches_done):
    imgs = next(iter(test_dataloader))
    real_A = Variable(imgs["A"].type(HalfTensor))
    real_B = Variable(imgs["B"].type(HalfTensor))
    labels = Variable(imgs["LAB"].type(HalfTensor))
    fake_B = generator(real_A, labels)
    
    #fake_B1 = fake_B[:, :, 0:0+opt.img_width//2, 0:0+opt.img_height//2] #(x,y) = (0,0)
    #fake_B2 = fake_B[:, :, 0:0+opt.img_width//2, 128:128+opt.img_height//2] #(x,y) = (0, 128)
    #fake_B3 = fake_B[:, :, 128:128+opt.img_width//2, 0:0+opt.img_height//2] #(x,y)=(128,0)
    #fake_B4 = fake_B[:, :, 128:128+opt.img_width//2, 128:128+opt.img_height//2] #(x,y) = (128,128)

    # SAVE PATCHES
    #img_sample_patch = torch.cat((fake_B1.data, fake_B2.data, fake_B3.data, fake_B4.data), -2)
    #save_image(img_sample_patch, "images/%s/%s_p.png" % (opt.experiment, batches_done), nrow=5, normalize=True)
    
    # MAGNITUDE SPECTRA
    fake_spec = sample_spectra(fake_B.data)
    real_spec = sample_spectra(real_B.data)
    fake_spec, real_spec = (fake_spec.expand(-1, 3, -1, -1)), (real_spec.expand(-1, 3, -1, -1))
    img_mag = torch.cat((fake_spec.data, real_spec.data), -2)
    save_image(img_mag, "images/%s/%s_mag.png" % (opt.experiment, batches_done), nrow=5, normalize=True)
    
    # GLOBAL
    img_sample_global = torch.cat((real_A.data, fake_B.data, real_B.data), -2)
    save_image(img_sample_global, "images/%s/%s_g.png" % (opt.experiment, batches_done), nrow=5, normalize=True)
    
    

##############################
#       Initialize
##############################

# Initialize generator and discriminator
input_shape_global = (opt.channels, opt.img_height, opt.img_width)

generator = GeneratorUNet(input_shape_global)
discriminator1 = Discriminator1(input_shape_global)

generator = generator.cuda()
discriminator1 = discriminator1.cuda()

criterion_GAN.cuda()
criterion_label.cuda()
criterion_lpips.cuda()
triplet_loss.cuda()
criterion_temp.cuda()
criterion_amp.cuda()
criterion_phase.cuda()

################    
# nn.DataParallel
################

generator = torch.nn.DataParallel(generator, device_ids=[0,1,2])
discriminator1 = torch.nn.DataParallel(discriminator1, device_ids=[0,1,2])

if opt.epoch != 0:
    # Load pretrained models /home/local/AD/cordun1/experiments/faPVTgan/saved_models/0209_devcom_TripTemp
    generator.load_state_dict(torch.load("/home/local/AD/cordun1/experiments/TFC-GAN/saved_models/%s/generator_%d.pth" % (opt.experiment, opt.epoch)))
    discriminator1.load_state_dict(torch.load("/home/local/AD/cordun1/experiments/TFC-GAN/saved_models/%s/discriminator1_%d.pth" % (opt.experiment, opt.epoch)))

else:
    # Initialize weights
    generator.apply(weights_init_normal)
    discriminator1.apply(weights_init_normal)


################    
# Optimizers
################
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator1.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))


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

#Notice now there is an annots_csv that references the file list with the age, gender, ethn labels
dataloader = DataLoader(
    ImageDataset(root = "/home/local/AD/cordun1/experiments/data/%s" % opt.dataset_name,
                 annots_csv = opt.annots_csv,
                 transforms_=transforms_),
    batch_size=opt.batch_size,
    shuffle=True,
    num_workers=opt.n_cpu,
    drop_last=True,
)

test_dataloader = DataLoader(
    TestImageDataset(root = "/home/local/AD/cordun1/experiments/data/%s" % opt.dataset_name,
                     annots_csv = opt.test_annots_csv,
                     transforms_=transforms_,
        mode="test"),
    batch_size=1,
    shuffle=True,
    num_workers=1,
)

# Tensor type - only use HalfTensor in this AMP script
HalfTensor = torch.cuda.HalfTensor if cuda else torch.HalfTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor # for labels

##############################
#       Training
##############################

prev_time = time.time()
f = open('/home/local/AD/cordun1/experiments/TFC-GAN/LOGS/{}.txt'.format(opt.out_file), 'a+')
#print_network(generator, 'generator')
#print_network(discriminator1, 'discriminator1')

# Try AMP
scaler = GradScaler()
for epoch in range(opt.epoch, opt.n_epochs):
    for i, batch in enumerate(dataloader):
        real_A = Variable(batch["A"].type(HalfTensor))
        real_B = Variable(batch["B"].type(HalfTensor))
        B1 = Variable(batch["B1"].type(HalfTensor))
        B2 = Variable(batch["B2"].type(HalfTensor))
        B3 = Variable(batch["B3"].type(HalfTensor))
        B4 = Variable(batch["B4"].type(HalfTensor))
        TB = Variable(batch["T_B"].type(HalfTensor))
        labels = Variable(batch["LAB"].type(HalfTensor))
        
        # Adversarial ground truths: global image
        valid_ones = Variable(HalfTensor(np.ones((real_A.size(0), *patch_for_g))), requires_grad=False)
        valid = valid_ones.fill_(0.9)
        fake = Variable(HalfTensor(np.zeros((real_A.size(0), *patch_for_g))), requires_grad=False)

        # ------------------
        #  Train Generators
        # ------------------
        """
        I know in favtgan, I passed fake labels as noise; 
        this time I'm going to pass the real labels and see what happens G(A, labels)
        https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/acgan/acgan.py
        """

        print("+ + + optimizer_G.zero_grad() + + +")
        optimizer_G.zero_grad()

        with autocast(): 
            
            # Need to pass generator noisy labels - do they need to be OHEd?
            gen_gender = Variable(HalfTensor(np.random.randint(0, 2, (opt.batch_size, 1))))
            gen_ethn = Variable(HalfTensor(np.random.randint(0, 4, (opt.batch_size, 1))))
            gen_age = Variable(HalfTensor(np.random.randint(0, 3, (opt.batch_size, 1))))
            gen_labels = torch.cat((gen_gender, gen_ethn, gen_age), dim=1)
            
            # why can't I give it the real label? 
            fake_B = generator(real_A, labels) # needs to go in as floats to the Generator

            #>> Adverarial - How fake and how real
            # relativistic loss
            pred_fake, gen_f, eth_f, age_f = discriminator1(fake_B, real_A) # use these for label loss
            real_pred, gen_r, eth_r, age_r = discriminator1(real_B, real_A) # I don't use these label preds
            loss_GAN_g = criterion_GAN(pred_fake - real_pred.detach(), valid) # GAN loss (fake, valid)

            # >> Label Loss
            # but then gen labels need to be converted back to flat longs for the CE loss
            # these will be random every time, meaning the label loss will never have a chance to converge
            #gen_gender = label_formatter(gen_gender)
            #gen_ethn = label_formatter(gen_ethn)
            #gen_age = label_formatter(gen_age)
            
            gender = label_formatter(labels[:, 0])
            ethn = label_formatter(labels[:, 1])
            age = label_formatter(labels[:, 2])
            loss_label = criterion_label(gen_f, gender) + criterion_label(eth_f, ethn) + criterion_label(age_f, age)
      
            
            #>>Triplet - Structural integrity 
            # triplet loss on the patches of fake_B
            fake_B1 = fake_B[:, :, 0:0+opt.img_width//2, 0:0+opt.img_height//2] #(x,y) = (0,0)
            fake_B2 = fake_B[:, :, 0:0+opt.img_width//2, 128:128+opt.img_height//2] #(x,y) = (0, 128)
            fake_B3 = fake_B[:, :, 128:128+opt.img_width//2, 0:0+opt.img_height//2] #(x,y)=(128,0)
            fake_B4 = fake_B[:, :, 128:128+opt.img_width//2, 128:128+opt.img_height//2] #(x,y) = (128,128)
            """
            # Here I randomize the negatives
            random_patches = torch.stack([B1, B2, B3, B4])
            patch_num = 4
            # randomize the negatives
            B1_trip_loss = triplet_loss(fake_B1, B1, random_patches[np.random.randint(patch_num, size=1).item()])
            B2_trip_loss = triplet_loss(fake_B2, B2, random_patches[np.random.randint(patch_num, size=1).item()])
            B3_trip_loss = triplet_loss(fake_B3, B3, random_patches[np.random.randint(patch_num, size=1).item()])
            B4_trip_loss = triplet_loss(fake_B4, B4, random_patches[np.random.randint(patch_num, size=1).item()])
            loss_triplet_patch = 0.25*(B1_trip_loss + B2_trip_loss + B3_trip_loss + B4_trip_loss)
            """
            
            #>>Temperature loss using Triplet Loss
            # fake_B temps
            TFB_ = vectorize_temps(fake_B)
            
            # data augmented B temps, serves as negatives
            transform_jit = transforms.ColorJitter(brightness=0.5, contrast=0.75, saturation=1.5, hue=0.5)
            B_tf = transform_jit(real_B)
            TBTF = vectorize_temps(B_tf)
            TB = TB.reshape(TB.size(0), 1, TB.size(1), TB.size(2))
            loss_temp_g = criterion_temp(TFB_, TB, TBTF)*lambda_t
            
            #>>LPIPS loss - perceptual similarity
            loss_pix_g = criterion_lpips(fake_B, real_B)
            
            #>>Fourier Transform Loss for Each Patch
            A1f, P1f = fft_components(fake_B1) 
            A2f, P2f = fft_components(fake_B2)
            A3f, P3f = fft_components(fake_B3)
            A4f, P4f = fft_components(fake_B4)
            
            A1r, P1r = fft_components(B1)
            A2r, P2r = fft_components(B2)
            A3r, P3r = fft_components(B3)
            A4r, P4r = fft_components(B4)
            
            loss_Amp = 0.25*(criterion_amp(A1f, A1r) + criterion_amp(A2f, A2r) + criterion_amp(A3f, A3r) + criterion_amp(A4f, A4r))
            loss_Pha = 0.25*(criterion_phase(P1f, P1r) + criterion_phase(P2f, P2r) + criterion_phase(P3f, P3r) + criterion_phase(P4f, P4r))
            loss_FFT = 1/2*(loss_Amp + loss_Pha)
            
            #>>Total Generator Loss
            
            loss_G = loss_GAN_g + loss_label + loss_pix_g + 0.10*loss_temp_g + 0.001*loss_FFT
            #loss_G = loss_GAN_g + loss_pix_g + loss_triplet_patch + loss_label + 0.10*loss_temp_g + 0.001*loss_FFT

        scaler.scale(loss_G).backward()
        scaler.step(optimizer_G)
        print("+ + + optimizer_G.step() + + +")

        # ----------------------
        #  Train Discriminator 1
        # ----------------------

        print("+ + + optimizer_D.zero_grad() + + +")
        optimizer_D.zero_grad()
        
        with autocast():
            #>> Adversarial Losses
            # real
            pred_real_g, pred_real_gen, pred_real_ethn, pred_real_age = discriminator1(real_B, real_A)
            
            #fake_B is based on A + noise
            pred_fake_g, pred_fake_gen, pred_fake_ethn, pred_fake_age = discriminator1(fake_B.detach(), real_A)

            #adv loss
            loss_real_g = criterion_GAN(pred_real_g - pred_fake_g, valid) # D real loss(real, valid)
            loss_fake_g = criterion_GAN(pred_fake_g - pred_real_g, fake) # D fake loss(fake, fake)
            
            #>> Label Losses
            # real label loss
            # CE(input, target); input => float, target => long tensor
            real_loss_label = 1/3*(criterion_label(pred_real_gen, gender) + criterion_label(pred_real_ethn, ethn) + criterion_label(pred_real_age, age))
            
            # fake label loss <- I don't know what will happen here
            gen_gender = label_formatter(gen_gender)
            gen_ethn = label_formatter(gen_ethn)
            gen_age = label_formatter(gen_age)
            fake_loss_label = 1/3*(criterion_label(pred_fake_gen, gen_gender) + criterion_label(pred_fake_ethn, gen_ethn) + criterion_label(pred_fake_age, gen_age))

            # TOTAL LOSSES - average both
            loss_D = 1/2*((loss_real_g + real_loss_label) + (loss_fake_g + fake_loss_label))
            

        scaler.scale(loss_D).backward()
        scaler.step(optimizer_D)
        print("+ + + optimizer_D.step() + + +")
        
        # one scalar update for all scaler
        # instead of one for each model
        scaler.update()

        # --------------
        #  Log Progress
        # --------------

        batches_done = epoch * len(dataloader) + i  
        batches_left = opt.n_epochs * len(dataloader) - batches_done
        time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
        prev_time = time.time()

        # Print log
        sys.stdout.write(
            "\r |Experiment: %s| [Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f | GAN G: %f | lpips: %f | temp_G: %f | D_real_lab: %f | D_fake_lab: %f | fft: %f | G_loss_label: % f] ETA: %s"
            % (
                opt.experiment, 
                epoch, #%d
                opt.n_epochs, #%d
                i, #%d
                len(dataloader), #%d
                loss_D.item(), #%f
                loss_G.item(), #%f - total G loss
                loss_GAN_g.item(),
                loss_pix_g.item(),
                loss_temp_g.item(),
                real_loss_label.item(),
                fake_loss_label.item(),
                loss_FFT.item(),
                loss_label.item(),
                time_left, #%s
            )
        )

        f.write(
            "\r |Experiment: %s| [Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f | GAN G: %f | lpips: %f | temp_G: %f | D_real_lab: %f | D_fake_lab: %f | fft: %f | G_loss_label: % f] ETA: %s"
            % (
                opt.experiment, 
                epoch, #%d
                opt.n_epochs, #%d
                i, #%d
                len(dataloader), #%d
                loss_D.item(), #%f
                loss_G.item(), #%f - total G loss
                loss_GAN_g.item(),
                loss_pix_g.item(),
                loss_temp_g.item(),
                real_loss_label.item(),
                fake_loss_label.item(),
                loss_FFT.item(),
                loss_label.item(),
                time_left, #%s
            )
        )
         # If at sample interval save image
        if batches_done % opt.sample_interval == 0:
            sample_images(batches_done)

                
    if opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0:
        # Save model checkpoints
        torch.save(generator.state_dict(), "/home/local/AD/cordun1/experiments/TFC-GAN/saved_models/%s/generator_%d.pth" % (opt.experiment, epoch))
        torch.save(discriminator1.state_dict(), "/home/local/AD/cordun1/experiments/TFC-GAN/saved_models/%s/discriminator1_%d.pth" % (opt.experiment, epoch))
    
f.close()
