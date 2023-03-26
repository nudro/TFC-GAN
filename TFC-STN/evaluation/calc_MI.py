from PIL import Image
import cv2
import sys
import os
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as mpimg
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image               # to load images
import pandas as pd
from skimage.metrics import structural_similarity as ssim
import kornia as K
from kornia import morphology as morph
import matplotlib.pyplot as plt
from matplotlib import rcParams
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--exp", type=str, default="0302_STN21_Devcom_NewModel3", help="experiment")
parser.add_argument("--path", type=str, default="/home/local/AD/cordun1/experiments/TFC-GAN-STN/crop_results/", help="path")
opt = parser.parse_args()
print(opt)

# turn png to tensor
def to_tensor(path):
    img=Image.open(path)
    #display(img)
    preprocess = transforms.Compose([
        transforms.Resize((256,256), Image.Resampling.BICUBIC),
        transforms.Grayscale(),
        transforms.ToTensor()])
    
    pil_to_tensor = preprocess(img).unsqueeze_(0)
    return pil_to_tensor

# image plot
def hist(IM1, IM2):
    # https://matthew-brett.github.io/teaching/mutual_information.html
    rcParams['figure.figsize'] = 6,3
    plt.rcParams.update({'font.size': 10})
    fig, axes = plt.subplots(1, 2)
    axes[0].hist(IM1.ravel(), bins=20)
    axes[0].set_title('IM1')
    axes[1].hist(IM2.ravel(), bins=20)
    axes[1].set_title('IM2')
    
# image plot    
def corr_plot(IM1, IM2):
    plt.plot(IM1.ravel(), IM2.ravel(), '.')
    plt.xlabel('IM1')
    plt.ylabel('IM2')
    plt.title('IM1 vs IM2')
    print("Pearson Correlation:", np.corrcoef(IM1.ravel(), IM2.ravel())[0, 1])

    
def log_hist(IM1, IM2, plot=True):
    hist_2d, x_edges, y_edges = np.histogram2d(IM1.ravel(), IM2.ravel(), bins=20)
    
    if plot: 
        # Show log histogram, avoiding divide by 0
        hist_2d_log = np.zeros(hist_2d.shape)
        non_zeros = hist_2d != 0
        hist_2d_log[non_zeros] = np.log(hist_2d[non_zeros])
        plt.imshow(hist_2d_log.T, origin='lower')
        plt.xlabel('IM1 signal bin')
        plt.ylabel('IM2 signal bin')
    return hist_2d


def mutual_information(hgram):
    """ Mutual information for joint histogram"""
    # Convert bins counts to probability values
    pxy = hgram / float(np.sum(hgram))
    px = np.sum(pxy, axis=1) # marginal for x over y
    py = np.sum(pxy, axis=0) # marginal for y over x
    px_py = px[:, None] * py[None, :] # Broadcast to multiply marginals
    # Now we can do the calculation using the pxy, px_py 2D arrays
    nzs = pxy > 0 # Only non-zero pxy values contribute to the sum
    mi = np.sum(pxy[nzs] * np.log(pxy[nzs] / px_py[nzs]))
    return mi

# plot the morphological gradient
def MI_pairs(num, path, exp):
    # Pass the Real_A, Real_B, and Reg_B image paths (.png)
    device = 'cpu' # 'cuda:0' for GPU
    kernel = torch.tensor([[0, 1, 0],[1, 1, 1],[0, 1, 0]]).to(device)
    
    os.makedirs("/home/local/AD/cordun1/experiments/TFC-GAN-STN/plots/{}".format(exp), exist_ok=True)

    real_A = 'real_A/{}_real_A.png'.format(str(num))
    real_B = 'real_B/{}_real_B.png'.format(str(num))
    reg_B = 'reg_B/{}_reg_B.png'.format(str(num))

    real_A_path = path + exp + "/" + real_A
    real_B_path = path + exp + "/" + real_B
    reg_B_path = path + exp + "/" + reg_B
    
    # to tensor
    A_img = to_tensor(real_A_path)
    B_img = to_tensor(real_B_path)
    rB_img = to_tensor(reg_B_path)
    
    # to numpy 
    A = (A_img.reshape(256, 256)).numpy()
    B = (B_img.reshape(256, 256)).numpy()
    GB = (rB_img.reshape(256, 256)).numpy()
    
    return A, B, GB
    

def calculate(A, B, GB):  
    h_bef = log_hist(A, B, plot=False)
    bef = mutual_information(h_bef)
    
    h_aft = log_hist(A, GB, plot=False)
    aft = mutual_information(h_aft)

    return bef, aft
     
    
if __name__ == '__main__':

    real_A_files = opt.path + opt.exp + "/real_A"
    files = os.listdir(real_A_files)
    # no silly! not for i in range(len(real_A_files):
    # for in range(0, len(files)-1):

    # count 0 - k 
    before = []
    after = []
    for i in range(0, len(files)-1):
        A, B, GB = MI_pairs(i, opt.path, opt.exp)
        #hist(IM1, IM2) # for plots only
        #corr_plot(IM1, IM2) # for plots only
        bef, aft = calculate(A, B, GB)
        
        before.append(bef)
        after.append(aft)
        
    print("len before:", len(before))
    print("len after:", len(after))
    
    before_reg = (np.array(before)).mean()
    after_reg = (np.array(after)).mean()
        
    print("Exp: {}, Before: {}, After: {}".format(opt.exp, before_reg, after_reg))
        
        