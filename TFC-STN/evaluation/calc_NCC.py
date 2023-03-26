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
import numpy.ma as ma
from scipy.stats import norm
from scipy.optimize import curve_fit
from scipy.integrate import trapz, simps

# Ref: https://xcdskd.readthedocs.io/en/latest/cross_correlation/cross_correlation_coefficient.html#Application-as-an-Image-Similarity-Measure

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

def norm_data(data):
    """
    normalize data to have mean=0 and standard_deviation=1
    """
    mean_data=np.mean(data)
    std_data=np.std(data, ddof=1)
    #return (data-mean_data)/(std_data*np.sqrt(data.size-1))
    return (data-mean_data)/(std_data)


def ncc(data0, data1):
    """
    normalized cross-correlation coefficient between two data sets

    Parameters
    ----------
    data0, data1 :  numpy arrays of same size
    
    In our case, 256x256, 1 channel grayscale
    """
    return (1.0/(data0.size-1)) * np.sum(norm_data(data0)*norm_data(data1))


def ncc_pairs(num, path, exp):
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
    bef = ncc(A,B)
    aft = ncc(A,GB)

    return bef, aft
     
    
if __name__ == '__main__':

    real_A_files = opt.path + opt.exp + "/real_A"
    #real_B_files = opt.path + opt.exp + "/" + real_B
    #reg_B_files = opt.path + opt.exp + "/" + reg_B
    
    # count 0 - k 
    before = []
    after = []
    
    for i in range(0, len(real_A_files)):
        A, B, GB = ncc_pairs(i, opt.path, opt.exp)
        bef, aft = calculate(A, B, GB)
        
        before.append(bef)
        after.append(aft)
        
        
    before_reg = (np.array(before)).mean()
    after_reg = (np.array(after)).mean()
        
    print("Exp: {}, NCC Before: {}, NCC After: {}".format(opt.exp, before_reg, after_reg))
        
        