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
        transforms.Resize((256,256), Image.BICUBIC),
        transforms.Grayscale(),
        transforms.ToTensor()])
    
    pil_to_tensor = preprocess(img).unsqueeze_(0)
    return pil_to_tensor


# plot the morphological gradient
def ssim_pairs(num, path, exp):
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
    
    # apply mogrphological grad using kernel
    m_A = 1. - (morph.gradient(A_img, kernel)) # Morphological gradient
    m_B = 1. - (morph.gradient(B_img, kernel)) # Morphological gradient
    m_GB = 1. - (morph.gradient(rB_img, kernel)) # Morphological gradient
    
    m_A_np = (m_A.reshape(256, 256)).numpy()
    m_B_np = (m_B.reshape(256, 256)).numpy()
    m_GB_np = (m_GB.reshape(256, 256)).numpy()
    
    return m_A_np, m_B_np, m_GB_np
    

def calculate(m_A_np, m_B_np, m_GB_np):  
    bef = ssim(m_A_np, m_B_np)
    aft = ssim(m_A_np, m_GB_np)
    return bef, aft
     
    
if __name__ == '__main__':

    real_A_files = opt.path + opt.exp + "/real_A"
    files = os.listdir(real_A_files)

    # count 0 - k 
    before = []
    after = []
    for i in range(0, len(files)-1):
        m_A_np, m_B_np, m_GB_np = ssim_pairs(i, opt.path, opt.exp)
        bef, aft = calculate(m_A_np, m_B_np, m_GB_np)
        before.append(bef)
        after.append(aft)
        
    before_reg = (np.array(before)).mean()
    after_reg = (np.array(after)).mean()
        
    print("Exp: {}, Before: {}, After: {}".format(opt.exp, before_reg, after_reg))
        
        
