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
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--exp", type=str, default="0302_STN21_Devcom_NewModel3", help="experiment")
parser.add_argument("--path", type=str, default="/home/local/AD/cordun1/experiments/TFC-GAN-STN/crop_results/", help="path")
opt = parser.parse_args()
print(opt)


# Make Difference Plots
def difference_plot(num,  path, exp):
    
    os.makedirs("/home/local/AD/cordun1/experiments/TFC-GAN-STN/plots/{}".format(exp), exist_ok=True)

    real_A = 'real_A/{}_real_A.png'.format(str(num))
    real_B = 'real_B/{}_real_B.png'.format(str(num))
    reg_B = 'reg_B/{}_reg_B.png'.format(str(num))

    real_A_path = path + exp + "/" + real_A
    real_B_path = path + exp + "/" + real_B
    reg_B_path = path + exp + "/" + reg_B

    # PIL to open
    A = Image.open(real_A_path).convert('L')
    rB = Image.open(real_B_path).convert('L')
    gB = Image.open(reg_B_path).convert('L')

    # Numpy for matplotlib
    A_array = np.asarray(A)
    rB_array = np.asarray(rB)
    gB_array = np.asarray(gB)

    fig = plt.figure()
    fig = plt.figure(figsize=(16,6))
    plt.subplots_adjust(wspace=0.0, hspace=0.0)
    plt.rcParams.update({'figure.max_open_warning': 0})
    plt.rcParams.update({'font.size': 18})
    
    ax1 = plt.subplot(1, 5, 1)
    imgplot = plt.imshow(A_array, cmap = 'bone', vmax = 255)
    ax1.set_xticks([]) 
    ax1.set_yticks([]) 
    plt.title("Visible")
    
    ax2 = plt.subplot(1, 5, 2)
    imgplot = plt.imshow(rB_array, cmap = 'bone', vmax = 255)
    ax2.set_xticks([]) 
    ax2.set_yticks([]) 
    plt.title("Before")
    
    ax3 = plt.subplot(1, 5, 3)
    imgplot = plt.imshow(gB_array, cmap = 'bone', vmax = 255)
    ax3.set_xticks([]) 
    ax3.set_yticks([]) 
    plt.title("Registered")

    ax4 = plt.subplot(1, 5, 4)
    imgplot = plt.imshow(1.0*A_array - rB_array, vmin = -200, vmax = 50, cmap = 'RdBu')
    ax4.set_xticks([]) 
    ax4.set_yticks([]) 
    plt.title("Diff. Before")
    
    ax5 = plt.subplot(1, 5, 5)
    imgplot = plt.imshow(1.0*A_array - gB_array, vmin = -200, vmax = 50, cmap = 'RdBu')
    ax5.set_xticks([]) 
    ax5.set_yticks([]) 
    plt.title("Diff. Registered")
    plt.savefig('/home/local/AD/cordun1/experiments/TFC-GAN-STN/plots/{}/{}.pdf'.format(exp,num), bbox_inches = "tight")

# Make Grids
def difference_grid(num,  path, exp):   
    
    os.makedirs("/home/local/AD/cordun1/experiments/TFC-GAN-STN/plots/{}".format(exp), exist_ok=True)

    real_A = 'real_A/{}_real_A.png'.format(str(num))
    real_B = 'real_B/{}_real_B.png'.format(str(num))
    reg_B = 'reg_B/{}_reg_B.png'.format(str(num))

    real_A_path = path + exp + "/" + real_A
    real_B_path = path + exp + "/" + real_B
    reg_B_path = path + exp + "/" + reg_B
    
    # Grid lines at these intervals (in pixels)
    # dx and dy can be different
    dx, dy = 20,20

    # Custom (rgb) grid color
    grid_color = [10,0,0]

    fig = plt.figure()
    fig = plt.figure(figsize=(16,6))
    plt.subplots_adjust(wspace=0.0, hspace=0.0)
    plt.rcParams.update({'figure.max_open_warning': 0})
    plt.rcParams.update({'font.size': 18})
    
    ax1 = plt.subplot(1, 4, 1)
    img = plt.imread(real_A_path)
    img[:,::dy,:] = grid_color
    img[::dx,:,:] = grid_color
    ax1.set_xticks([]) 
    ax1.set_yticks([]) 
    plt.imshow(img)
    plt.title("Visible")
    
    ax2 = plt.subplot(1, 4, 2)
    img = plt.imread(real_B_path)
    img[:,::dy,:] = grid_color
    img[::dx,:,:] = grid_color
    ax2.set_xticks([]) 
    ax2.set_yticks([]) 
    plt.imshow(img)
    plt.title("Before")
    
    ax3 = plt.subplot(1, 4, 3)
    img = plt.imread(real_A_path)
    img[:,::dy,:] = grid_color
    img[::dx,:,:] = grid_color
    ax3.set_xticks([]) 
    ax3.set_yticks([]) 
    plt.imshow(img)
    plt.title("Visible")
    
    ax4 = plt.subplot(1, 4, 4)
    img = plt.imread(reg_B_path)
    img[:,::dy,:] = grid_color
    img[::dx,:,:] = grid_color
    ax4.set_xticks([]) 
    ax4.set_yticks([]) 
    plt.imshow(img)
    plt.title("Registered")

    plt.savefig('/home/local/AD/cordun1/experiments/TFC-GAN-STN/plots/{}/{}_grid.pdf'.format(exp,num), bbox_inches = "tight")


if __name__ == '__main__':

    real_A_files = opt.path + opt.exp + "/real_A"
    files = os.listdir(real_A_files)
    #real_B_files = opt.path + opt.exp + "/" + real_B
    #reg_B_files = opt.path + opt.exp + "/" + reg_B
    
    # count 0 - k 
    for i in range(0, len(files)-1):
        print(i)
        difference_grid(i, opt.path, opt.exp)
        difference_plot(i, opt.path, opt.exp)