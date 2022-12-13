from PIL import Image
import numpy as np
from PIL import Image
import pandas as pd
import argparse
import os
from os import listdir,makedirs
from os.path import isfile,join
import cv2
import os,glob
#from skimage.measure import compare_ssim
from skimage.metrics import structural_similarity
# This function was renamed from skimage.measure.compare_ssim to skimage.metrics.structural_similarity.
import re
import itertools

"""
Script to convert real and fake images in a diretory to arrays,
and compute the PSNR and SSIM.

Note: This script is Devcom-file-specific

"""

################
# CONVERSION
################
def get_arrays(infile):
    arrays = [] #just a list of arrays
    filenames = [] # list of file names at the same indices
    file_nums = [] #the number which we will merge on later

    dirs = os.listdir(infile)
    for item in dirs:
        fullpath = os.path.join(infile, item)

        img = Image.open(fullpath) # open it

        f, e = os.path.splitext(fullpath)
        name = os.path.basename(f) # I need the filename to match it up right
        num_label = [int(s) for s in re.findall(r'\d+', name)]
        print(name)
        print(num_label)
        filenames.append(name) #save the filename
        file_nums.append(num_label)

        img.load()
        data = np.asarray(img, dtype="float32") # converts to array
        arrays.append(data) #store it in an array
    file_nums_ = list(itertools.chain.from_iterable(file_nums))
    return arrays, filenames, file_nums_

################
# PSNR
################
def calculate_psnr(img1, img2, max_value=255):
    # img1 = real
    # img2 = fake
    """"Calculating peak signal-to-noise ratio (PSNR) between two images."""
    mse = np.mean((np.array(img1, dtype=np.float32) - np.array(img2, dtype=np.float32)) ** 2)
    if mse == 0:
        return 100
    val = 20 * np.log10(max_value / (np.sqrt(mse)))
    return val

def psnr(master_array):
    values = []
    for i in range(0, len(master_array)): #len is the same for fake and real
        psnr_val = calculate_psnr(master_array[i][4], master_array[i][2]) #master_array[i][4] is original image, whereas [i][2] is the fake one
        values.append(psnr_val)
    #print(values)
    table = pd.DataFrame(values)
    return values, table

################
# SSIM
################
def convert_grayscale(real, fake):
    """
    real - directory for real_B
    fake - directory for fake_B

    """
    reals = []
    reals_image_names = []
    reals_nums = []
    files = [f for f in listdir(real) if isfile(join(real,f))]
    for image in files:
        num_label = [int(s) for s in re.findall(r'\d+', image)]
        reals_image_names.append(image)
        reals_nums.append(num_label)
        img = cv2.imread(os.path.join(real,image))
        gray_REAL = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        reals.append(gray_REAL)
    real_digits_ = list(itertools.chain.from_iterable(reals_nums))

    fakes = []
    fakes_image_names = []
    fakes_nums = []
    files_ = [f for f in listdir(fake) if isfile(join(fake,f))]
    for image in files_:
        num_label = [int(s) for s in re.findall(r'\d+', image)]
        fakes_image_names.append(image)
        fakes_nums.append(num_label)
        img_ = cv2.imread(os.path.join(fake,image))
        gray_FAKE = cv2.cvtColor(img_,cv2.COLOR_BGR2GRAY)
        fakes.append(gray_FAKE)
    fake_digits_ = list(itertools.chain.from_iterable(fakes_nums))
    return reals, reals_image_names, real_digits_, fakes, fakes_image_names, fake_digits_

# BELOW HAS DEVCOM-specifci TEST SET file names
# mssimfloat
# RETURNS The mean structural similarity index over the image.

def ssim(master_array):
    values = []
    for i in range(0, len(master_array)): #len is the same for fake and real
        #master_array[i][4] is original image, whereas [i][2] is the fake one
        score, diff = structural_similarity(master_array[i][4], master_array[i][2], full=True, multichannel=True)
        diff = (diff * 255).astype("uint8")
        values.append(score)
    print(values)
    table = pd.DataFrame(values)
    #devcom_test_files = pd.read_csv('devcom_test_set.txt', sep=" ", header=None)
    devcom_test_files = pd.read_csv(opt.test_set+".txt", sep=" ", header=None)
    b = devcom_test_files[0] # images 0 through 105 in the fake and real dirs follow exactly the order of the test files
    ssim_table = pd.concat([table, b], axis=1)
    return ssim_table


def main(fake_dir, real_dir, experiment, test_set):
    """
    fake_dir path to fake_B images
    real_dir path to real_B images
    experiment is a string like 'exp_a1'
    test_set is the output.txt of files names - depends on Devcom 5% or full Devcom/test set
    PSNR - 'get_arrays'
    SSIM - 'convert_grayscale'
    """
    # PSNR metrics
    fake_B, fake_B_names, fake_B_nums = get_arrays(fake_dir) # convert to arrays
    fake_B_cat = [list(a) for a in zip(fake_B_nums, fake_B_names, fake_B)]

    real_B, real_B_names, real_B_nums = get_arrays(real_dir) # convert to arrays
    real_B_cat = [list(a) for a in zip(real_B_nums, real_B_names, real_B)]

    # create dataframes <- need to work on a more efficient way to enumerate the list
    fb = pd.DataFrame(fake_B_cat)
    rb = pd.DataFrame(real_B_cat)
    merged = fb.merge(rb, on=0)
    master = merged.values.tolist() #change back into list of lists
    devcom_test_files = pd.read_csv(opt.test_set+".txt", sep=" ", header=None)
    #devcom_test_files = pd.read_csv('devcom_test_set.txt', sep=" ", header=None)
    b = devcom_test_files[0] # images 0 through 105 in the fake and real dirs follow exactly the order of the test files

    psnr_values, psnr_table = psnr(master)
    psnr_table = pd.concat([psnr_table, b], axis=1)
    psnr_table.to_csv('/Users/catherine/Documents/GANs_Research/my_imps/research_models/TFCGAN_CVPR/evaluation/Devcom/%s/psnr.csv' % (experiment))

    # SSIM metrics
    reals, reals_image_names, real_digits_, fakes, fakes_image_names, fake_digits_ = convert_grayscale(real_dir, fake_dir)
    real_zip = [list(a) for a in zip(real_digits_, reals_image_names, reals)]
    fake_zip = [list(a) for a in zip(fake_digits_, fakes_image_names, fakes)]
    r_df = pd.DataFrame(real_zip)
    f_df = pd.DataFrame(fake_zip)
    merged = f_df.merge(r_df, on=0)
    master = merged.values.tolist()
    ssim_table = ssim(master)
    ssim_table.to_csv('/Users/catherine/Documents/GANs_Research/my_imps/research_models/TFCGAN_CVPR/evaluation/Devcom/%s/ssim.csv' % (experiment))


### MAIN ###
if __name__ == '__main__':

    #fake_dir = '/Users/catherine/Documents/GANs_Research/my_imps/research_models/evaluation/exp_a1/fake_B'
    #real_dir = '/Users/catherine/Documents/GANs_Research/my_imps/research_models/evaluation/exp_a1/real_B'

    parser = argparse.ArgumentParser()
    parser.add_argument("--test_set", type=str, default="devcom_test_set", help="list of test set file names")
    parser.add_argument("--real_dir", type=str, default="none", help="path real_B directory")
    parser.add_argument("--fake_dir", type=str, default="none", help="path fake_B directory")
    parser.add_argument("--experiment", type=str, default="none", help="experiment name")
    opt = parser.parse_args()
    print(opt)

    main(opt.fake_dir, opt.real_dir, opt.experiment, opt.test_set)
