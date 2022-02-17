from PIL import Image
import numpy as np
from PIL import Image
import re
import itertools
import pandas as pd
import cv2
import argparse
import os

"""
Script to convert real and fake images in a diretory to arrays,
and compute the PSNR and SSIM.

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

def bhatt(master_array):
    method = cv2.HISTCMP_BHATTACHARYYA
    values = []
    for i in range(0, len(master_array)): #len is the same for fake and real
        real = np.array(master_array[i][4])
        fake = np.array(master_array[i][2])

        cv_imgR = cv2.cvtColor((real.astype(np.uint8)), cv2.COLOR_BGR2RGB)
        cv_imgF = cv2.cvtColor((fake.astype(np.uint8)), cv2.COLOR_BGR2RGB)

        histR = cv2.calcHist([cv_imgR], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        histR = cv2.normalize(histR, histR).flatten()

        histF = cv2.calcHist([cv_imgF], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        histF = cv2.normalize(histF, histF).flatten()

        d = cv2.compareHist(histR, histF, method)
        print("Bhattacharyya Distance is:", d)

        values.append(d)

    table = pd.DataFrame(values)
    return values, table

def main(fake_dir, real_dir, experiment, test_set):
    # fake_dir path to fake_B images
    # real_dir path to real_B images
    # experiment is a string like 'exp_a1'
    #test_set is the output.txt of files names - depends on Devcom 5% or full Devcom/test set
    # Conversion into arrays
    fake_B, fake_B_names, fake_B_nums = get_arrays(fake_dir)
    fake_B_cat = [list(a) for a in zip(fake_B_nums, fake_B_names, fake_B)]

    real_B, real_B_names, real_B_nums = get_arrays(real_dir)
    real_B_cat = [list(a) for a in zip(real_B_nums, real_B_names, real_B)]

    # create dataframes <- need to work on a more efficient way to enumerate the list
    print("Creating dataframes...")
    fb = pd.DataFrame(fake_B_cat)
    rb = pd.DataFrame(real_B_cat)
    merged = fb.merge(rb, on=0)
    master = merged.values.tolist() #change back into list of lists
    print("First index of master for checking/n:", master[0])

    # Used to merge later
    #devcom_test_files = pd.read_csv('devcom_test_set.txt', sep=" ", header=None)
    devcom_test_files = pd.read_csv(opt.test_set+".txt", sep=" ", header=None)
    b = devcom_test_files[0] # images 0 through 105 in the fake and real dirs follow exactly the order of the test files

    # Bhattacharyya metrics
    bhatt_values, bhatt_table = bhatt(master)
    bhatt_table = pd.concat([bhatt_table, b], axis=1)
    bhatt_table.to_csv('/Users/catherine/Documents/GANs_Research/my_imps/research_models/faPVTgan/evaluation/Devcom/%s/bhatt.csv' % (experiment))


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
