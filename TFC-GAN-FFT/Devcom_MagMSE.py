from PIL import Image
import numpy as np
import pandas as pd
import argparse
from os import listdir,makedirs
from os.path import isfile,join
import cv2
import os,glob
import re
import itertools
import matplotlib.pyplot as plt
from scipy.fft import fft2, fftshift
from skimage import img_as_float
from skimage.color import rgb2gray
from skimage.filters import window
import skimage.io
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
from matplotlib import cm

################
# CONVERSION
################

def get_arrays(infile):
    arrays = [] # just a list of arrays
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
        arrays.append(data) # store it in an array

    file_nums_ = list(itertools.chain.from_iterable(file_nums))
    return arrays, filenames, file_nums_

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


################
# MSE
################



def mse_spec(master_array):
    def my_wimage_fft(image):
        image_f = np.abs(fftshift(fft2(image)))
        return image_f
    
    values = []
    for i in range(0, len(master_array)):  # len is the same for fake and real
        real = master_array[i][4]  # just a path
        fake = master_array[i][2]  # just a path
        
        # calculate images
        real_arr = np.array(real, dtype=np.float32)
        fake_arr = np.array(fake, dtype=np.float32)
        
        real_mag = np.log(my_wimage_fft(real_arr))
        fake_mag = np.log(my_wimage_fft(fake_arr))
        
        try: 
            mse_val = mean_squared_error(real_mag, fake_mag)
        except:
            continue
        
        print(mse_val)
        
        values.append(mse_val)

    table = pd.DataFrame(values)
    return values, table


def make_spectra(image, i, mode, experiment):
    # convert to a PIL imge, grayscale
    img_ = Image.fromarray(np.uint8(image), 'L')
    arr = np.array(img_) #turn into numpy

    f_result = np.fft.fft2(arr) # setting this to regular FFT2 to make magnitude spectra
    fshift = np.fft.fftshift(f_result)
    magnitude_spectrum = np.log(np.abs(fshift)) 
    
    #data = Image.fromarray(magnitude_spectrum)
    #new_p = data.convert("RGB")
    #print(new_p)
    
    import matplotlib.image

    matplotlib.image.imsave("/home/local/AD/cordun1/experiments/TFC-GAN/evaluation/Devcom/{}/spectra/{}_{}.png".format(experiment, i, mode), magnitude_spectrum)
 
    # Note: i correlates with column 0 in .csv
    #new_p.save("/home/local/AD/cordun1/experiments/TFC-GAN/evaluation/Devcom/{}/spectra/{}_{}.png".format(experiment, i, mode))

             
def save_spectra_images(master_array, experiment):
    for i in tqdm(range(0, len(master_array))):
        
        real = master_array[i][4] #just a path
        fake = master_array[i][2] # just a path
        
        # send to make spectra
        real_spec = make_spectra(real, i, mode='real', experiment=experiment)
        fake_spec = make_spectra(fake, i, mode='fake', experiment=experiment)

    print("Done!")


def main(fake_dir, real_dir, experiment, test_set):
    """
    fake_B, fake_B_names, fake_B_nums = get_arrays(fake_dir) # convert to arrays
    fake_B_cat = [list(a) for a in zip(fake_B_nums, fake_B_names, fake_B)]

    real_B, real_B_names, real_B_nums = get_arrays(real_dir) # convert to arrays
    real_B_cat = [list(a) for a in zip(real_B_nums, real_B_names, real_B)]

    # create dataframes <- need to work on a more efficient way to enumerate the list
    fb = pd.DataFrame(fake_B_cat)
    rb = pd.DataFrame(real_B_cat)
    merged = fb.merge(rb, on=0)
    master = merged.values.tolist() #change back into list of lists

    
    """
    
    # Images need to be ready in 2channel for MSE 
    reals, reals_image_names, real_digits_, fakes, fakes_image_names, fake_digits_ = convert_grayscale(real_dir, fake_dir)
    real_zip = [list(a) for a in zip(real_digits_, reals_image_names, reals)]
    fake_zip = [list(a) for a in zip(fake_digits_, fakes_image_names, fakes)]
    r_df = pd.DataFrame(real_zip)
    f_df = pd.DataFrame(fake_zip)
    merged = f_df.merge(r_df, on=0)
    master = merged.values.tolist()

    # calculate MSE magnitude spectra values
    mse_values, mse_table = mse_spec(master)
    mse_values_ = np.array(mse_values)

    # Return average
    print("--------------Average MSE of Magnitude Spectra: [ {} ]-------------".format(mse_values_.mean()))

    # Save spectra images
    save_spectra_images(master, experiment)
    
    devcom_test_files = pd.read_csv(opt.test_set+".txt", sep=" ", header=None)
    b = devcom_test_files[0] # images 0 through 105 in the fake and real dirs follow exactly the order of the test files
    mse_table = pd.concat([mse_table, b], axis=1)
    mse_table.to_csv('/home/local/AD/cordun1/experiments/TFC-GAN/evaluation/Devcom/%s/mse_spec.csv' % (experiment))



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

    os.makedirs("/home/local/AD/cordun1/experiments/TFC-GAN/evaluation/Devcom/%s/spectra/" % opt.experiment, exist_ok=True)

    main(opt.fake_dir, opt.real_dir, opt.experiment, opt.test_set)
