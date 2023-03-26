import PIL
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
import sys
import os
import argparse
import shutil

"""
Crops a 256 x 768 images of stacked images from the test phase:
real_A, fake_B, real_B and puts them into respective directories. Run
this before evaluation.py

img_sample_global = torch.cat((real_A.data, real_B.data, warped_B.data, fake_A.data, fake_B.data), -1)

"""

#model arg is for stargan that aligns the test images side-by-side
def crop_it(infile_path, RA_out, RB_out, REGB_out, experiment):
    
    dirs = os.listdir(infile_path)
    counter = 0
    for item in dirs:
        fullpath = os.path.join(infile_path, item)
        if os.path.isfile(fullpath):
            counter += 1
            im = Image.open(fullpath) # open the source image
            f, e = os.path.splitext(fullpath) # file and its extension like a1, .png

            real_A = im.crop((0, 0, 256, 256))
            real_B = im.crop((256, 0, 512, 256))
            reg_B = im.crop((512, 0, 768, 256))
            fake_A = im.crop((768, 0, 1024, 256))
            fake_B = im.crop((1024, 0, 1280, 256))

            save_rA_fname = os.path.join(RA_out, os.path.basename(f) + '_real_A' + '.png')
            save_regB_fname = os.path.join(REGB_out, os.path.basename(f) + '_reg_B' + '.png')
            save_rB_fname = os.path.join(RB_out, os.path.basename(f) + '_real_B'+ '.png')

            real_A.save(save_rA_fname, quality=100)
            reg_B.save(save_regB_fname, quality=100)
            real_B.save(save_rB_fname, quality=100)

            print(counter)

# crop_it(infile_path, RA_out, RB_out, REGB_out, experiment):
def main(inpath, RA_out, RB_out, REGB_out, experiment):
    crop_it(infile_path=inpath, RA_out=RA_out, RB_out=RB_out, REGB_out=REGB_out, experiment=experiment)


### MAIN ###
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--inpath", type=str, default="none", help="path to test results original images")
    parser.add_argument("--RA_out", type=str, default="none", help="path to real_A visible dir")
    parser.add_argument("--REGB_out", type=str, default="none", help="path to reg_B dir")
    parser.add_argument("--RB_out", type=str, default="none", help="path real_B thermal dir")
    parser.add_argument("--experiment", type=str, default="none", help="experiment_name")
    opt = parser.parse_args()
    print(opt)

    os.makedirs("/home/local/AD/cordun1/experiments/TFC-GAN-STN/crop_results/%s/" % opt.experiment, exist_ok=True)
    os.makedirs("/home/local/AD/cordun1/experiments/TFC-GAN-STN/crop_results/%s/real_A" % opt.experiment, exist_ok=True)
    os.makedirs("/home/local/AD/cordun1/experiments/TFC-GAN-STN/crop_results/%s/reg_B" % opt.experiment, exist_ok=True)
    os.makedirs("/home/local/AD/cordun1/experiments/TFC-GAN-STN/crop_results/%s/real_B" % opt.experiment, exist_ok=True)
    
    main(opt.inpath, opt.RA_out, opt.RB_out, opt.REGB_out, opt.experiment)
