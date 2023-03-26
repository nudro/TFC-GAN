import PIL
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
import sys
import os
import argparse

"""
Crops a 256 x 768 images of stacked images from the test phase:
real_A, fake_B, real_B and puts them into respective directories. Run
this before evaluation.py

STN experiment 0823_STN 6 outputs 256x256 stack of:
real_A, fake_A, reg_B, fake_A_from_reg_B, real_B

assumes a total of 5 images from the test script

Just want real_A and reg_B and real_B

"""

def crop_it(infile_path, RA_out, RegB_out, RB_out):
    dirs = os.listdir(infile_path)
    counter = 0
    for item in dirs:
        fullpath = os.path.join(infile_path, item)
        if os.path.isfile(fullpath):
            counter += 1
            im = Image.open(fullpath) # open the source image
            f, e = os.path.splitext(fullpath) # file and its extension like a1, .png
            """
            # do the cropping - may need to modify dependingn on experiment
            # A, B, FB, FWB, RB
            
            real_A = im.crop((0, 0, 256, 256))
            real_B = im.crop((0, 256, 256, 512))
            fake_B = im.crop((0, 512, 256, 768))
            fake_wB = im.crop((0, 768, 256, 1024))
            reg_B = im.crop((0, 1024, 256, 1280))
            """

            #0901_STN_V8_OG
            #A, B, fake_A, fake_B, warped_fB, REG_B (6)
            #real_A = im.crop((0, 0, 256, 256))
            #real_B = im.crop((0, 256, 256, 512))
            #reg_B = im.crop((0, 1024, 256, 1280))
            
            # 0922_STN_V8_OG_fBA_Mesh
            real_A = im.crop((0, 0, 256, 256))
            real_B = im.crop((0, 256, 256, 512))
            #fake_B = im.crop((0, 512, 256, 768))
            #fake_wB = im.crop((0, 768, 256, 1024))
            reg_B = im.crop((0, 1024, 256, 1280))


            save_rA_fname = os.path.join(RA_out, os.path.basename(f) + '.png')
            save_regB_fname = os.path.join(RegB_out, os.path.basename(f) + '.png')
            save_rB_fname = os.path.join(RB_out, os.path.basename(f) + '.png')

            real_A.save(save_rA_fname, quality=100)
            reg_B.save(save_regB_fname, quality=100)
            real_B.save(save_rB_fname, quality=100)
            print(counter)
            #if counter <=10:
                #display(real_A, fake_B, real_B)

def main(inpath, RA_out, RegB_out, RB_out):
    crop_it(infile_path=inpath, RA_out=RA_out, RegB_out=RegB_out, RB_out=RB_out)

### MAIN ###
if __name__ == '__main__':
    # use crop.sh to declare paths
    parser = argparse.ArgumentParser()
    parser.add_argument("--inpath", type=str, default="none", help="path to test results original images")
    parser.add_argument("--RA_out", type=str, default="none", help="path to real_A visible dir")
    parser.add_argument("--RegB_out", type=str, default="none", help="path reg_B thermal dir")
    parser.add_argument("--RB_out", type=str, default="none", help="path to real_B dir")
    parser.add_argument("--experiment", type=str, default="none", help="experiment_name")
    opt = parser.parse_args()
    print(opt)

    os.makedirs("/home/local/AD/cordun1/experiments/TFC-GAN-STN/%s/real_B" % opt.experiment, exist_ok=True)
    os.makedirs("/home/local/AD/cordun1/experiments/TFC-GAN-STN/%s/reg_B" % opt.experiment, exist_ok=True)
    os.makedirs("/home/local/AD/cordun1/experiments/TFC-GAN-STN/%s/real_A" % opt.experiment, exist_ok=True)

    main(opt.inpath, opt.RA_out, opt.RegB_out, opt.RB_out)
