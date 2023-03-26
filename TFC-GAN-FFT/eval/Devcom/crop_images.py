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

"""

#model arg is for stargan that aligns the test images side-by-side
def crop_it(model, infile_path, RA_out, RB_out, FB_out, experiment):

    if model=='stargan':
        print("Stargan: copying real_B to new real_B dir.")
        realB_source = "/Users/catherine/Documents/GANs_Research/my_imps/research_models/TFCGAN_CVPR/evaluation/Devcom/pix2pix_Devcom/real_B/"
        destination = "/Users/catherine/Documents/GANs_Research/my_imps/research_models/TFCGAN_CVPR/evaluation/Devcom/%s/real_B/" % experiment

        for file_name in os.listdir(realB_source):
            src = realB_source + file_name
            dst = destination + file_name

            if os.path.isfile(src):
                shutil.copy(src, dst)
                print('copied', src)

    dirs = os.listdir(infile_path)
    counter = 0
    for item in dirs:
        fullpath = os.path.join(infile_path, item)
        if os.path.isfile(fullpath):
            counter += 1
            im = Image.open(fullpath) # open the source image
            f, e = os.path.splitext(fullpath) # file and its extension like a1, .png

            if model=='custom':            # do the cropping
                real_A = im.crop((0, 0, 256, 256))
                fake_B = im.crop((0, 256, 256, 512))
                real_B = im.crop((0, 512, 256, 768))

                save_rA_fname = os.path.join(RA_out, os.path.basename(f) + '_real_A' + '.png')
                save_fB_fname = os.path.join(FB_out, os.path.basename(f) + '_fake_B' + '.png')
                save_rB_fname = os.path.join(RB_out, os.path.basename(f) + '_real_B'+ '.png')

                real_A.save(save_rA_fname, quality=100)
                fake_B.save(save_fB_fname, quality=100)
                real_B.save(save_rB_fname, quality=100)

            elif model=='stargan':
                real_A = im.crop((0, 0, 256, 256))
                fake_B = im.crop((512, 0, 768, 256))

                save_rA_fname = os.path.join(RA_out, os.path.basename(f) + '_real_A' + '.png')
                save_fB_fname = os.path.join(FB_out, os.path.basename(f) + '_fake_B' + '.png')

                real_A.save(save_rA_fname, quality=100)
                fake_B.save(save_fB_fname, quality=100)
            print(counter)





def main(model, inpath, RA_out, RB_out, FB_out, experiment):
    crop_it(model, infile_path=inpath, RA_out=RA_out, RB_out=RB_out, FB_out=FB_out, experiment=experiment)


### MAIN ###
if __name__ == '__main__':
    # inpath = '/Users/catherine/Documents/GANs_Research/my_imps/research_models/experiments/exp_a1/test_results/eurecom_faces'
    # RA_out = '/Users/catherine/Documents/GANs_Research/my_imps/research_models/evaluation/exp_a1/real_A'
    # RB_out = '/Users/catherine/Documents/GANs_Research/my_imps/research_models/evaluation/exp_a1/real_B'
    # FB_out = '/Users/catherine/Documents/GANs_Research/my_imps/research_models/evaluation/exp_a1/fake_B'

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="custom", help="custom, stargan")
    parser.add_argument("--inpath", type=str, default="none", help="path to test results original images")
    parser.add_argument("--RA_out", type=str, default="none", help="path to real_A visible dir")
    parser.add_argument("--RB_out", type=str, default="none", help="path real_B thermal dir")
    parser.add_argument("--FB_out", type=str, default="none", help="path to fake_B dir")
    parser.add_argument("--experiment", type=str, default="none", help="experiment_name")
    opt = parser.parse_args()
    print(opt)

    os.makedirs("/Users/catherine/Documents/GANs_Research/my_imps/research_models/TFCGAN_CVPR/evaluation/Devcom/%s/fake_B" % opt.experiment, exist_ok=True)
    os.makedirs("/Users/catherine/Documents/GANs_Research/my_imps/research_models/TFCGAN_CVPR/evaluation/Devcom/%s/real_B" % opt.experiment, exist_ok=True)
    os.makedirs("/Users/catherine/Documents/GANs_Research/my_imps/research_models/TFCGAN_CVPR/evaluation/Devcom/%s/real_A" % opt.experiment, exist_ok=True)

    main(opt.model, opt.inpath, opt.RA_out, opt.RB_out, opt.FB_out, opt.experiment)
