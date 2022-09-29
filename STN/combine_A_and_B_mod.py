import os
import numpy as np
import cv2
import argparse
from multiprocessing import Pool


"""
Pairs without needing a train or test sub-directory.
Official pairing script from: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/datasets/combine_A_and_B.py

Usage:
python datasets/combine_A_and_B.py --fold_A /path/to/data/A --fold_B /path/to/data/B --fold_AB /path/to/data
A and B image names must be EXACTLY the same like: A/1.jpg and B/1.jpg

Exa:

A: '/home/ec2-user/scripts/frames/0009/20201215T172319' # RGB
B: '/home/ec2-user/scripts/thermal/0009/20201215T172319' # THERMAL
AB: '/home/ec2-user/scripts/pairs/0009/20201215T172319'

python datasets/combine_A_and_B.py --fold_A /path/to/data/A --fold_B /path/to/data/B --fold_AB /path/to/data
Ref: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/datasets.md

"""


def image_write(path_A, path_B, path_AB):
    im_A = cv2.imread(path_A, 1) # python2: cv2.CV_LOAD_IMAGE_COLOR; python3: cv2.IMREAD_COLOR
    im_B = cv2.imread(path_B, 1) # python2: cv2.CV_LOAD_IMAGE_COLOR; python3: cv2.IMREAD_COLOR
    im_AB = np.concatenate([im_A, im_B], 1)
    cv2.imwrite(path_AB, im_AB)


parser = argparse.ArgumentParser('create image pairs')
parser.add_argument('--fold_A', dest='fold_A', help='input directory for image A', type=str, default='../dataset/50kshoes_edges')
parser.add_argument('--fold_B', dest='fold_B', help='input directory for image B', type=str, default='../dataset/50kshoes_jpg')
parser.add_argument('--fold_AB', dest='fold_AB', help='output directory', type=str, default='../dataset/test_AB')
parser.add_argument('--num_imgs', dest='num_imgs', help='number of images', type=int, default=1000000)
parser.add_argument("--experiment", type=str, default="none", help="experiment_name")
args = parser.parse_args()

print("STARTING!")

os.makedirs("/home/local/AD/cordun1/experiments/TFC-GAN-STN/%s/pairs/real/" % args.experiment, exist_ok=True)
os.makedirs("/home/local/AD/cordun1/experiments/TFC-GAN-STN/%s/pairs/reg/" % args.experiment, exist_ok=True)

for arg in vars(args):
    print('[%s] = ' % arg, getattr(args, arg))


img_fold_A = args.fold_A
print("A:", img_fold_A)

img_fold_B = args.fold_B
print("B:", img_fold_B)

img_list = os.listdir(img_fold_A)

num_imgs = min(args.num_imgs, len(img_list))
print('use %d/%d images' % (num_imgs, len(img_list)))

img_fold_AB = args.fold_AB

if not os.path.isdir(img_fold_AB):
    os.makedirs(img_fold_AB)
    print('number of images = %d' % (num_imgs))

for n in range(num_imgs):
    name_A = img_list[n] #1.jpg
    path_A = os.path.join(img_fold_A, name_A)
    name_B = name_A
    path_B = os.path.join(img_fold_B, name_B)

    if os.path.isfile(path_A) and os.path.isfile(path_B):
        name_AB = name_A
        path_AB = os.path.join(img_fold_AB, name_AB)
        print("path_AB:", path_AB)

        im_A = cv2.imread(path_A, 1) # python2: cv2.CV_LOAD_IMAGE_COLOR; python3: cv2.IMREAD_COLOR
        im_B = cv2.imread(path_B, 1) # python2: cv2.CV_LOAD_IMAGE_COLOR; python3: cv2.IMREAD_COLOR
        im_AB = np.concatenate([im_A, im_B], 1)
        cv2.imwrite(path_AB, im_AB)

print("Done!")
