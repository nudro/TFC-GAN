from PIL import Image
import os
import numpy as np
import cv2
import glob
import torch
import pandas as pd
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable

"""
class TempVector_PyTorch(object):

    def __init__(self, image, d):
        # image is a path for all images - fullpath
        self.image = image
        self.d = d
        
    def replace_with_dict2(self, ar, dic):
        # Extract out keys and values
        k = np.array(list(dic.keys()))
        v = np.array(list(dic.values()))
        # Get argsort indices
        sidx = k.argsort()
        ks = k[sidx]
        vs = v[sidx]
        return vs[np.searchsorted(ks,ar)]
        
    def make_pixel_vectors(self):
        img = np.array(self.image)
        img = img[:, :, 0] # Red channel - for thermal, they're all the same (dtype=uint8)
        temps = self.replace_with_dict2(img, self.d) 
        return temps
"""
    
class ImageDataset(Dataset):
    
    def __init__(self, root, transforms_=None, mode="train"):
        self.transform = transforms.Compose(transforms_)
        self.files = sorted(glob.glob(os.path.join(root, mode) + "/*.*"))
        #self.T = np.linspace(24, 38, num=256) # 0 - 255 indices of temperatures in Celsius
        #self.d = dict(enumerate((self.T).flatten(), 0)) # dictionary like {0: 24.0, 1: 24.054901960784314, etc.}
     
        if mode == "test":
            self.files.extend(sorted(glob.glob(os.path.join(root, "test") + "/*.*")))

    def __getitem__(self, index):

        img = Image.open(self.files[index % len(self.files)])
        w, h = img.size
        img_A = img.crop((0, 0, w / 2, h))
        img_B = img.crop((w / 2, 0, w, h))
       
        newsize = (256, 256)
        img_A = img_A.resize(newsize, Image.BICUBIC) # Image.Resampling.BICUBIC
        img_B = img_B.resize(newsize, Image.BICUBIC)

        # Setting the points for cropped image
        left_1 = 0
        top_1 = 0
        right_1 = 256 - 256/2
        bottom_1 = 256 - 256/2

        left_2 = 256/2
        top_2 = 0
        right_2 = 256
        bottom_2 = 256 - 256/2

        left_3 = 0
        top_3 = 256 - 256/2
        right_3 = 256 - 256/2
        bottom_3 = 256

        left_4 = 256 - 256/2
        top_4 = 256 - 256/2
        right_4 = 256
        bottom_4 = 256

        # Cropped image of above dimension
        # (It will not change original image)
        A1 = img_A.crop((left_1, top_1, right_1, bottom_1))
        A2 = img_A.crop((left_2, top_2, right_2, bottom_2))
        A3 = img_A.crop((left_3, top_3, right_3, bottom_3))
        A4 = img_A.crop((left_4, top_4, right_4, bottom_4))
        
        B1 = img_B.crop((left_1, top_1, right_1, bottom_1))
        B2 = img_B.crop((left_2, top_2, right_2, bottom_2))
        B3 = img_B.crop((left_3, top_3, right_3, bottom_3))
        B4 = img_B.crop((left_4, top_4, right_4, bottom_4))

        # turns into tensors and normalizes
        img_A = self.transform(img_A)
        img_B = self.transform(img_B)
        
        
        A1 = self.transform(A1)
        A2 = self.transform(A2)
        A3 = self.transform(A3)
        A4 = self.transform(A4)
        
        B1 = self.transform(B1)
        B2 = self.transform(B2)
        B3 = self.transform(B3)
        B4 = self.transform(B4)
    
        return {"A": img_A, 
                "B": img_B, 
                "A1":A1,
                "A2":A2,
                "A3":A3,
                "A4":A4,
                "B1": B1, 
                "B2": B2, 
                "B3": B3, 
                "B4": B4}


    def __len__(self):
        return len(self.files)
    
    
class TestImageDataset(Dataset):
    
    def __init__(self, root, transforms_=None, mode="test"):
        self.transform = transforms.Compose(transforms_)
        self.files = sorted(glob.glob(os.path.join(root, mode) + "/*.*"))

    def __getitem__(self, index):

        img = Image.open(self.files[index % len(self.files)])
        w, h = img.size
        img_A = img.crop((0, 0, w / 2, h))
        img_B = img.crop((w / 2, 0, w, h))
       
        newsize = (256, 256)
        img_A = img_A.resize(newsize, Image.BICUBIC)
        img_B = img_B.resize(newsize, Image.BICUBIC)
        
        # Setting the points for cropped image
        left_1 = 0
        top_1 = 0
        right_1 = 256 - 256/2
        bottom_1 = 256 - 256/2

        left_2 = 256/2
        top_2 = 0
        right_2 = 256
        bottom_2 = 256 - 256/2

        left_3 = 0
        top_3 = 256 - 256/2
        right_3 = 256 - 256/2
        bottom_3 = 256

        left_4 = 256 - 256/2
        top_4 = 256 - 256/2
        right_4 = 256
        bottom_4 = 256

        # Cropped image of above dimension
        # (It will not change original image)
        A1 = img_A.crop((left_1, top_1, right_1, bottom_1))
        A2 = img_A.crop((left_2, top_2, right_2, bottom_2))
        A3 = img_A.crop((left_3, top_3, right_3, bottom_3))
        A4 = img_A.crop((left_4, top_4, right_4, bottom_4))
        
        B1 = img_B.crop((left_1, top_1, right_1, bottom_1))
        B2 = img_B.crop((left_2, top_2, right_2, bottom_2))
        B3 = img_B.crop((left_3, top_3, right_3, bottom_3))
        B4 = img_B.crop((left_4, top_4, right_4, bottom_4))
        
        img_A = self.transform(img_A)
        img_B = self.transform(img_B)
        
        A1 = self.transform(A1)
        A2 = self.transform(A2)
        A3 = self.transform(A3)
        A4 = self.transform(A4)
        
        B1 = self.transform(B1)
        B2 = self.transform(B2)
        B3 = self.transform(B3)
        B4 = self.transform(B4)
    
        return {"A": img_A, 
                "B": img_B, 
                "A1":A1,
                "A2":A2,
                "A3":A3,
                "A4":A4,
                "B1": B1, 
                "B2": B2, 
                "B3": B3, 
                "B4": B4}



    def __len__(self):
        return len(self.files)
    
