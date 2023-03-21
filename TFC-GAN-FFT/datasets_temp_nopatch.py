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

    
class ImageDataset(Dataset):
    
    def __init__(self, root, transforms_=None, mode="train"):
        self.transform = transforms.Compose(transforms_)
        self.files = sorted(glob.glob(os.path.join(root, mode) + "/*.*"))
        self.T = np.linspace(24, 38, num=256) # 0 - 255 indices of temperatures in Celsius
        self.d = dict(enumerate((self.T).flatten(), 0)) # dictionary like {0: 24.0, 1: 24.054901960784314, etc.}
     
        if mode == "test":
            self.files.extend(sorted(glob.glob(os.path.join(root, "test") + "/*.*")))

    def __getitem__(self, index):

        img = Image.open(self.files[index % len(self.files)])
        w, h = img.size
        img_A = img.crop((0, 0, w / 2, h))
        img_B = img.crop((w / 2, 0, w, h))
        
        # do the transforms here while it's still an image
        # Note - the resizing to 256 x 256 is passed in here not at transforms
       
        newsize = (256, 256)
        img_A = img_A.resize(newsize, Image.BICUBIC)
        img_B = img_B.resize(newsize, Image.BICUBIC)
        
        # temps
        vectorizer = TempVector_PyTorch(img_B, self.d)
        img_B_temps = torch.Tensor(vectorizer.make_pixel_vectors())
       
        """
        Later on, I could always pass an arg, if num_patch=9, then
        use this set of crops. Since the crops below would be if
        num_patch = 4.
        
        """

        # turns into tensors and normalizes
        img_A = self.transform(img_A)
        img_B = self.transform(img_B)
    
        return {"A": img_A, 
                "B": img_B, 
               "T_B": img_B_temps}


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
        img_B = img.crop((w / 2, 0, w, h)) # only reading this in so I can get a sample image
        
        # do the transforms here while it's still an image
        # Note - the resizing to 256 x 256 is passed in here not at transforms
       
        newsize = (256, 256)
        img_A = img_A.resize(newsize, Image.BICUBIC)
        img_B = img_B.resize(newsize, Image.BICUBIC)
        
        img_A = self.transform(img_A)
        img_B = self.transform(img_B)
        
        return {"A": img_A,
               "B": img_B,}


    def __len__(self):
        return len(self.files)
    
