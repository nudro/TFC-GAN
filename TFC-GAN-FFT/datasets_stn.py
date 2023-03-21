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

    
class ImageDataset(Dataset):
    
    def __init__(self, root, transforms_=None, mode="train"):
        self.transform = transforms.Compose(transforms_)
        self.files = sorted(glob.glob(os.path.join(root, mode) + "/*.*"))

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
        img_A = img_A.resize(newsize, Image.Resampling.BICUBIC)
        img_B = img_B.resize(newsize, Image.Resampling.BICUBIC)
    

        # turns into tensors and normalizes
        img_A = self.transform(img_A)
        img_B = self.transform(img_B)

    
        return {"A": img_A, 
                "B": img_B}


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
        
        # do the transforms here while it's still an image
        # Note - the resizing to 256 x 256 is passed in here not at transforms
       
        newsize = (256, 256)
        img_A = img_A.resize(newsize, Image.Resampling.BICUBIC)
        img_B = img_B.resize(newsize, Image.Resampling.BICUBIC)
        
        img_A = self.transform(img_A)
        img_B = self.transform(img_B)
        
        return {"A": img_A,
               "B": img_B,}


    def __len__(self):
        return len(self.files)
    
