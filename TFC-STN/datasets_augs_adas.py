import glob
import random
import os
import numpy as np
import torch
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

 
class TestDataset(Dataset):
    """
    You cannot pass a transforms here, otherwise it will do random transforms on 
    A or B, but not jointly. The only way to make pairwise, same transforms 
    is to use the aug_transform() fxn below.
    
    Flips the A,B pair together to evaluate robustness of Registration.
    Resize -> Flip -> Tensor -> Normalize
    """
    def __init__(self,root, transforms_=None, mode="test"):
        self.files = sorted(glob.glob(os.path.join(root, mode) + "/*.*"))

    def aug_transform(self, A, B):
        # A and B are PIL images
        
        affine_transfomer = transforms.RandomAffine(degrees=(5, 10), translate=(0.1, 0.2), scale=(1.9, 2.75))
        
        # Resize
        resize = transforms.Resize(size=(256, 256))
        A = resize(A)
        B = resize(B)

        # Randomly introduce an affine warp only to the thermal image 
        # Will make things larger
        if random.random() > 0.5:
            B = affine_transformer(B)
        
        # Transform to tensor
        A = TF.to_tensor(A)
        B = TF.to_tensor(B)
        
        # Normalize
        normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        A = normalize(A)
        B = normalize(B)
        
        return A, B

    def __getitem__(self, index):

        img = Image.open(self.files[index % len(self.files)])
        w, h = img.size
        img_A = img.crop((0, 0, w / 2, h)) # PIL
        img_B = img.crop((w / 2, 0, w, h)) # PIL
        
        # augmented, but remains paired
        img_A_aug, img_B_aug = self.aug_transform(img_A, img_B)
        
        return { "A": img_A_aug,
               "B": img_B_aug}

    def __len__(self):
        return len(self.files)

    
