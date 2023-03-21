import glob
import random
import os
import numpy as np
import torch
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

#I added attributes=None

class ImageDataset(Dataset):
    def __init__(self, annots_csv, root, transforms_=None, mode="train"):
        self.transform = transforms.Compose(transforms_)
        self.annots = pd.read_csv(annots_csv)
        self.files = sorted(glob.glob(os.path.join(root, mode) + "/*.*"))
        if mode == "train":
            self.files.extend(sorted(glob.glob(os.path.join(root, "test") + "/*.*")))

    def __getitem__(self, index):

        img = Image.open(self.files[index % len(self.files)])
        w, h = img.size
        img_A = img.crop((0, 0, w / 2, h))
        img_B = img.crop((w / 2, 0, w, h))

        if np.random.random() < 0.5:
            img_A = Image.fromarray(np.array(img_A)[:, ::-1, :], "RGB")
            img_B = Image.fromarray(np.array(img_B)[:, ::-1, :], "RGB")

        img_A = self.transform(img_A)
        img_B = self.transform(img_B)

        labels = self.annots.iloc[index, 1] #the label is the second col
        #print(labels, self.annots.iloc[index, 0]) 
        labels = np.array([labels])
        #labels = labels.astype('float').reshape(-1) #prev: .reshape(-1,1)
        labels = torch.Tensor(labels)
        #print("labels:", labels)
        #print("labels.shape:", labels.shape)
        return {"A": img_A, "B": img_B, "LAB": labels}

    def __len__(self):
        return len(self.files)
    
class TestImageDataset(Dataset):
    def __init__(self, annots_csv, root, transforms_=None, mode="test"):
        self.transform = transforms.Compose(transforms_)
        self.annots = pd.read_csv(annots_csv)
        self.files = sorted(glob.glob(os.path.join(root, mode) + "/*.*"))
        if mode == "train":
            self.files.extend(sorted(glob.glob(os.path.join(root, "test") + "/*.*")))

    def __getitem__(self, index):

        img = Image.open(self.files[index % len(self.files)])
        w, h = img.size
        img_A = img.crop((0, 0, w / 2, h))
        img_B = img.crop((w / 2, 0, w, h))

        img_A = self.transform(img_A)
        img_B = self.transform(img_B)

        labels = self.annots.iloc[index, 1] #the label is the second col
        labels = np.array([labels])
        #labels = labels.astype('float').reshape(-1) #prev: .reshape(-1,1)
        labels = torch.Tensor(labels)
        print("labels:", labels)
        print("labels.shape:", labels.shape)
        return {"A": img_A, "B": img_B, "LAB": labels}

    def __len__(self):
        return len(self.files)