from __future__ import print_function
import os
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms, datasets

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file, usecols=['ImageId', 'TrueLabel', 'TargetClass'])
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0]) + ".png"
        image = Image.open(img_path)
        img_name = self.img_labels.iloc[idx, 0]
        label = self.img_labels.iloc[idx, 1]
        target = self.img_labels.iloc[idx, 2]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label, target, img_name

class CustomImageDatasetMultiTarget(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file, usecols=['ImageId', 'TrueLabel', 'TargetClass'])
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0]) + ".png"
        image = Image.open(img_path)
        img_name = self.img_labels.iloc[idx, 0]
        label = self.img_labels.iloc[idx, 1]
        target = self.img_labels.iloc[idx, 2]

        #generate 10 labels (randomly picked)
        target_list = list(range(target, target+10))

        #make sure target > 0 and < 999
        if target + 10 > 999:
            target_list = list(range(target-10, target))

        #make sure true labels are not in target list
        if label in target_list:
            target_list = list(range(target, target + 11))
            if target + 11 > 999:
                target_list = list(range(target - 11, target))
            target_list.remove(label)

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label, target_list, img_name