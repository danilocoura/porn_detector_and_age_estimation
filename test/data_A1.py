import PIL
import torch
import torchvision
from torchvision import transforms, datasets 
import torchvision.datasets
import os
import pandas as pd
import math
import scipy
from sklearn.neighbors import KernelDensity
import numpy as np
import config as c
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)

def make_dataset(root):
    images = []
    targets = []
    for folder in os.listdir(root):
        for image_path in os.listdir(os.path.join(root, folder)):
            images.append(os.path.join(root, folder, image_path))
            if folder.startswith("porn"):
                targets.append(0)
            else:
                targets.append(1)
    return images, targets

class PornDatasetA1(datasets.VisionDataset):
    def __init__(self, root, loader=default_loader, transforms=None, transform=None, target_transform=None, is_valid_file=None):
        super(PornDatasetA1, self).__init__(root, transform=None, target_transform=None)
        self.samples, self.targets = make_dataset(root)
        if len(self.samples) == 0:
            raise (RuntimeError("Found 0 files in this set"))
        self.loader = default_loader
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        path = self.samples[index]
        sample = self.loader(path)
        target = self.targets[index]
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target, path
    
    def __len__(self):
        return len(self.samples)


def get_default_transforms():
    # Define a data transformation to convert PIL images to PyTorch tensors
    # and normalize the data using the predetermined pixel means and standard
    # deviations
    
    test_transform = transforms.Compose(
        [transforms.Resize(size=(224, 224),interpolation=PIL.Image.NEAREST),
         transforms.ToTensor(),
         transforms.Normalize(c.PIXEL_MEAN, c.PIXEL_STD)]
    )
    
    return test_transform

def get_datasets(test_transform):
    test_dataset = PornDatasetA1(root=c.DATA_DIR, transform=test_transform)
    return test_dataset

def get_data(bt_size):
    test_transform = get_default_transforms()
    test_dataset = get_datasets(test_transform)
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=c.BATCH_SIZE, shuffle=False)
    return testloader
