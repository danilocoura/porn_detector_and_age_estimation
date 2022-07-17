import PIL
import torch
import torchvision
from torchvision import transforms, datasets 
import torchvision.datasets
import os
import pandas as pd
import config as c
import math
import scipy
from sklearn.neighbors import KernelDensity
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F

class PornDatasetA2(datasets.VisionDataset):
    def __init__(self, file_names, images, transforms=None, transform=None, root=None):
        super(PornDatasetA2, self).__init__(root, transform=None, target_transform=None)
        self.file_names = file_names
        self.images = images
        if len(self.images) == 0:
            raise (RuntimeError("Found 0 files in this set"))
        self.transform = transform

    def __getitem__(self, index):
        path = self.file_names[index]
        sample = self.images[index]
        if self.transform is not None:
            sample = self.transform(sample)
        
        return sample, path
    
    def __len__(self):
        return len(self.images)


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

def get_datasets(test_transform, file_names, images):
    test_dataset = PornDatasetA2(file_names, images, transform=test_transform)
    return test_dataset

def get_data(bt_size, file_names, images):
    test_transform = get_default_transforms()
    test_dataset = get_datasets(test_transform, file_names, images)
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=c.BATCH_SIZE, shuffle=False)
    return testloader
