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

def normalize(x):
    return x/sum(x)

def get_valid_sigma(mu,sigma):
    left = mu-sigma/2 if (mu-sigma > 0) else 0
    right = mu+sigma/2 if (mu+sigma < 100) else 100
    return right - left
    
def get_corrected_sigma(mu,sigma):
    valid_sigma = get_valid_sigma(mu,sigma)
    return sigma*(sigma/valid_sigma)

def get_normal_distribution(mu, sigma, adjust_sigma):
    # adicionar esse valor caso o sigma seja variavel (nao pode ser zero)
    sigma += 10**-10
    # Usar esse valor corrigido quando usar o desvio padrao das idades aparentes
    if(adjust_sigma):
        sigma = get_corrected_sigma(mu,sigma)
    bins = np.array([*range(101)])
    values = 1/(sigma * np.sqrt(2 * np.pi)) * np.exp( - (bins - mu)**2 / (2 * sigma**2) )
    return normalize(values)

def get_sigma_apparent(model, age):
    device = utils.get_device()
    if (device == "GPU") or (device == "PARALLEL"):
        model = model.to('cuda')
        if (device == "PARALLEL"):
            model = nn.DataParallel(model)
    using_gpu = (device != "CPU")
    model.eval()
    age_tensor = torch.Tensor([age])
    if (using_gpu):
        age_tensor = age_tensor.cuda()
    return model(age_tensor)[0].cpu().data.numpy()

def make_dataset(root, set_type, proportion, alpha, seed):
    images = []
    targets_dist_real = []
    targets_age = []
    image_paths = os.listdir(root)
    count = len(image_paths)
    random.seed(seed)
    random.shuffle(image_paths)
    slice_val = int(count - (count *proportion[1]))
    slices = [0, slice_val, count]
    for i in range(slices[set_type.value],slices[set_type.value+1]):
        images.append(os.path.join(root, image_paths[i]))
        real_age = int(image_paths[i].split(_)[0])
        sigma = alpha * get_sigma_apparent(model, real_age)
        targets_dist_real.append(get_normal_distribution(real_age, sigma, False))
    return images, np.asarray(targets_dist_real,dtype=np.float32), np.asarray(targets_age,dtype=np.float32)[:, np.newaxis]

class newDB(datasets.VisionDataset):
    def __init__(self, root, set_type, proportion, alpha=1, seed=1234, loader=default_loader, transforms=None, transform=None, target_transform=None, is_valid_file=None):
        super(newDB, self).__init__(root, transform=None, target_transform=None)
        self.samples, self.targets_dist_real, self.targets_age = make_dataset(root, set_type, proportion, alpha, seed)
        if len(self.samples) == 0:
            raise (RuntimeError("Found 0 files in this set"))
        self.loader = default_loader
        self.transform = transform
        self.target_transform = target_transform
        self.transforms = transform

    def __getitem__(self, index):
        path = self.samples[index]
        sample = self.loader(path)
        target_dist_real = self.targets_dist_real[index]
        target_age = self.targets_age[index]
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target_dist_real, target_age
    
    def __len__(self):
        return len(self.samples)


def get_default_transforms():
    # Define a data transformation to convert PIL images to PyTorch tensors
    # and normalize the data using the predetermined pixel means and standard
    # deviations
    train_transform = transforms.Compose(
        [transforms.Resize(256),
         #transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
         #transforms.RandomAffine((-20,20),(0.05, 0.05),(0.95, 1.05)),
         transforms.RandomOrder([
             transforms.CenterCrop(224),
             transforms.RandomHorizontalFlip(),
         ]),
         transforms.ToTensor(),
         transforms.Normalize(c.PIXEL_MEAN, c.PIXEL_STD)]
    )
    val_transform = transforms.Compose(
        [transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize(c.PIXEL_MEAN, c.PIXEL_STD)]
    )
    
def get_datasets(train_transform, val_transform, alpha, proportion):
    train_dataset = ImageRealProbDistributuion(root=c.DIR_NEW_DB, set_type=c.SET.TRAINING, transform=train_transform, alpha=alpha)
    val_dataset = ImageRealProbDistributuion(root=c.DIR_NEW_DB, set_type=c.SET.VALIDATION, transform=val_transform, alpha=alpha)
    return train_dataset, val_dataset

def get_data(bt_size, alpha, proportion):

    train_transform, val_transform, = get_default_transforms()
    train_dataset, val_dataset = get_datasets(train_transform, val_transform, alpha, proportion)
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=bt_size, shuffle=False)
    valloader = torch.utils.data.DataLoader(val_dataset, batch_size=bt_size, shuffle=True)
    return trainloader, valloader
