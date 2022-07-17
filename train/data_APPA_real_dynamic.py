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
import utils

class Regressor(torch.nn.Module):
    def __init__(self):
        super(Regressor, self).__init__()
        self.hidden1 = torch.nn.Linear(1, 80)   # hidden layer
        self.hidden2 = torch.nn.Linear(80, 70)   # hidden layer
        self.hidden3 = torch.nn.Linear(70, 60)   # hidden layer
        self.hidden4 = torch.nn.Linear(60, 40)   # hidden layer
        self.hidden5 = torch.nn.Linear(40, 40)   # hidden layer
        self.hidden6 = torch.nn.Linear(40, 20)   # hidden layer
        self.predict = torch.nn.Linear(20, 1)   # output layer

        self.dropout = torch.nn.Dropout(0.0)

    def forward(self, x):
        x = F.relu(self.hidden1(x))
        x = self.dropout(x)
        x = F.relu(self.hidden2(x))      # activation function for hidden layer
        x = self.dropout(x)
        x = F.relu(self.hidden3(x))      # activation function for hidden layer
        x = self.dropout(x)
        x = F.relu(self.hidden4(x))      # activation function for hidden layer
        x = self.dropout(x)
        x = F.relu(self.hidden5(x))      # activation function for hidden layer
        x = self.dropout(x)
        x = F.relu(self.hidden6(x))      # activation function for hidden layer
        x = self.predict(x)             # linear output
        return x

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

def make_dataset_real(root, set_type, seed, alpha, only_face):
    model = Regressor()
    print(model)
    model.load_state_dict(torch.load(os.path.join("regressor","regressor.pth.pt")))
    images = []
    targets_dist_real = []
    targets_age = []
    if (set_type == c.SET.TRAINING):
        set_type_str = "train"
    elif (set_type == c.SET.VALIDATION):
        set_type_str = "val"
    elif (set_type == c.SET.TEST):
        set_type_str = "test"
    else:
       raise (RuntimeError("Wrong set type"))
    df = pd.read_csv(os.path.join(root,"gt_avg_{0}.csv").format(set_type_str)).sample(frac=1, random_state=seed)
    #df = pd.read_csv(os.path.join(root,"gt_avg_{0}_red.csv").format(set_type_str)).sample(frac=1, random_state=seed)
    for i in range(len(df.index)):
        file_name = df.iloc[i]['file_name']
        if (only_face):
            file_name += "_face.jpg"
        images.append(os.path.join(root,set_type_str,file_name))
        real_age = df.iloc[i]['real_age']
        sigma = alpha * get_sigma_apparent(model, real_age)
        targets_dist_real.append(get_normal_distribution(real_age, sigma, False))
        #targets_dist_real.append(get_normal_distribution(real_age, sigma, True))
        targets_age.append(df.iloc[i]['real_age'])
    return images, np.asarray(targets_dist_real,dtype=np.float32), np.asarray(targets_age,dtype=np.float32)[:, np.newaxis]


class ImageRealProbDistributuion(datasets.VisionDataset):
    def __init__(self, root, set_type, alpha=1, transforms=None, transform=None, target_transform=None, is_valid_file=None, loader=default_loader, only_face=True, seed=0):
        super(ImageRealProbDistributuion, self).__init__(root, transforms=None, transform=None, target_transform=None)
        self.samples, self.targets_dist_real, self.targets_age = make_dataset_real(root, set_type, seed, alpha, only_face)
        if len(self.samples) == 0:
            raise (RuntimeError("Found 0 files in this set"))
        self.only_face = only_face
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
    test_transform = transforms.Compose(
        [transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize(c.PIXEL_MEAN, c.PIXEL_STD)]
    )

    return train_transform, val_transform, test_transform

def get_datasets(train_transform, val_transform, test_transform, alpha):
    train_dataset = ImageRealProbDistributuion(root=c.APPA_REAL_DIR, set_type=c.SET.TRAINING, transform=train_transform, alpha=alpha)
    val_dataset = ImageRealProbDistributuion(root=c.APPA_REAL_DIR, set_type=c.SET.VALIDATION, transform=val_transform, alpha=alpha)
    test_dataset = ImageRealProbDistributuion(root=c.APPA_REAL_DIR, set_type=c.SET.TEST, transform=test_transform, alpha=alpha)
    return train_dataset, val_dataset, test_dataset

def get_data(bt_size, alpha):

    train_transform, val_transform, test_transform = get_default_transforms()
    train_dataset, val_dataset, test_dataset = get_datasets(train_transform, val_transform, test_transform, alpha)
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=bt_size, shuffle=True)
    valloader = torch.utils.data.DataLoader(val_dataset, batch_size=bt_size, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=bt_size, shuffle=True)
    return trainloader, valloader, testloader
