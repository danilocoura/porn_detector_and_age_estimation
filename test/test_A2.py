
import data_A2
from torch.utils.data import DataLoader, sampler
import config as c
import torch
import torch.nn as nn
from torch import cuda
import pandas as pd
from pyseeta import Detector
from pyseeta import Aligner
from PIL import Image, ImageDraw, ImageOps
import numpy as np
import math
from torchvision import transforms
import PIL
import os

train_on_gpu = cuda.is_available()

def test_model(model, test_loader, criterion):
    y_range = torch.Tensor(np.asarray([*range(101)])[:, np.newaxis])
    paths = []
    ages = []
    # Don't need to keep track of gradients
    with torch.no_grad():
        # Set to evaluation mode
        model.eval()

        # Validation loop
        count = 0
        for data, path in test_loader:
            # Tensors to gpu
            if train_on_gpu:
                data, path = data.cuda(), path.cuda()

            # Forward pass
            output = model(data)
            prediction = torch.mm(output,y_range)
            paths += path
            ages += prediction.numpy().tolist()
    return paths, ages

#retorna a face detectada já rotacionada com os olhos alinhados horizontalmente
def get_face(aligner, image_color, image_gray, face):
    landmarks = aligner.align(image_gray, face)
    size=0.4
    face_width = face.right - face.left
    x = landmarks[1][0] - landmarks[0][0]
    y = landmarks[1][1] - landmarks[0][1]
    degrees = math.degrees(math.atan2(y,x))
    x_center = face.left + (face_width)/2
    y_center = face.top + (face_width)/2
    if (image_color.width > image_color.height):
        border = int(image_color.width*size)
    else:
        border = int(image_color.height*size)
    image_color = ImageOps.expand(image_color,border)    
    image_color = image_color.rotate(degrees,center=(x_center+border,y_center+border))
    image_color = image_color.crop((face.left+border - face_width*size, face.top+border-face_width*size, face.right+border+face_width*size, face.bottom+border+face_width*size))
    return image_color

#varre o .csv da primeira etapa buscando todas as imagens pornograficas para procurar faces e determinar a idade
def run_test(model):
    detector = Detector()
    detector.set_min_face_size(30)
    file_names = []
    face_imgs = []
    aligner = Aligner()
    df = pd.read_csv(c.CSV_A1)
    for i in range(len(df.index)):
        file_name = df.iloc[i]['path']   
        a1 = df.iloc[i]['a1']
        if a1 == 0:
            image_color = Image.open(file_name).convert('RGB')
            image_gray = image_color.convert('L')
            faces = detector.detect(image_gray)
            for face in faces:
                face2model = get_face(aligner, image_color, image_gray, face)
                file_names.append(file_name)
                face_imgs.append(face2model)
    detector.release()
    #carrega todas as imagens pornograficas que possuem faces para serem testadas 
    test_dataloader = data_A2.get_data(c.BATCH_SIZE, file_names, face_imgs)
    criterion = nn.NLLLoss()
    paths, ages = test_model(model,test_dataloader,criterion)
    df = df.set_index('path')
    for ii, path in enumerate(paths):
        if math.isnan(df.loc[path,'a2']):
            df.loc[path,'a2'] = ages[ii][0]
        else:
            if df.loc[path,'a2'] > ages[ii][0]:
                df.loc[path,'a2'] = ages[ii][0]    
    df.to_csv("A2.csv")
#run_test()    
