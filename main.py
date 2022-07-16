import torch
import test_A1 as A1
import test_A2 as A2
import torch.nn as nn
from torchvision import models
import pandas as pd
import config as c
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#Pega o modelo vazio para detecção de pornografia
def get_pretrained_densenet121():
    model = models.densenet121(pretrained=True)
    n_inputs = model.classifier.in_features
    n_classes = 2
    model.classifier = nn.Sequential(
        nn.Linear(n_inputs, 256), nn.ReLU(), nn.Dropout(0.2),
        nn.Linear(256, n_classes), nn.LogSoftmax(dim=1))
    return model 

#Pega o modelo vazio para estimativa de idade facial
def get_pretrained_densenet161():
    model = models.densenet161(pretrained=True)
    n_inputs = model.classifier.in_features
    n_classes = 101
    model.classifier = nn.Sequential(
        nn.Linear(n_inputs, 1024), nn.ReLU(), nn.Dropout(0.2),
        nn.Linear(1024, n_classes), nn.Softmax(dim=1))
    return model 

#carrega o modelo de detecção de pornografia no respectivo modelo vazio para a primeira etapa de testes   
def load_model_A1():
    device = torch.device('cpu')
    model = get_pretrained_densenet121()

    checkpoint = torch.load("densenet121_TT_LR_2_-8.pt",map_location=device)
    #state_dict =checkpoint['state_dict']

    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in checkpoint.items():
        name = k[7:] # remove 'module.' of dataparallel
        new_state_dict[name]=v

    model.load_state_dict(new_state_dict)
    return model

#carrega o modelo de estimação de idade facial no respectivo modelo vazio para a segunda etapa de testes
def load_model_A2():
    device = torch.device('cpu')
    model = get_pretrained_densenet161()

    checkpoint = torch.load("dynamic_appa_real_0.pt",map_location=device)
    #state_dict =checkpoint['state_dict']

    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in checkpoint.items():
        name = k[7:] # remove 'module.' of dataparallel
        new_state_dict[name]=v

    model.load_state_dict(new_state_dict)
    model.eval()
    return model

#gera um arquivo .csv com o resultado final dos testes
def generate_results():
    df = pd.read_csv(c.CSV_A2)
    df = df.reindex(columns = df.columns.tolist() + ['GT','pred1','pred2'])
    
    for i in range(len(df.index)):
        if (not df.iloc[i]['path'].find("porn_adult") == -1):
            df.loc[df['path'] == df.iloc[i]['path'],'GT'] = "ADT"
        elif (not df.iloc[i]['path'].find("porn_child") == -1):
            df.loc[df['path'] == df.iloc[i]['path'],'GT'] = "CHD"
        else:
            df.loc[df['path'] == df.iloc[i]['path'],'GT'] = "NOT"
    for i in range(len(df.index)):
        if (df.iloc[i]['a1'] == 1):
            df.loc[df['path'] == df.iloc[i]['path'],'pred1'] = "NOT"
        else:
            if (df.iloc[i]['a2'] == -1):
                df.loc[df['path'] == df.iloc[i]['path'],'pred1'] = "UNDEFINED"
            else:
                if (df.iloc[i]['a2'] < c.LIM_INF):
                    df.loc[df['path'] == df.iloc[i]['path'],'pred1'] = "CHD"
                elif (df.iloc[i]['a2'] >= c.LIM_SUP):
                    df.loc[df['path'] == df.iloc[i]['path'],'pred1'] = "ADT"
                else:
                    df.loc[df['path'] == df.iloc[i]['path'],'pred1'] = "UNDEFINED"
    df = df.set_index('path')
    print(df)
    df.to_csv("FINAL.csv")

#calcula e imprime matriz de confusão com resultados da detecção de pornograia 
def calculate_results_porn_not_porn():
    df = pd.read_csv(c.CSV_FINAL)
    matrix = np.zeros((2,2))
    for i in range(len(df.index)):
        matrix[get_number_porn(df.loc[i]['GT']),get_number_porn(df.loc[i]['pred1'])] +=1
    print(matrix)
    acuracia = np.diag(matrix).sum() / matrix.sum()
    print("Acurácia: {0}".format(acuracia))

#retorna o numero de acordo com o tipo de imagem (porn ou not_port)
def get_number_porn(class_name):
    if (class_name == 'NOT'):
        return 0
    else:
        return 1

#Calcula os resultados da predição
def calculate_results_pred():
    df = pd.read_csv(c.CSV_FINAL)
    matrix = np.zeros((3,4))
    for i in range(len(df.index)):
        matrix[get_number(df.loc[i]['GT']),get_number(df.loc[i]['pred1'])] +=1
    print(matrix)
    conf_df = pd.DataFrame(data=matrix, index=['NOT', 'ADULT', 'CHILD'], columns=['NOT', 'ADULT', 'CHILD', 'UNDEFINED'])
    #sns.heatmap(conf_df, annot=True, fmt='d', cbar=False)
    sns.heatmap(conf_df, annot=True, fmt='.1f', cbar=False, xticklabels=4, yticklabels=4)
    plt.savefig("conf_matx.jpg", bbox_inches='tight')
    acuracia = np.diag(matrix).sum() / (matrix.sum() - (matrix[1,3] + matrix[2,3]))
    indefinidos = (matrix[1,3] + matrix[2,3]) / matrix.sum()
    print("Acurácia: {0}".format(acuracia))
    print("Indefinidos: {0}".format(indefinidos))

#retorna o numero de acordo com o tipo de imagem (porn, not_port)
def get_number(class_name):
    if (class_name == 'NOT'):
        return 0
    elif (class_name == 'ADT'):
        return 1
    elif (class_name == 'CHD'):
        return 2
    else:
        return 3

#imprime a matriz de confusão com os resultados
def print_matrix(matrix, row_labels, col_labels):
    for i in range(len(matrix[:,0])):
        aux = ""
        for j in range(len(matrix[0,:])):
            aux += "{:>5}".format(matrix[i,j]) 
        print(aux)
#As 2 etapas do teste:
#etapa A1: determina se as imagens sao pornograficas ou nao
A1.run_test(load_model_A1())
#etapa A2: de todas as imagens pornográficas realiza-se a estimativa de idade facial, quando da existência de faces
A2.run_test(load_model_A2())

#Resultados
#Gera um resultado final (FINAL.csv)
generate_results(False)
#Calcula a acuracia da detecção de pornografia
calculate_results_porn_not_porn()
#Calcula a acurácia estimativa de idade (menor ou maior de idade)
calculate_results_pred()

