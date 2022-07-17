from torch.utils.data import DataLoader, sampler
import data_A1
import config as c
import torch
import torch.nn as nn
from torch import cuda
import pandas as pd

train_on_gpu = cuda.is_available()

def test_model(model, test_loader, criterion):
    paths = []
    a1 = []
    # keep track of training and validation loss each epoch
    test_loss = 0.0
    test_acc = 0
    
    # Don't need to keep track of gradients
    with torch.no_grad():
        # Set to evaluation mode
        model.eval()

        # Validation loop
        count = 0
        for data, target, path in test_loader:
            count = count+1
            # Tensors to gpu
            if train_on_gpu:
                data, target, path = data.cuda(), target.cuda(), path.cuda()

            # Forward pass
            output = model(data)

            # Validation loss
            loss = criterion(output, target)
            # Multiply average loss times the number of examples in batch
            test_loss += loss.item() * data.size(0)

            # Calculate validation accuracy
            _, pred = torch.max(output, dim=1)
            correct_tensor = pred.eq(target.data.view_as(pred))
            accuracy = torch.mean(
                correct_tensor.type(torch.FloatTensor))
            # Multiply average accuracy times the number of examples
            test_acc += accuracy.item() * data.size(0)
            print(test_acc/(c.BATCH_SIZE*count))
            paths += path
            a1 += pred.numpy().tolist()

        test_loss = test_loss / len(test_loader.dataset)
        test_acc = test_acc / len(test_loader.dataset)
        
        create_dataframe(paths, a1)
    return test_loss, test_acc

#cria .csv "A1.CVS" com resultados referentes a detecção de pornografia
def create_dataframe(paths, a1):
    d = {'path':paths,'a1':a1,'a2':[None]*len(paths), 'a3':[None]*len(paths)}
    df = pd.DataFrame(d)
    df = df.set_index('path')
    df.to_csv("A1.csv")
    print(df)

def run_test(model):
    #carrega os dois tipos de dados (not_porn e porn) 
    test_dataloader = data_A1.get_data(c.BATCH_SIZE)
    criterion = nn.NLLLoss()
    #realiza o teste
    test_loss, test_acc = test_model(model,test_dataloader,criterion)
    print(test_loss)
    print(test_acc)
