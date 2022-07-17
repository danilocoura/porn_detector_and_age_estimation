import torch
import numpy as np
import os
import torch.nn as nn
import torch.optim as optim
import config as c
import utils
from timeit import default_timer as timer
import data_APPA_real_dynamic as data_APPA_real_dynamic
import results as rs
import models
import pandas as pd
import os

import imp

def get_net_config():
    net = models.get_model_from_name("densenet161")
    device = utils.get_device()
    if (device == "GPU") or (device == "PARALLEL"):
        net = net.to('cuda')
        if (device == "PARALLEL"):
            net = nn.DataParallel(net)
    net.load_state_dict(torch.load("densenet161_imdb_wiki_dynamic.pt"))
    criterion = nn.KLDivLoss(reduction='batchmean')
    optimizer = optim.Adam(net.parameters(),lr=c.INITIAL_LR)
    if (c.TRAINING_TYPE == c.TRAINING.SMART_DECAY):
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=c.SMART_DECAY_PATIENCE, factor=c.LR_REDUCING_FACTOR)
    elif (c.TRAINING_TYPE == c.TRAINING.DECAY_PERIOD):
        if type(c.LR_DECAY_PERIOD) is int:
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=c.LR_DECAY_PERIOD, gamma=c.LR_REDUCING_FACTOR)
        else:
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=c.LR_DECAY_PERIOD, gamma=c.LR_REDUCING_FACTOR)
    return net, criterion, optimizer, scheduler, device

def pre_train(times):

    for alpha in c.ALPHA:
        iteration_list = []
        parameter_list = []
        val_mae_list = []
        test_mae_list = []
        for i in range(times):
            net, criterion, optimizer, scheduler, device = get_net_config()
            train_loader, val_loader, test_loader = data_APPA_real_dynamic.get_data(c.BATCH_SIZE, alpha)
            val_mae, test_mae, log = train(net, criterion, optimizer, scheduler, train_loader, val_loader, test_loader, device, "dynamic_appa_real")
            if not os.path.exists("results"):
                os.mkdir("results")
            rs.save_log(log, "dynamic", alpha, i)
            iteration_list.append(i)
            parameter_list.append(alpha)
            val_mae_list.append(val_mae.item())
            test_mae_list.append(test_mae.item())
            data = {
                'iteration': iteration_list,
                'parameter': parameter_list,
                'val_mae': val_mae_list,
                'test_mae': test_mae_list
            }
            df = pd.DataFrame(data)
        rs.save_csv(df,"dynamic",alpha)

#def test(model, filename):
def test(model, test_loader):
    device = utils.get_device()
    train_on_gpu = (device != "CPU")
    '''
    if (device == "GPU") or (device == "PARALLEL"):
        model = model.to('cuda')
        if (device == "PARALLEL"):
            print("Paralel")
            model = nn.DataParallel(model)
    model.load_state_dict(torch.load(filename))
    '''
    #train_loader, val_loader, test_loader = data_APPA_real.get_data(c.BATCH_SIZE)
    y_range = torch.Tensor(np.asarray([*range(101)])[:, np.newaxis])
    model.eval()
    count = 0 
    with torch.no_grad():
        # Set to evaluation mode
        model.eval()
        mae = 0
        # Validation loop
        for data, target_dist, target_age in test_loader:
            count += c.BATCH_SIZE
            # Tensors to gpu
            if train_on_gpu:
                data, target_age, y_range = data.cuda(), target_age.cuda(), y_range.cuda()

            # Forward pass
            output = model(data)
            prediction = torch.mm(output,y_range)
            result = torch.abs(prediction.sub(target_age))
                    
            mae += torch.sum(result)
            print("{0:.2f}% - Parcial MAE {1}".format(count/len(test_loader.dataset),mae/count))
        
        mae = mae / len(test_loader.dataset)
        print("final MAE {0}".format(mae))
        return mae

def train(model, criterion, optimizer, scheduler, train_loader, val_loader, test_loader, device, save_name):
    """Train a PyTorch Model

    Params
    --------
        model (PyTorch model): cnn to train
        criterion (PyTorch loss): objective to minimize
        optimizer (PyTorch optimizier): optimizer to compute gradients of model parameters
        train_loader (PyTorch dataloader): training dataloader to iterate through
        valid_loader (PyTorch dataloader): validation dataloader used for early stopping
        save_file_name (str ending in '.pt'): file path to save the model state dict
        max_epochs_stop (int): maximum number of epochs with no improvement in validation loss for early stopping
        n_epochs (int): maximum number of training epochs
        print_every (int): frequency of epochs to print training stats

    Returns
    --------
        model (PyTorch model): trained cnn with best weights
        history (DataFrame): history of train and validation loss and accuracy
    """
    train_on_gpu = (device != "CPU")
    save_file_name = "{0}.pt".format(save_name)

    model.train()  # Set the model to training mode
    best_valid_loss = np.Inf
    best_validation = 0
    best_train = 0
    best_model = None
    best_epoch = 0
    epochs_no_improve = 0
    if not os.path.exists("results"):
        os.mkdir("results")
    log = save_name

    y_range = torch.Tensor(np.asarray([*range(101)])[:, np.newaxis])
    # Train the model on the training set NUM_EPOCHS times
    for epoch in range(1, c.NUM_EPOCHS + 1):
        print('Epoch {0} start.'.format(epoch))

        # keep track of training and validation loss each epoch
        train_loss = 0.0
        valid_loss = 0.0

        train_mae = 0
        valid_mae = 0

        # Set to training
        model.train()
        overall_start = timer()

        # Train the model on each batch in trainloader
        for ii, (data, target_dist, target_age) in enumerate(train_loader):
            # Tensors to gpu
            if train_on_gpu:
                data, target_dist, target_age, y_range = data.cuda(), target_dist.cuda(), target_age.cuda(), y_range.cuda()

            # Clear gradients
            optimizer.zero_grad()
            # Predicted outputs are log probabilities
            output = model(data) + 10**-10
            # Loss and backpropagation of gradients
            loss = criterion(torch.log(output), target_dist)
            loss.backward()

            # Update the parameters
            optimizer.step()

            # Track train loss by multiplying average loss by number of examples in batch
            train_loss += loss.item() * data.size(0)

            prediction = torch.mm(output,y_range) 
            result = torch.abs(prediction.sub(target_age)) 
            mae = torch.mean(result)
            train_mae += mae * data.size(0)

            # Track training progress
            print('Epoch: {0}\t{1:.2f}% complete. {2:.2f} seconds elapsed in epoch.\r'.format(epoch,100 * (ii + 1) / len(train_loader),timer() - overall_start))
            #model.epochs += 1

        # Don't need to keep track of gradients
        with torch.no_grad():
            # Set to evaluation mode
            model.eval()

            # Validation loop
            for data, target_dist, target_age in val_loader:
                # Tensors to gpu
                if train_on_gpu:
                    data, target_dist, target_age, y_range = data.cuda(), target_dist.cuda(), target_age.cuda(), y_range.cuda()

                # Forward pass
                output = model(data) + 10**-10

                # Validation loss
                loss = criterion(torch.log(output), target_dist)
                # Multiply average loss times the number of examples in batch
                valid_loss += loss.item() * data.size(0)

                prediction = torch.mm(output,y_range) 
                result = torch.abs(prediction.sub(target_age))
                mae = torch.mean(result)
                valid_mae += mae * data.size(0)

            # Calculate average losses
            train_loss = train_loss / len(train_loader.dataset)
            valid_loss = valid_loss / len(val_loader.dataset)

            # Calculate average mae
            train_mae = train_mae / len(train_loader.dataset)
            valid_mae = valid_mae / len(val_loader.dataset)

            log += '\n\nEpoch: {0} - LR: {1}'.format(epoch, optimizer.param_groups[0]['lr'])
            log += '\nTraining Loss: {0:.4f} \tValidation Loss: {1:.4f}'.format(train_loss,valid_loss)
            log += '\nTraining MAE: {0:.2f}\t Validation MAE: {1:.2f}'.format(train_mae,valid_mae)

            # Print training and validation results
            if (epoch + 1) % c.PRINT_EVERY == 0:
                #print('\nEpoch: {0}, LR: {1}'.format(epoch,scheduler.get_lr()))
                print('\n\nEpoch: {0} - LR: {1}'.format(epoch, optimizer.param_groups[0]['lr']))
                print('\nTraining Loss: {0:.4f} \tValidation Loss: {1:.4f}'.format(train_loss,valid_loss))
                print('\nTraining MAE: {0:.2f}\t Validation MAE: {1:.2f}'.format(train_mae,valid_mae))

            if (c.TRAINING_TYPE == c.TRAINING.SMART_DECAY): 
                scheduler.step(valid_loss)
            elif (c.TRAINING_TYPE == c.TRAINING.DECAY_PERIOD):
                scheduler.step()

            # Save the model if validation loss decreases
            if valid_loss < best_valid_loss:
                epochs_no_improve = 0
                # Load the best state dict
                # Save model
                torch.save(model.state_dict(), os.path.join("results",save_file_name))
                # Attach the optimizer
                model.optimizer = optimizer
                best_valid_loss = valid_loss
                best_validation = valid_mae
                best_epoch = epoch
            else:
                epochs_no_improve +=1
                if (c.EARLY_STOPPING and epochs_no_improve >= c.EARLY_STOPPING_PATIENCE):
                    log += '\nEarly Stopping! Total epochs: {0}. Best epoch: {1} with loss: {2:.2f} and MAE: {3:.2f}'.format(epoch,best_epoch,best_valid_loss,best_validation)  
                    print('\nEarly Stopping! Total epochs: {0}. Best epoch: {1} with loss: {2:.2f} and MAE: {3:.2f}'.format(epoch,best_epoch,best_valid_loss,best_validation))
                    total_time = timer() - overall_start
                    print('{0:.2f} total seconds elapsed. {1:.2f} seconds per epoch.'.format(total_time,total_time / (epoch+1)))
                    # Load the best state dict
                    model.load_state_dict(torch.load(os.path.join("results",save_file_name)))
                    # Attach the optimizer
                    model.optimizer = optimizer
                    test_mae = test(model, test_loader)
                    log += "\n Test MAE: {0}".format(test_mae)
                    return best_validation, test_mae, log
    # Attach the optimizer
    model.optimizer = optimizer
    # Record overall time and print out stats
    total_time = timer() - overall_start
    print('\nBest epoch: {0} with loss: {1:.2f} and MAE: {2:.2f}'.format(best_epoch,best_valid_loss,best_validation))
    log += '\n\nBest epoch: {0} with loss: {1:.2f} and MAE: {2:.2f}'.format(best_epoch,best_valid_loss,best_validation)
    print('{0:.2f} total seconds elapsed. {1:.2f} seconds per epoch.'.format(total_time,total_time / (epoch+1)))
    test_mae = test(model, test_loader)
    log += "\n Test MAE: {0}".format(test_mae)
    return best_validation, test_mae, log

