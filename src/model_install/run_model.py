import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
import warnings
import string
import numpy as np

from src.model_install.model import LSTMModel, LeNet, BiLSTM
from ..utils import *
import sys
sys.path.append("../")
from src.add_config import *

"""
    LSTM model
"""

char2ix = {x: idx + 1 for idx, x in enumerate([c for c in string.printable])}
maxlen = 127
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

lr = model_config['lr']
batch_size = model_config['batch_size']
epochs = client_config['num_epochs']

max_features = 101
embed_size = 64
hidden_size = 64
n_layers = 1


def pad_sequences(encoded_domains, maxlen):
    domains = []
    for domain in encoded_domains:
        if len(domain) >= maxlen:
            domains.append(domain[:maxlen])
        else:
            domains.append([0] * (maxlen - len(domain)) + domain)
    return np.asarray(domains)

def domain2tensor(domains):
    encoded_domains = [[char2ix[y] for y in domain] for domain in domains]
    padded_domains = pad_sequences(encoded_domains, maxlen)
    tensor_domains = torch.LongTensor(padded_domains)
    return tensor_domains

def decision(x):
    return x >= 0.5

def train_lstm(model, trainloader, device, **kwargs):
    batch_size = kwargs['batch_size']
    criterion = kwargs['criterion']
    optimizer = kwargs['optimizer']

    model.train()
    clip = 5
    h = model.init_hidden(domain2tensor(["0"] * batch_size))
    # batch_size = inputs.size(0) 
    # hidden = (torch.zeros(1, batch_size, 64).to(device),
    #         torch.zeros(1, batch_size, 64).to(device))


    correct = 0 
    total = 0
    running_loss = 0

    for inputs, labels in (tqdm(trainloader)):
        inputs = inputs.to(device)
        labels = labels.to(device)
        # h = model.init_hidden(domain2tensor(["0"]*batch_size))
        h = tuple([each.data for each in h])

        model.zero_grad()
        output, h = model(inputs, h)
        loss = criterion(output.squeeze(), labels.float())
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        running_loss += loss.item()
        prediction = decision(output)
        total += prediction

        correct += sum(prediction == labels)
        accuracy = correct / total
    return accuracy, loss

def test_lstm(model, testloader, device, **kwargs):
    batch_size = kwargs['batch_size']
    criterion = kwargs['criterion']
    optimizer = kwargs['optimizer']
    val_h = model.init_hidden(domain2tensor(["0"] * batch_size))
    model.eval()
    with torch.no_grad():
        eval_losses = []
        total = 0
        correct = 0

        for eval_inputs, eval_labels in tqdm(testloader):
            eval_inputs = eval_inputs.to(device)
            eval_labels = eval_labels.to(device)

            val_h = tuple([x.data for x in val_h])
            eval_output, val_h = model(eval_inputs, val_h)

            eval_prediction = decision(eval_output)
            total += len(eval_prediction)
            correct += sum(eval_prediction == eval_labels)

            eval_loss = criterion(eval_output.squeeze(), eval_labels.float())
            eval_losses.append(eval_loss.item())

    return correct / total, np.mean(eval_losses)

"""
    CNN model
"""

def train_cnn_model(model, trainloader, device, **kwargs):
    model.train()
    running_loss = 0
    running_loss = 0
    correct = 0 
    total = 0
    optimizer = kwargs['optimizer']
    criterion = kwargs['criterion']
    for inputs, lables in (tqdm(trainloader)):
        inputs, lables = inputs.to(device), lables.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, lables)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += lables.size(0)
        correct += predicted.eq(lables).sum().item()

    epoch_loss = running_loss / len(trainloader)
    epoch_acc = 100.0 * correct / total
    return epoch_acc, epoch_loss

def test_cnn_model(model, testloader, device, **kwargs):
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    criterion = kwargs['criterion']
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            _, predicted = outputs.max(1) # return maximum value in the input matrix
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    epoch_loss = test_loss / len(testloader)
    epoch_acc = 100.0 * correct / total
    return epoch_acc, epoch_loss

"""
    Client trainning side
"""

def trainning_model(trainloader, testloader, **kwargs):
    """
        model if LSTM-difference install
        device
        maxlen
        lr
        epoch
    """
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    if "model_run" not in kwargs:
        model = LSTMModel(max_features, embed_size, hidden_size, n_layers, num_classes=data_config['num_classes']).to(device)
        warnings.warn(f"Not input model. Model {model_run} is being used for trainning . . .")
    else:
        model_run = kwargs['model_run']
    
    if model_run == 'LSTMModel':
        model = LSTMModel(max_features, embed_size, hidden_size, n_layers, num_classes=data_config['num_classes']).to(device)
        # torch.save(model.state_dict(), "src/parameter/local_client.pt")
    elif model_run == 'Lenet':
        model = LeNet(num_classes=kwargs['num_classes']).to(device)
        # torch.save(model.state_dict(), "src/parameter/local_client.pt")
    
    if server_config['strategy'] == 'Local':
        if logger_config['show'] == True:
            logger.info("load model in local_client.pt")
            
        model.load_state_dict(torch.load("src/parameter/local_client.pt", map_location=device))
    else:
        model.load_state_dict(torch.load("src/parameter/client_model.pt", map_location=device))

    if "lr" not in kwargs:
        lr = 2e-5
        # warnings.warn(f"Please import learning rate - lr to trainning. Using learning rate = {lr}")
    else:
        lr = kwargs['lr']

    # if "optimizer" not in kwargs:
    #     optimizer = optim.RMSprop(params=model.parameters(), lr=lr)
    #     raise ValueError("Please import optimizer to trainning!!")
    # else:
    #     optimizer = kwargs['optimizer']

    if model_run == 'LSTMModel':
        optimizer = optim.RMSprop(params=model.parameters(), lr=lr)
    elif model_run == 'Lenet':
        optimizer = optim.RMSprop(params=model.parameters(), lr=lr)

    # if "criterion" not in kwargs:
    #     criterion = optim.RMSprop(params=model.parameters(), lr=lr)
    #     warnings.warn(f"Criterion is used {criterion} for {model_use}")
    # else:
    #     criterion = kwargs['criterion']

    if model_run == 'LSTMModel':
        # criterion = nn.BCELoss(reduction='mean') # for DGA binary classification
        criterion = nn.CrossEntropyLoss() # for DGA multi classification
    elif model_run == 'Lenet': 
        criterion = nn.CrossEntropyLoss() 

    if "epochs" not in kwargs:
        epochs = 10
        warnings.warn(f"Please import epochs to trainning. Using epochs = {epochs}")
    else:
        epochs = kwargs['epochs']

    for epoch in range(epochs):
        if model_run == 'LSTMModel':
            train_accuracy, train_loss = train_lstm(model=model, trainloader=trainloader, device=device,
                                                    criterion=criterion, optimizer=optimizer, epoch=epoch,
                                                    batch_size=batch_size)
            test_acc, test_loss = test_lstm(model=model, testloader=testloader, device=device,
                                                    criterion=criterion, optimizer=optimizer, epoch=epoch,
                                                    batch_size=batch_size)
        elif model_run == 'Lenet':
            train_accuracy, train_loss = train_cnn_model(model=model, trainloader=trainloader, device=device,
                                                         optimizer=optimizer, criterion = criterion, epoch=epoch,
                                                         batch_size=batch_size)
            test_acc, test_loss = test_cnn_model(model=model, testloader=testloader, device=device,
                                                         optimizer=optimizer, criterion = criterion, epoch=epoch,
                                                         batch_size=batch_size)

        print_log(f"Epoch: {epoch + 1}/{epochs} \n", show_time= True)
        print_log(f"Trainning \n: Acc: {train_accuracy}, Loss: {train_loss}", color_="yellow")
        # print_log(f"Testing \n: Acc: {test_acc}, Loss: {test_loss}", color_="yellow")

    return model.state_dict()







    

    

    
    

    

    
