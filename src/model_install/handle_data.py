from src.model_install.model import LSTMModel, BiLSTM, LeNet

import pandas as pd
import warnings
import os
import numpy as np
from colorama import Fore, Back, Style
from sklearn.model_selection import train_test_split
from collections import defaultdict

import torch
import torchvision
from torch.utils.data import DataLoader, TensorDataset, Dataset
import torchvision.transforms as transforms

from src.add_config import server_config
from src.model_install.setup import processing_domain, setup_seed, save_to_pkl, mkdirs, print_dataset, read_yaml
import sys
sys.path.append("../")
from src.logging import *

"""
    Get DGA dataset
"""

def get_Dataset(datasetname, datapath):
    if datasetname == "FashionMnist":
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,))
        ])
        trainset = torchvision.datasets.FashionMNIST(root=datapath, train=True,
                                                        download=True, transform=transform)
        testset = torchvision.datasets.FashionMNIST(root=datapath, train=False,
                                                    download=True, transform=transform)
    elif datasetname == "Cifar10":
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        transform_test = transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        trainset = torchvision.datasets.CIFAR10(root=datapath, train=True,
                                                download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR10(root=datapath, train=False,
                                               download=True, transform=transform_test)
    elif datasetname == 'Cifar100':
        transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor(),
                                              transforms.Normalize(mean=[0.507, 0.487, 0.441],
                                                                   std=[0.267, 0.256, 0.276])])
        transform_test = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize(mean=[0.507, 0.487, 0.441],
                                                                  std=[0.267, 0.256, 0.276])])

        trainset = torchvision.datasets.CIFAR100(root=datapath, train=True,
                                                 download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR100(root=datapath, train=False,
                                                download=True, transform=transform_test)
        
    elif datasetname == "dga_data_binary":
        print(Fore.YELLOW + "DGA_dataset_binary label is using . . .")
        maxlen = 127

        data_folder = "data/dga"
        dga_types = [dga_type for dga_type in os.listdir(data_folder) if os.path.isdir(f"{data_folder}/{dga_type}")]
        print(dga_types)
        my_df = pd.DataFrame(columns=['domain', 'type', 'label'])
        for dga_type in dga_types:
            files = os.listdir(f"{data_folder}/{dga_type}")
            for file in files:
                with open(f"{data_folder}/{dga_type}/{file}", 'r') as fp:
                    domains_with_type = [[(line.strip()), dga_type, 1] for line in fp.readlines()]
                    appending_df = pd.DataFrame(domains_with_type, columns=['domain', 'type', 'label'])
                    my_df = pd.concat([my_df, appending_df], ignore_index=True)

        with open(f'{data_folder}/benign.txt', 'r') as fp:
            domains_with_type = [[(line.strip()), 'benign', 0] for line in fp.readlines()[:]] # read all file
            appending_df = pd.DataFrame(domains_with_type, columns=['domain', 'type', 'label'])
            my_df = pd.concat([my_df, appending_df], ignore_index=True)

        train_test_df, val_df = train_test_split(my_df, test_size=0.1, shuffle=True) 

        padded_domains, encoded_labels = processing_domain(train_test_df, maxlen)

        X_train, X_test, y_train, y_test = train_test_split(padded_domains, encoded_labels, test_size=0.10, shuffle=True)

        trainset = TensorDataset(torch.tensor(X_train, dtype=torch.long), torch.Tensor(y_train))
        testset = TensorDataset(torch.tensor(X_test, dtype=torch.long), torch.Tensor(y_test))
    
    elif datasetname == "dga_data":
        print(Fore.YELLOW + "DGA_dataset is using . . .")
        maxlen = 127

        data_folder = "data/dga"
        dga_types = [dga_type for dga_type in os.listdir(data_folder) if os.path.isdir(f"{data_folder}/{dga_type}")]
        print(dga_types)
        my_df = pd.DataFrame(columns=['domain', 'type', 'label'])

        current_label = 0
        for dga_type in dga_types:
            files = os.listdir(f"{data_folder}/{dga_type}")
            for file in files:
                current_label += 1
                with open(f"{data_folder}/{dga_type}/{file}", 'r') as fp:
                    domains_with_type = [[(line.strip()), dga_type, current_label] for line in fp.readlines()]
                    appending_df = pd.DataFrame(domains_with_type, columns=['domain', 'type', 'label'])
                    my_df = pd.concat([my_df, appending_df], ignore_index=True)
        logger.debug(my_df)

        with open(f'{data_folder}/benign.txt', 'r') as fp:
            domains_with_type = [[(line.strip()), 'benign', 0] for line in fp.readlines()[:]] # read all file
            appending_df = pd.DataFrame(domains_with_type, columns=['domain', 'type', 'label'])
            my_df = pd.concat([my_df, appending_df], ignore_index=True)

        train_test_df, val_df = train_test_split(my_df, test_size=0.1, shuffle=True) 

        padded_domains, encoded_labels = processing_domain(train_test_df, maxlen)

        X_train, X_test, y_train, y_test = train_test_split(padded_domains, encoded_labels, test_size=0.10, shuffle=True)

        trainset = TensorDataset(torch.tensor(X_train, dtype=torch.long), torch.Tensor(y_train))
        testset = TensorDataset(torch.tensor(X_test, dtype=torch.long), torch.Tensor(y_test))

    return trainset, testset

"""
    *** Split_dataset ***
"""

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]
        self.target = dataset.targets[self.idxs]

    def __len__(self):
        return len(self.idxs)
    
    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label
    
def split_data(dataset_use, **kwargs):

    """
    IID type:
        `iid_equal_size`
        `iid_diff_size`
    
    Non-IID type:

        `beta`:
            beta << 1: Some clients may receive the majority of the data, while other clients receive very little.
                This means that the data will be distributed very unevenly.
            beta = 1: Random assignment with moderate variation.
            beta >> 1: Data will be distributed almost evenly among clients.

        `noniid_label_distribution`
            1. Each client has all num_classes, but the samples in each class are unevenly
            distributed according to the coefficient rho, representing the percentage of samples
            from 1-2 classes in the client's dataset.
                - Data among clients is equal.
                - Data among clients is unequal, divided according to the beta distribution of Dirichlet.

        `noniid_label_quantity`        
            2. Each client randomly selects 2-3 classes, and the number of samples in these classes is also
            randomly distributed.
                - Data among clients is equal.
                - Data among clients is unequal, divided according to the beta distribution of Dirichlet.
        Condition:
            Any two clients sharing the same class will have non-overlapping samples for that class.   
    """

    if "dataset" not in kwargs:
        raise ValueError("Please input name of the dataset!!")
    else:
        datasetname = kwargs['dataset']
        print(Fore.YELLOW + f"Dataset {datasetname} is using!!")
    
    if 'num_client' not in kwargs:
        num_client = 10
        warnings.warn(
            "num_client is not being included. set num_client = 10")
    else:
        num_client = kwargs['num_client']
    
    if 'partition' not in kwargs:
        partition = 'iid_equal_size'
        warnings.warn(
            "partition: iid_equal_size")
    else:
        partition = kwargs['partition']
        
    if "beta" not in kwargs:
        beta = 0.5
        warnings.warn(
            f"partition:{partition} | beta is not provided. Set to 0.5.")
    else:
        beta = kwargs['beta']

    if 'num_classes' not in kwargs:
        raise ValueError(
            f"The num_classes parameter needs to be set for the partition {partition}.")
    else:
        num_classes = kwargs['num_classes']

    # try:
    #     num_unique_class = len(torch.unique(dataset.targets))
    # except TypeError:
    #     print('dataset.targets is not of tensor type! Proper actions are required.')
    #     exit()
    # assert num_classes == num_unique_class, f"num_classes is set to {num_classes}, but number of unique class detected in ylables are {num_unique_class}."

    if 'data_for_client' not in kwargs:
        
        data_for_client = 'FedCSP/data/sperated_data_client'
        mkdirs(data_for_client)
        warnings.warn(
            "data_client is saved in folder: sperated_data_client")
    else:
        data_for_client = kwargs['data_for_client']

    if 'data_volume_each_client' not in kwargs:
        data_volume_each_client = 'equal'
        warnings.warn(
            "data_client is saved in folder: sperated_data_client")
    else:
        data_volume_each_client = kwargs['data_volume_each_client']

    seed = 2024
    setup_seed(seed)
    num_samples = len(dataset_use)

    client_indices = []
    client_data = []
    client_targets = []
    client_dataset = TensorDataset()

    all_client_dataset = {}

    dataloader = DataLoader(dataset_use, batch_size=num_samples, shuffle=True)
    data, targets = next(iter(dataloader))

    if partition == 'iid_equal_size':
        data_per_client = num_samples // num_client
        indices = torch.randperm(num_samples)

        for cid in range(num_client):
            start_idx = cid * data_per_client
            end_idx = (cid + 1) * data_per_client if cid != num_client else num_samples
            
            client_indices = indices[start_idx: end_idx]
            client_data = data[client_indices]
            client_targets = targets[client_indices]

            client_dataset = TensorDataset(client_data, client_targets)
            all_client_dataset[f'client_{cid}']=client_dataset

            client_save_file = os.path.join(data_for_client, f"client_{cid}.pkl")
            # client_save_file = f"{data_for_client}/client_{cid}.pkl"
            save_to_pkl(client_dataset, client_save_file)

            print(f"Client {cid} data saved to {client_save_file}")
    
    elif partition == "iid_diff_size":
        """save samples in each labels to dict"""
        label_to_indices = defaultdict(list)
        for idx, label in enumerate(targets):
            label_to_indices[label.item()].append(idx)
        
        client_indices = [[] for _ in range(num_client)]
        client_data_ratios = np.random.dirichlet([beta] * num_client) * num_samples

        for label, indices in label_to_indices.items():
            np.random.shuffle(indices)
            num_data_per_client = np.array([int(ratio) for ratio in (client_data_ratios / len(targets.unique()))])
            
            start_idx = 0
            for cid in range(num_client):
                end_idx = start_idx + num_data_per_client[cid]
                client_indices[cid].extend(indices[start_idx:end_idx])
                start_idx = end_idx

        for cid in range(num_client):
            client_data = data[client_indices[cid]]
            client_targets = targets[client_indices[cid]]
            client_dataset=TensorDataset(client_data, client_targets)
            all_client_dataset[f'client_{cid}']=client_dataset

            client_save_file = os.path.join(data_for_client, f"client_{cid}.pkl")
            save_to_pkl(client_data, client_save_file)

            print(f"Client {cid} data saved to {client_save_file}")

    elif partition == "noniid_label_distribution":

        label_to_indices = defaultdict(list)
        for idx, label in enumerate(targets):
            label_to_indices[label.item()].append(idx)

        client_indices = [[] for _ in range(num_client)]

        if data_volume_each_client == 'unequal':
            for label, indices in label_to_indices.items():
                np.random.shuffle(indices)
                proportions = np.random.dirichlet([beta] * num_client)
                proportions = (np.cumsum(proportions) * len(indices)).astype(int)[:-1]

                client_splits = np.split(indices, proportions)
                for cid in range(num_client):
                    client_indices[cid].extend(client_splits[cid])

        elif data_volume_each_client == 'equal':
            for label, indices in label_to_indices.items():
                np.random.shuffle(indices)
                num_samples_per_client = len(indices) // num_client
                for cid in range(num_client):
                    start_idx = cid * num_samples_per_client
                    end_idx = (cid + 1) * num_samples_per_client if cid != num_client else len(indices)
                    client_indices[cid].extend(indices[start_idx:end_idx])
        else:
            raise ValueError("Don't know how to distribute data volume across clients!!")

        for cid in range(num_client):
            client_data = data[client_indices[cid]]
            client_targets = targets[client_indices[cid]]
            client_dataset = TensorDataset(client_data, client_targets)
            all_client_dataset[f'client_{cid}']=client_dataset

            client_save_file = os.path.join(data_for_client, f"client_{cid}.pkl")
            save_to_pkl(client_dataset, client_save_file)

            print(f"Client {cid} data saved to {client_save_file}")

    elif partition == "noniid_label_quantity":
        # Clients randomly select 2-3 classes and the number of samples in these classes varies
        label_to_indices = defaultdict(list)
        for idx, label in enumerate(targets):
            label_to_indices[label.item()].append(idx)

        for label in label_to_indices.keys():
            np.random.shuffle(label_to_indices[label])

        client_indices = [[] for _ in range(num_client)]

        # Determine how many classes each client will randomly pick (2-3)
        for cid in range(num_client):
            chosen_classes = np.random.choice(num_classes, np.random.randint(2, 4), replace=False)
            for class_idx in chosen_classes:
                if data_volume_each_client == 'unequal':
                    # Dirichlet distribution for unequal sample size
                    proportions = np.random.dirichlet([beta] * len(chosen_classes))
                    num_samples = int(proportions.sum() * len(label_to_indices[class_idx]))
                    client_indices[cid].extend(label_to_indices[class_idx][:num_samples])
                    label_to_indices[class_idx] = label_to_indices[class_idx][num_samples:]
                elif data_volume_each_client == 'equal':
                    # Equal data size among clients
                    num_samples = len(label_to_indices[class_idx]) // num_client
                    client_indices[cid].extend(label_to_indices[class_idx][:num_samples])
                    label_to_indices[class_idx] = label_to_indices[class_idx][num_samples:]
                else:
                    raise ValueError("Don't know how to distribute data volume across clients!!")

        for cid in range(num_client):
            client_data = data[client_indices[cid]]
            client_targets = targets[client_indices[cid]]
            client_dataset = TensorDataset(client_data, client_targets)
            all_client_dataset[f'client_{cid}']=client_dataset

            client_save_file = os.path.join(data_for_client + f"/client_{cid}.pkl")
            save_to_pkl(client_dataset, client_save_file)

            print(f"Client {cid} data saved to {client_save_file}")
        
    return all_client_dataset




# if __name__ == "__main__":
#     trainset, getset = get_Dataset("Cifar10", "D:\\Project\\FedCSP\\data\\images")

#     print(f"All data :")
#     print_dataset(trainset)
#     print("\n")

#     config = read_yaml('D:\\Project\\FedCSP\\config.yaml')

#     data_config = server_config
#     num_client = 10
#     all_client_dataset = split_data(trainset, datasetname = data_config['datasetname'],
#                                   data_for_client = data_config['data_for_client'], num_classes=data_config['num_classes'],
#                                   partition=data_config['partition'], data_volume_each_client = data_config['data_volume_each_client'],
#                                   beta = data_config['beta'], rho = data_config['rho'], num_client = 10)
    
    # for cid in range(num_client):
    #     data_client = all_client_dataset[f'client_{cid}']
    #     print(f"Data in client {cid} :")
    #     print_dataset(data_client)
    #     print("\n")