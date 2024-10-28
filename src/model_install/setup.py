import os
import random
import numpy as np
import string
import torch
import pickle
import yaml

from torch.utils.data import TensorDataset

"""
    Setting
"""

def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def save_to_pkl(obj, path):
    with open(path, 'wb') as file:
        pickle.dump(obj, file, pickle.HIGHEST_PROTOCOL)


def load_from_pkl(path):
    with open(path, 'rb') as file:
        data = pickle.load(file)
    return data

def mkdirs(dirpath):
    if not os.path.exists(dirpath):
        os.makedirs(dirpath, exist_ok=True)

def read_yaml(path):
    with open(path, "r") as stream:
        config = yaml.load(stream, Loader=yaml.Loader)
    return config

def print_dataset(dataset):
    print(f"Number of samples in the trainset: {len(dataset)}")
    labels = torch.tensor([label for _, label in dataset])
    unique_labels, label_counts = torch.unique(labels, return_counts=True)

    print(f"Labels present in the trainset: {unique_labels}")
    print(f"Number of each label in the trainset: {label_counts}")


"""
    Normalized DGA Data
"""

def pad_sequences(encoded_domains, maxlen):
    domains = []
    for domain in encoded_domains:
        if len(domain) >= maxlen:
            domains.append(domain[:maxlen])
        else:
            domains.append([0]*(maxlen-len(domain))+domain)
    return np.asarray(domains)

def processing_domain(df, maxlen):
    domains = df['domain'].to_numpy()
    labels = df['label'].to_numpy()

    char2ix = {x:idx+1 for idx, x in enumerate([c for c in string.printable])}
    ix2char = {ix:char for char, ix in char2ix.items()}

    # Convert characters to int and pad
    encoded_domains = [[char2ix[y] for y in x] for x in domains]
    encoded_labels = [0 if x == 0 else 1 for x in labels]

    print(f"Number of samples: {len(encoded_domains)}")
    print(f"One-hot dims: {len(char2ix) + 1}")
    encoded_labels = np.asarray([label for idx, label in enumerate(encoded_labels) if len(encoded_domains[idx]) > 1])
    encoded_domains = [domain for domain in encoded_domains if len(domain) > 1]

    assert len(encoded_domains) == len(encoded_labels)

    padded_domains = pad_sequences(encoded_domains, maxlen)

    return padded_domains, encoded_labels

