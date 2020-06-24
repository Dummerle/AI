import numpy as np
import torch
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets
from utils import log
from settings import transformation, batch_size


def load_split_test(datadir="Data/"):

    train_data = datasets.ImageFolder(datadir + "train/", transform=transformation)
    test_data = datasets.ImageFolder(datadir + "test/", transform=transformation)

    num_train = len(train_data)
    num_test = len(test_data)

    indices_train = list(range(num_train))
    indices_test = list(range(num_test))

    split_train = int(np.floor(num_train))
    split_test = int(np.floor(num_test))

    np.random.shuffle(indices_train)
    np.random.shuffle(indices_test)

    train_idx, test_idx = indices_train[:split_train], indices_test[:split_test]

    train_sampler = SubsetRandomSampler(train_idx)
    test_sampler = SubsetRandomSampler(test_idx)

    trainloader = torch.utils.data.DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
    testloader = torch.utils.data.DataLoader(test_data, sampler=test_sampler, batch_size=1)
    log("Dataset loaded")
    return trainloader, testloader

