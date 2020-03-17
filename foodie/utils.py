import sys
import logging
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from matplotlib import pyplot as plt


def imshow(t):
    t -= t.min()
    t /= t.max()
    plt.imshow(t.permute(1, 2, 0))
    plt.show()


def train_test_loader(dataset, train_ratio=0.9, **options):
    """
    Given a PyTorch dataset, return train and test loader from randomly sampled indices
    """
    data_len = len(dataset)
    idxs = list(range(data_len))
    train_idxs = np.random.choice(idxs, size=int(
        train_ratio * data_len), replace=False)
    test_idxs = list(set(idxs) - set(train_idxs))
    train_loader = DataLoader(
        dataset, sampler=SubsetRandomSampler(train_idxs), **options)
    test_loader = DataLoader(
        dataset, sampler=SubsetRandomSampler(test_idxs), **options)
    return train_loader, test_loader


def setup_custom_logger(name):
    formatter = logging.Formatter(fmt='%(asctime)s %(levelname)-8s %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')
    handler = logging.FileHandler('log.txt', mode='w')
    handler.setFormatter(formatter)
    screen_handler = logging.StreamHandler(stream=sys.stdout)
    screen_handler.setFormatter(formatter)
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    logger.addHandler(screen_handler)
    return logger
