# neuralnet.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 10/29/2019
# Modified by Mahir Morshed for the spring 2021 semester
# Modified by Joao Marques (jmc12) for the fall 2021 semester 

"""
This is the main entry point for MP5. You should only modify code
within this file and neuralnet_part1.py,neuralnet_leaderboard -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from utils import get_dataset_from_arrays
from torch.utils.data import DataLoader


def standardize(x):
    return (x - x.mean(dim=1, keepdim=True)) / x.std(dim=1, keepdim=True)


class NeuralNet(nn.Module):
    def __init__(self, lrate, loss_fn, in_size, out_size, momentum=0.9):
        """
        Initializes the layers of your neural network.

        @param lrate: learning rate for the model
        @param loss_fn: A loss function defined as follows:
            @param yhat - an (N,out_size) Tensor
            @param y - an (N,) Tensor
            @return l(x,y) an () Tensor that is the mean loss
        @param in_size: input dimension
        @param out_size: output dimension
        """
        super(NeuralNet, self).__init__()
        self.loss_fn = loss_fn
        self.layers = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 64, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 256, kernel_size=(2, 2), padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(256 * 7 * 7, 4)
        )
        self.optimizer = optim.SGD(self.parameters(), lr=lrate, momentum=momentum)

    def forward(self, x):
        """Performs a forward pass through your neural net (evaluates f(x)).

        @param x: an (N, in_size) Tensor
        @return y: an (N, out_size) Tensor of output from the network
        """
        x = torch.reshape(x, (len(x), 3, 31, 31))
        return self.layers(x)

    def step(self, x, y):
        """
        Performs one gradient step through a batch of data x with labels y.

        @param x: an (N, in_size) Tensor
        @param y: an (N,) Tensor
        @return L: total empirical risk (mean of losses) for this batch as a float (scalar)
        """
        x = standardize(x)
        # x = torch.reshape(x, (len(x), 3, 31, 31))
        self.optimizer.zero_grad()
        yhat = self(x)
        loss = self.loss_fn(yhat, y)
        loss.backward()
        self.optimizer.step()
        return float(torch.mean(loss).detach().cpu().numpy())


def fit(train_set, train_labels, dev_set, epochs, batch_size=100):
    """ Make NeuralNet object 'net' and use net.step() to train a neural net
    and net(x) to evaluate the neural net.

    @param train_set: an (N, in_size) Tensor
    @param train_labels: an (N,) Tensor
    @param dev_set: an (M,) Tensor
    @param epochs: an int, the number of epochs of training
    @param batch_size: size of each batch to train on. (default 100)

    This method _must_ work for arbitrary M and N.

    The model's performance could be sensitive to the choice of learning rate.
    We recommend trying different values in case your first choice does not seem to work well.

    @return losses: list of total loss at the beginning and after each epoch.
            Ensure that len(losses) == epochs.
    @return yhats: an (M,) NumPy array of binary labels for dev_set
    @return net: a NeuralNet object
    """
    net = NeuralNet(lrate=3e-3, loss_fn=F.cross_entropy, in_size=31 * 31 * 3, out_size=4, momentum=0.9)
    train_loader = DataLoader(get_dataset_from_arrays(train_set, train_labels), batch_size=batch_size, shuffle=False)

    losses = []
    for epoch in range(epochs):
        for batch in train_loader:
            train_x = batch['features']
            train_y = batch['labels']
            net.step(train_x, train_y)
        losses.append(float(F.cross_entropy(net(train_set), train_labels).detach().numpy()))
    yhats = np.argmax(net(standardize(dev_set)).detach().numpy(), axis=1)

    return losses, yhats, net
