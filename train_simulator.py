import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.optimize import minimize

import os
import argparse
import pdb
import sys
import pickle
import logging
import random
import csv
import math
import json

from network import Reservoir, Simulator, Hypothesizer, HypothesisNet

from utils import log_this, load_rb, Bunch, fill_undefined_args
from helpers import get_optimizer, get_criterion


def train(config):

    net = HypothesisNet(config)

    log = log_this(net.args, 'logs', config.name, checkpoints=False)

    simulator = net.simulator
    reservoir = net.reservoir
    W_ro = net.W_ro

    batch_size = 10

    criterion = nn.MSELoss()
    train_params = simulator.parameters()
    optimizer = optim.Adam(train_params, lr=1e-3)

    for i in range(500):

        reservoir.reset()
        optimizer.zero_grad()

        prop = torch.Tensor(np.random.normal(size=(batch_size, net.args.D)))
        state = torch.Tensor(np.random.normal(size=(batch_size, net.args.L)))
        sim_out = simulator(state, prop)

        # run reservoir 10 steps, so predict 10 steps in future
        res_outs = []
        for j in range(10):
            res_outs.append(W_ro(reservoir(prop)))

        # get state output
        res_final = res_outs[-1] + state

        # calculate euclidean loss
        diff = torch.norm(res_final - sim_out, dim=1)
        loss = criterion(diff, torch.zeros_like(diff))

        loss.backward()
        optimizer.step()

        if i % 100 == 0 and i != 0:
            print(f'iteration: {i} | loss {loss}')

    save_model_path = os.path.join(log.run_dir, f'model_{log.run_id}.pth')
    torch.save(net.state_dict(), save_model_path)
    print(f'saved model to {save_model_path}')




if __name__ == '__main__':

    config = {'simulator_seed': 0, 'reservoir_seed': 0, 'reservoir_x_seed': 0}
    

    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='sim_test')
    args = parser.parse_args()

    config = fill_undefined_args(args, config)

    train(config)