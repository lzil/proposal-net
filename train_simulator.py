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

from network import Reservoir, HypothesisNet

from utils import log_this, load_rb, Bunch, fill_undefined_args
from helpers import get_optimizer, get_criterion


def train(config):

    net = HypothesisNet(config)

    # simulator = Simulator(config)
    # reservoir = Reservoir()

    if not config.no_log:
        log = log_this(net.args, 'logs', config.name, checkpoints=False)

    simulator = net.simulator
    reservoir = net.reservoir
    W_ro = net.W_ro

    batch_size = 10

    criterion = nn.MSELoss()
    train_params = simulator.parameters()
    optimizer = optim.Adam(train_params, lr=1e-3)

    for i in range(1000):

        reservoir.reset()
        optimizer.zero_grad()

        prop = torch.Tensor(np.random.uniform(-5, 5, size=(batch_size, net.args.D)))
        state = torch.Tensor(np.random.uniform(-5, 5, size=(batch_size, net.args.L)))
        sim_out = simulator(state, prop)

        # run reservoir 10 steps, so predict 10 steps in future
        res_outs = []
        for j in range(10):
            res_outs.append(W_ro(reservoir(prop)))

        actions = sum(res_outs)

        # get state output
        res_final = actions + state
        
        res_final_val = actions.roll(1, 0) + state

        # calculate euclidean loss
        diff = torch.norm(res_final - sim_out, dim=1)
        loss = criterion(diff, torch.zeros_like(diff))

        diff_val = torch.norm(res_final_val - sim_out, dim=1)
        loss_val = criterion(diff_val, torch.zeros_like(diff_val))

        loss.backward()
        optimizer.step()

        if i % 50 == 0 and i != 0:
            print(f'iteration: {i} | loss {loss} | loss_val {loss_val}')

    if not config.no_log:
        save_model_path = os.path.join(log.run_dir, f'model_{log.run_id}.pth')
        torch.save(net.state_dict(), save_model_path)
        print(f'saved model to {save_model_path}')

def test(config):
    net = HypothesisNet(config)

    simulator = net.simulator
    reservoir = net.reservoir
    W_ro = net.W_ro

    batch_size = 50

    criterion = nn.MSELoss()
    train_params = simulator.parameters()
    optimizer = optim.Adam(train_params, lr=1e-3)

    reservoir.reset()
    optimizer.zero_grad()

    prop = torch.Tensor(np.random.normal(size=(batch_size, net.args.D)))
    state = torch.Tensor(np.random.normal(size=(batch_size, net.args.L)))
    sim_out = simulator(state, prop)
    print(reservoir.J.weight)

    # run reservoir 10 steps, so predict 10 steps in future
    res_outs = []
    for j in range(10):
        res_outs.append(W_ro(reservoir(prop)))

    actions = sum(res_outs)

    # get state output
    res_final = actions + state
    sim_out_val = sim_out.roll(1, 0) + state

    # calculate euclidean loss
    diff = torch.norm(res_final - sim_out, dim=1)
    loss = criterion(diff, torch.zeros_like(diff))

    diff2 = torch.norm(res_final - sim_out_val, dim=1)
    loss2 = criterion(diff2, torch.zeros_like(diff))


    print(f'loss {loss}, loss2 {loss2}')




if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='sim_test')
    parser.add_argument('--sim_seed', default=0, type=int)
    parser.add_argument('--res_seed', default=0, type=int)
    parser.add_argument('--res_x_seed', default=0, type=int)
    parser.add_argument('--model_path', default=None, type=str)

    parser.add_argument('--no_log', action='store_true')
    parser.add_argument('--test', action='store_true')
    args = parser.parse_args()

    if args.test and args.model_path is not None:
        test(args)
    else:
        train(args)