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


def train(args):

    net = HypothesisNet(args)

    if not args.no_log:
        log = log_this(net.args, 'logs/sim', args.name, checkpoints=False)

    simulator = net.simulator
    if args.no_reservoir:
        layer = net.W_ro
    else:
        layer = net.reservoir

    batch_size = 10

    criterion = nn.MSELoss()
    train_params = simulator.parameters()
    optimizer = optim.Adam(train_params, lr=args.lr)

    for i in range(args.iters):

        if not args.no_reservoir:
            layer.reset('random')
        optimizer.zero_grad()

        prop = torch.Tensor(np.random.normal(0, 5, size=(batch_size, net.args.D)))
        state = torch.Tensor(np.random.normal(0, 10, size=(batch_size, net.args.L)))
        sim_out = simulator(state, prop)

        # run reservoir 10 steps, so predict 10 steps in future
        outs = []
        for j in range(args.forward_steps):
            outs.append(layer(prop))

        actions = sum(outs)

        # get state output
        layer_out = actions + state
        
        # validation makes sure performance is poor if we use someone else's output
        layer_out_val = actions.roll(1, 0) + state

        # calculate euclidean loss
        loss = criterion(layer_out, sim_out)
        loss_val = criterion(layer_out_val, sim_out)

        loss.backward()
        optimizer.step()

        if i % 50 == 0 and i != 0:
            print(f'iteration: {i} | loss {loss} | loss_val {loss_val}')

    if not args.no_log:
        save_model_path = os.path.join(log.run_dir, f'model_{log.run_id}.pth')
        save_sim_path = os.path.join(log.run_dir, f'sim_{log.run_id}.pth')
        torch.save(net.state_dict(), save_model_path)
        torch.save(simulator.state_dict(), save_sim_path)
        print(f'saved model to {save_model_path}, sim to {save_sim_path}')

def test(args):
    net = HypothesisNet(args)

    simulator = net.simulator
    reservoir = net.reservoir
    W_ro = net.W_ro

    net.eval()

    batch_size = 50

    criterion = nn.MSELoss()
    train_params = simulator.parameters()

    reservoir.reset('random')

    prop = torch.Tensor(np.random.normal(size=(batch_size, net.args.D)))
    state = torch.Tensor(np.random.normal(size=(batch_size, net.args.L)))
    sim_out = simulator(state, prop)
    print(reservoir.J.weight)

    # run reservoir 10 steps, so predict 10 steps in future
    outs = []
    for j in range(args.forward_steps):
        outs.append(W_ro(reservoir(prop)))

    actions = sum(outs)

    # get state output
    layer_out = actions + state
    sim_out_val = sim_out.roll(1, 0) + state

    # calculate euclidean loss
    loss = criterion(layer_out, sim_out)
    loss2 = criterion(layer_out, sim_out_val)


    print(f'loss {loss}, loss2 {loss2}')




if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='test')
    parser.add_argument('--res_seed', default=0, type=int)
    parser.add_argument('--res_x_seed', default=0, type=int)
    parser.add_argument('--model_path', default=None, type=str)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--iters', default=4000, type=int, help='number of iterations to train for')
    parser.add_argument('--forward_steps', default=5, type=int)

    parser.add_argument('--no_reservoir', action='store_true')

    parser.add_argument('--no_log', action='store_true')
    parser.add_argument('--test', action='store_true')
    args = parser.parse_args()

    args.use_reservoir = not args.no_reservoir

    if args.test and args.model_path is not None:
        test(args)
    else:
        train(args)