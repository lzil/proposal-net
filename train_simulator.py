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

from utils import log_this, load_rb, Bunch
from helpers import get_optimizer, get_criterion


if __name__ == '__main__':
    L = 2
    Z = 2
    D = 5

    config = {'simulator_seed': 0, 'reservoir_seed': 0, 'reservoir_x_seed': 0, 'D': D}

    config = Bunch(**config)
    net = HypothesisNet(config)

    # args = Bunch(L=L, Z=Z, D=D, dataset='datasets/temp.pkl', out_act='none', reservoir_seed=1)

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

        prop = torch.Tensor(np.random.normal(size=(batch_size, D)))
        state = torch.Tensor(np.random.normal(size=(batch_size, L)))
        sim_out = simulator(state, prop)

        # run reservoir 10 steps, so predict 10 steps in future
        res_outs = []
        for j in range(10):
            res_outs.append(W_ro(reservoir(prop)))

        # get state output
        res_final = res_outs[-1] + state

        diff = torch.norm(res_final - sim_out, dim=1)
        loss = criterion(diff, torch.zeros_like(diff))

        loss.backward()

        optimizer.step()

        if i % 100 == 0 and i != 0:
            print(f'iteration: {i} | loss {loss}')


    reservoir.reset()
    optimizer.zero_grad()

    prop = torch.Tensor(np.random.normal(size=(batch_size, D)))
    state = torch.Tensor(np.random.normal(size=(batch_size, L)))
    sim_out = simulator(state, prop)
    res_out = W_ro(reservoir(prop))
        
    pdb.set_trace()


    
    logging.info(f'Initialized trainer. Using optimizer {args.optimizer}')
    n_iters = 0
    if args.optimizer == 'lbfgs-scipy':
        final_loss, n_iters = trainer.optimize_lbfgs('scipy')
    elif args.optimizer == 'lbfgs-pytorch':
        final_loss, n_iters = trainer.optimize_lbfgs('pytorch')
    elif args.optimizer in ['sgd', 'rmsprop', 'adam']:
        final_loss, n_iters = trainer.train()

    if args.slurm_id is not None:
        # if running many jobs, then we gonna put the results into a csv
        csv_path = os.path.join('logs', args.name.split('_')[0] + '.csv')
        csv_exists = os.path.exists(csv_path)
        with open(csv_path, 'a') as f:
            writer = csv.writer(f, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            labels_csv = ['slurm_id', 'D', 'N', 'bias', 'seed', 'rseed', 'xseed', 'rnoise', 'dset', 'niter', 'loss']
            vals_csv = [
                args.slurm_id, args.D, args.N, args.bias, args.seed,
                args.reservoir_seed, args.reservoir_x_seed, args.reservoir_noise,
                args.dataset, n_iters, final_loss
            ]
            if args.optimizer == 'adam':
                labels_csv.extend(['lr', 'epochs'])
                vals_csv.extend([args.lr, args.n_epochs])
            elif args.optimizer == 'lbfgs-scipy':
                pass

            if not csv_exists:
                writer.writerow(labels_csv)
            writer.writerow(vals_csv)

    logging.shutdown()