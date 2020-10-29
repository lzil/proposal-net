import numpy as np
import torch
import torch.nn as nn

import random
import os
import pdb
import json
import sys

from network import BasicNetwork, StateNet, HypothesisNet
from utils import Bunch, load_rb

from helpers import get_potential, goals_loss, update_goal_indices, get_x_y

# extracts the correct parameters N, D, O, etc. in order to properly create a net to load into
def load_model_path(path, config):
    if type(config) is dict:
        config = Bunch(**config)
    config.model_path = path

    if config.net == 'basic':
        net = BasicNetwork(config)
    elif config.net == 'state':
        net = StateNet(config)
    elif config.net == 'hypothesis':
        net = HypothesisNet(config)
    else:
        raise NotImplementedError

    net.eval()
    return net

# given a model and a dataset, see how well the model does on it
# works with plot_trained.py
def test_model(net, config, n_tests=0):
    dset = load_rb(config.dataset)

    dset_idx = range(len(dset))
    p_fn = get_potential(config)
    criterion = nn.MSELoss()
    if n_tests != 0:
        dset_idx = sorted(random.sample(range(len(dset)), n_tests))

    dset = [dset[i] for i in dset_idx]

    is_goals_task = config.dset_type == 'goals'
    x, y = get_x_y(dset, config.dset_type)

    with torch.no_grad():
        net.reset()

        losses = []
        outs = []

        if is_goals_task:
            ins = []
            n_pts = x.shape[1]
            cur_idx = torch.zeros(x.shape[0], dtype=torch.long)
            for j in range(config.goals_timesteps):
                net_in = x[torch.arange(x.shape[0]),cur_idx,:]#.reshape(-1, net.args.L)
                ins.append(net_in)
                #pdb.set_trace()
                net_out, extras = net(net_in, extras=True)
                # net_target = net_in.reshape(-1, net.args.Z)

                trial_losses = []
                # dones = torch.zeros(x.shape[0], dtype=torch.long)

                for k in range(len(net_out)):
                    # need to add the dimension back in so the goals loss fn works
                    net_out_k = net_out[k].unsqueeze(0)
                    x_k = x[k].unsqueeze(0)
                    # need to adjust this because we're separating losses from each other
                    step_loss, cur_idx[k] = goals_loss(net_out_k, x_k, cur_idx[k], threshold=config.goals_threshold)
                    trial_losses.append(step_loss)
                    # dones[k] = done.item()
                # cur_idx = update_seq_indices(x, cur_idx, dones)

                losses.append(np.array(trial_losses))
                outs.append(net_out)
            print(cur_idx)
            goals = x
            ins = torch.stack(ins, dim=1).squeeze()


        else:
            for j in range(x.shape[1]):
                # run the step
                net_in = x[:,j].reshape(-1, net.args.L)
                net_out = net(net_in)
                outs.append(net_out)
                net_target = y[:,j].reshape(-1, net.args.Z)

                trial_losses = []
                for k in range(len(dset)):
                    step_loss = criterion(net_out[k], net_target[k])
                    trial_losses.append(step_loss)
                losses.append(np.array(trial_losses))

            ins = x
            goals = x

    losses = np.sum(losses, axis=0)
    z = torch.stack(outs, dim=1).squeeze()

    data = list(zip(dset_idx, ins, goals, z, losses))

    final_loss = np.mean(losses, axis=0)

    return data, final_loss

