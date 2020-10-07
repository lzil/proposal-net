import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import matplotlib.cm as cm

import random
import pickle
import argparse
import pdb
import re
import os
import json

from utils import load_rb, fill_undefined_args, get_config
from testers import load_model_path, test_model

# for plotting some instances of a trained model on a specified dataset

parser = argparse.ArgumentParser()
parser.add_argument('model', help='path to a model file, to be loaded into pytorch')
parser.add_argument('--dataset', help='path to a dataset of trials')
parser.add_argument('--noise', default=0, help='noise to add to trained weights')
parser.add_argument('--res_noise', default=None, type=float)
parser.add_argument('--out_act', default=None, type=str)
parser.add_argument('--stride', default=1, type=int)
parser.add_argument('-x', '--reservoir_x_init', default=None, type=str)
parser.add_argument('-a', '--test_all', action='store_true')
parser.add_argument('-n', '--no_plot', action='store_true')
parser.add_argument('-t', '--seq_goals_timesteps', type=int, default=200, help='number of steps to run seq-goals datasets for')
parser.add_argument('--seq_goals_threshold', default=1, type=float, help='seq-goals-threshold')
parser.add_argument('--dists', action='store_true', help='to plot dists for seq-goals')
args = parser.parse_args()

with open(args.model, 'rb') as f:
    model = torch.load(f)

# assuming config is in the same folder as the model
config = get_config(args.model)

if args.noise != 0:
    J = model['W_f.weight']
    v = J.std()
    shp = J.shape
    model['W_f.weight'] += torch.normal(0, v * .5, shp)

    J = model['W_ro.weight']
    v = J.std()
    shp = J.shape
    model['W_ro.weight'] += torch.normal(0, v * .5, shp)

config = fill_undefined_args(args, config, overwrite_none=True)

# net_params = {
#     'dset': args.dataset,
#     'out_act': args.out_act,
#     'stride': args.stride,
#     'res_noise': args.res_noise
# }
net = load_model_path(args.model, config=config)


# dset = load_rb(args.dataset)

# params={
#     'dset': args.dataset,
#     'reservoir_x_init': args.reservoir_x_init,
#     'seq_goals_timesteps': args.seq_goals_timesteps,
#     'seq_goals_threshold': args.seq_goals_threshold
# }


if args.test_all:
    _, loss2 = test_model(net, config)
    print('avg summed loss (all):', loss2)

if not args.no_plot:
    data, loss = test_model(net, config, n_tests=12)
    print('avg summed loss (plotted):', loss)

    run_id = '/'.join(args.model.split('/')[-3:-1])

    fig, ax = plt.subplots(3,4,sharex=True, sharey=True, figsize=(12,7))

    if 'goals' in config.dataset:
        for i, ax in enumerate(fig.axes):
            ix, x, y, z, loss = data[i]
            xr = np.arange(len(x))

            ax.axvline(x=0, color='dimgray', alpha = 1)
            ax.axhline(y=0, color='dimgray', alpha = 1)
            ax.grid(True, which='major', lw=1, color='lightgray', alpha=0.4)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['bottom'].set_visible(False)

            # pdb.set_trace()
            if args.dists:
                dists = torch.norm(z - x, dim=1)
                ax.plot(dists)

            else:
                # plot actual goal points
                n_pts = y.shape[0]
                colors = iter(cm.Oranges(np.linspace(.2, 1, n_pts)))
                for j in range(n_pts):
                    ax.scatter(y[j][0], y[j][1], color=next(colors))
                    ax.annotate(j+1, (y[j][0], y[j][1]), ha='center', va='center')
                
                # plot model output
                z = z[::2]
                n_timesteps = z.shape[0]
                ts_colors = iter(cm.Blues(np.linspace(0.3, 1, n_timesteps)))
                for j in range(n_timesteps):
                    ax.scatter(z[j][0], z[j][1], color=next(ts_colors), s=5)

            ax.tick_params(axis='both', color='white')
            ax.set_title(f'trial {ix}, avg loss {np.round(float(loss), 2)}', size='small')
            #ax.set_ylim([-2,3])

    else:
        for i, ax in enumerate(fig.axes):
            ix, x, y, z, loss = data[i]
            xr = np.arange(len(x))

            ax.axvline(x=0, color='dimgray', alpha = 1)
            ax.axhline(y=0, color='dimgray', alpha = 1)
            ax.grid(True, which='major', lw=1, color='lightgray', alpha=0.4)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['bottom'].set_visible(False)

            ax.plot(xr, x, color='coral', alpha=0.5, lw=1, label='input')
            ax.plot(xr, y, color='coral', alpha=1, lw=1, label='target')
            ax.plot(xr, z, color='cornflowerblue', alpha=1, lw=1.5, label='response')

            ax.tick_params(axis='both', color='white')
            ax.set_title(f'trial {ix}, avg loss {np.round(float(loss), 2)}', size='small')
            ax.set_ylim([-2,3])

        fig.text(0.5, 0.04, 'timestep', ha='center', va='center')
        fig.text(0.06, 0.5, 'value', ha='center', va='center', rotation='vertical')

    handles, labels = ax.get_legend_handles_labels()
    fig.suptitle(f'Final performance: {run_id}')
    fig.legend(handles, labels, loc='center right')

    plt.show()


