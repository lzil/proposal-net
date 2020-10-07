
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import pdb
import random

import potentials

def get_optimizer(args, train_params):
    op = None
    if args.optimizer == 'adam':
        op = optim.Adam(train_params, lr=args.lr)
    elif args.optimizer == 'sgd':
        op = optim.SGD(train_params, lr=args.lr)
    elif args.optimizer == 'rmsprop':
        op = optim.RMSprop(train_params, lr=args.lr)
    elif args.optimizer == 'lbfgs-pytorch':
        op = optim.LBFGS(train_params, lr=0.75)
    return op

def get_criterion(args):
    if args.loss == 'mse':
        criterion = nn.MSELoss()
    elif args.loss == 'bce':
        criterion = nn.BCEWithLogitsLoss()
    elif args.loss == 'goals':
        criterion = goals_loss
    return criterion

def get_output_activation(args):
    if args.out_act == 'exp':
        fn = torch.exp
    elif args.out_act == 'relu':
        fn = nn.ReLU()
    elif args.out_act == 'none':
        fn = lambda x: x
    return fn

def get_potential(args):
    if args.goals_potential == 'none':
        p = potentials.none
    elif args.goals_potential == 'sin_xy':
        p = potentials.sin_xy
    elif args.goals_potential == 'gentle_slope':
        p = potentials.gentle_slope
    elif args.goals_potential == 'central_bump':
        p = potentials.central_bump
    return p


# loss function for sequential goals
def goals_loss(out, targets, indices, p_fn, threshold=1, update=True):
    target = targets[torch.arange(targets.shape[0]),indices,:]
    ps = []
    if len(out.shape) > 1:
        dists = torch.norm(out - target, dim=1)
        for pt in out:
            ps.append(p_fn(pt))
    else:
        # just one dimension so only one element in batch
        dists = torch.norm(out - target, dim=0, keepdim=True)
        ps = [p_fn(out)]

    done = (dists < threshold) * 1
    # update the indices while we're at it
    if update:
        indices = update_goal_indices(targets, indices, done)
    loss = torch.sum(dists) - indices.sum() + sum(ps)
    return loss, indices

# updating indices array to get the next targets for sequential goals
def update_goal_indices(targets, indices, done):
    indices = torch.clamp(indices + done, 0, len(targets[0]) - 1)
    return indices

# given batch and dset name, get the x, y pairs and turn them into Tensors
def get_x_y(batch, dset):
    if 'goals' in dset:
        x = torch.Tensor(batch)
        y = x
    else:
        x, y, _ = list(zip(*batch))
        x = torch.Tensor(x)
        y = torch.Tensor(y)

    return x, y

def get_dim(a):
    if hasattr(a, '__iter__'):
        return len(a)
    else:
        return 1
