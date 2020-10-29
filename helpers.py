
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import pdb
import random

import potentials


### RETRIEVAL

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
        p = potentials.p_none
    elif args.goals_potential == 'sin':
        p = potentials.p_sin
    elif args.goals_potential == 'slope':
        p = potentials.p_slope
    elif args.goals_potential == 'bump':
        p = potentials.p_bump
    return p

### LOSSES

# loss function for sequential goals
def goals_loss(out, targets, indices, threshold=1, update=True, lam_r=2):
    target = targets[torch.arange(targets.shape[0]),indices,:]
    if len(out.shape) > 1:
        dists = torch.norm(out - target, dim=1)
    else:
        # just one dimension so only one element in batch
        dists = torch.norm(out - target, dim=0, keepdim=True)

    done = (dists < threshold) * 1
    # update the indices while we're at it
    if update:
        indices = update_goal_indices(targets, indices, done)
    loss = torch.sum(dists) - lam_r * indices.sum()
    return loss, indices

# loss function for confidence
# labels are whether the simulator judged them to be right or wrong
def loss_confidence(conf, labels, lam_c=5, lam_w=10):
    loss = nn.MSELoss()(conf, labels.float())
    # give higher weight to those in which confidence is well-placed
    loss = lam_w * loss * (1 + lam_c * labels.long())
    return loss

# loss of the simulator in guessing target distance
def loss_simulator(out, target, lam_w=5):
    return lam_w * nn.MSELoss()(out, target)

# train the hypothesizer when a proposal is failed by the simulator
def loss_failed_prop(d_cur, d_prop, lam_w=1):
    return lam_w * nn.MSELoss()(d_cur, d_prop)


# updating indices array to get the next targets for sequential goals
def update_goal_indices(targets, indices, done):
    indices = torch.clamp(indices + done, 0, len(targets[0]) - 1)
    return indices


### DATA PROCESSING

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
