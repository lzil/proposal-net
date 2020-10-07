import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import os
import pickle
import pdb
import random
import copy
import sys

from utils import Bunch, load_rb, fill_undefined_args
from helpers import get_output_activation

DEFAULT_ARGS = {
    'L': 2,
    'T': 2,
    'D': 5,
    'N': 50,
    'Z': 2,
    'use_reservoir': True,
    'res_init_type': 'gaussian',
    'res_init_params': {'std': 1.5},
    'res_burn_steps': 200,
    'res_noise': 0,
    'bias': True,
    'network_delay': 0,
    'out_act': 'none',
    'stride': 1,
    'model_path': None
}

# SIMULATOR_ARGS.reservoir_seed = 0

# reservoir network. shouldn't be trained
class Reservoir(nn.Module):
    def __init__(self, args=DEFAULT_ARGS):
        super().__init__()
        self.args = fill_undefined_args(args, DEFAULT_ARGS, to_bunch=True)

        # if not hasattr(args, 'reservoir_seed'):
        #     self.args.reservoir_seed = random.randrange(1e6)
        if not hasattr(self.args, 'res_x_seed'):
            self.args.res_x_seed = np.random.randint(1e6)

        # self.J = nn.Linear(args.N, args.N, bias=False)
        self.activation = torch.tanh
        self.tau_x = 10

        self.n_burn_in = self.args.res_burn_steps
        self.res_x_seed = self.args.res_x_seed

        self.noise_std = args.res_noise

        # check whether we want to load reservoir from somewhere else
        # if args.model_path is not None:
        #     self.load_state_dict(torch.load(args.model_path))
        # else:
        self._init_vars()

        self.reset()

    def _init_vars(self):
        # rng_pt = torch.get_rng_state()
        # torch.manual_seed(self.args.reservoir_seed)
        self.W_u = nn.Linear(self.args.D, self.args.N, bias=False)
        self.W_u.weight.data = torch.normal(0, self.args.res_init_std, self.W_u.weight.shape) / np.sqrt(self.args.N)
        self.J = nn.Linear(self.args.N, self.args.N, bias=False)
        self.J.weight.data = torch.normal(0, self.args.res_init_std, self.J.weight.shape) / np.sqrt(self.args.N)
        # torch.set_rng_state(rng_pt)


    def burn_in(self, steps):
        for i in range(steps):
            g = self.activation(self.J(self.x))
            delta_x = (-self.x + g) / self.tau_x
            self.x = self.x + delta_x
        self.x.detach_()

    def forward(self, u):
        if type(u) is int and u == -1:
            # ensures that we don't add the bias term
            g = self.activation(self.J(self.x))
        else:
            g = self.activation(self.J(self.x) + self.W_u(u))
        # adding any inherent reservoir noise
        if self.noise_std > 0:
            gn = g + torch.normal(torch.zeros_like(g), self.noise_std)
        else:
            gn = g
        # reservoir dynamics
        delta_x = (-self.x + gn) / self.tau_x
        self.x = self.x + delta_x
        return self.x

    def reset(self, res_state=None, burn_in=True):
        if res_state is None:
            # load specified hidden state from seed
            res_state = self.res_x_seed

        if res_state == 'zero' or res_state == -1:
            # reset to 0
            self.x = torch.zeros((1, self.args.N))
        elif res_state == 'random' or res_state == -2:
            # reset to totally random value without using reservoir seed
            self.x = torch.normal(0, 1, (1, self.args.N))
        elif type(res_state) is int and res_state >= 0:
            # if any seed set, set the net to that seed and burn in
            rng_pt = torch.get_rng_state()
            torch.manual_seed(res_state)
            self.x = torch.normal(0, 1, (1, self.args.N))
            torch.set_rng_state(rng_pt)
        else:
            # load an actual particular hidden state
            # if there's an error here then highly possible that res_state has wrong form
            self.x = torch.from_numpy(res_state).float()

        if burn_in:
            self.burn_in(self.n_burn_in)

# doesn't have hypothesizer or simulator
class BasicNetwork(nn.Module):
    def __init__(self, args=DEFAULT_ARGS):
        super().__init__()
        args = fill_undefined_args(args, DEFAULT_ARGS, to_bunch=True)
        self.args = args
        
        # self.stride = args.stride
        # self.stride_step = 0
       
        if not hasattr(self.args, 'network_seed'):
            self.args.network_seed = random.randrange(1e6)
        self._init_vars()
        if self.args.model_path is not None:
            self.load_state_dict(torch.load(args.model_path))
        
        self.out_act = get_output_activation(self.args)
        self.network_delay = args.network_delay

        self.reset()

    def _init_vars(self):
        rng_pt = torch.get_rng_state()
        torch.manual_seed(self.args.network_seed)
        self.W_f = nn.Linear(self.args.L, self.args.D, bias=self.args.bias)
        if self.args.use_reservoir:
            self.reservoir = Reservoir(self.args)
            self.W_ro = nn.Linear(self.args.N, self.args.Z, bias=self.args.bias)
        else:
            self.W_ro = nn.Linear(self.args.D, self.args.Z, bias=self.args.bias)
        torch.set_rng_state(rng_pt)

    def forward(self, o, extras=False):
        # pdb.set_trace()
        # pass through the forward part
        u = self.W_f(o.reshape(-1, self.args.L))

        x = self.reservoir(u)

        # self.stride_step += 1
        # if self.stride_step % self.stride == 0:
        #     x = self.reservoir(u)
        #     self.stride_step = 0
        # else:
        #     x = self.reservoir(-1)
        #     if x.shape[0] != u.shape[0]:
        #         # to expand hidden layer to appropriate batch size
        #         mul = u.shape[0]
        #         x = x.repeat((mul, 1))

        z = self.W_ro(x)
        z = self.out_act(z)

        if self.network_delay > 0:
            z_delayed = self.delay_output[self.delay_ind]
            self.delay_output[self.delay_ind] = z
            self.delay_ind = (self.delay_ind + 1) % self.network_delay
            z = z_delayed
        if not extras:
            return z
        return z, {'x': x, 'u': u}


    def reset(self, res_state=None):
        self.reservoir.reset(res_state=res_state)
        # set up network delay mechanism. essentially a queue of length network_delay
        # with a pointer to the current index
        if self.network_delay > 0:
            self.delay_output = [None] * self.network_delay
            self.delay_ind = 0


# given state and action, predicts next state
class Simulator(nn.Module):
    def __init__(self, args=DEFAULT_ARGS):
        args = fill_undefined_args(args, DEFAULT_ARGS, to_bunch=True)
        super().__init__()
        self.args = args

        self.W_sim = nn.Sequential(
            nn.Linear(self.args.L + self.args.D, 16),
            nn.ReLU(),
            nn.Linear(16, self.args.T)
        )

    def forward(self, s, p):
        # in case we're working with a batch of unknown size
        # if len(s.shape) + 1 == len(p.shape):
        #     s = s.unsqueeze(0)
        s, p = adj_batch_dims(s, p)
        inp = torch.cat([s, p], dim=-1)
        pred = self.W_sim(inp)

        return pred

# in case we're working with a batch of unknown size
def adj_batch_dims(smaller, larger):
    if len(smaller.shape) + 1 == len(larger.shape):
        smaller = smaller.repeat((larger.shape[0], 1))
    return (smaller, larger)

# given current state and task, samples a proposal
class Hypothesizer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.sample_std = .1

        # simple one layer linear network for now, can generalize later
        # L dimensions for state, 2 dimensions for task (desired state)
        self.W_sample = nn.Linear(self.args.L + self.args.T, self.args.D)

    def forward(self, s, t):
        s, t = adj_batch_dims(s, t)
        inp = torch.cat([s, t], dim=-1)
        if self.sample_std > 0:
            inp = inp + torch.normal(torch.zeros_like(inp), self.sample_std)
        pred = self.W_sample(inp)
        return pred

# has hypothesizer
# doesn't have the simulator because the truth is just given to the network
class StateNet(nn.Module):
    def __init__(self, args=DEFAULT_ARGS):
        super().__init__()
        args = fill_undefined_args(args, DEFAULT_ARGS, to_bunch=True)
        self.args = args

        if not hasattr(self.args, 'network_seed'):
            self.args.network_seed = random.randrange(1e6)
        self._init_vars()
        if self.args.model_path is not None:
            self.load_state_dict(torch.load(args.model_path))

        self.reset()

    def _init_vars(self):
        self.hypothesizer = Hypothesizer(self.args)
        if self.args.use_reservoir:
            self.reservoir = Reservoir(self.args)
            self.W_ro = nn.Linear(self.args.N, self.args.Z, bias=self.args.bias)
        else:
            self.W_ro = nn.Linear(self.args.D, self.args.Z, bias=self.args.bias)


    def forward(self, t, extras=False):
        prop = self.hypothesizer(self.s, t)

        if self.args.use_reservoir:
            for i in range(self.args.res_frequency):
                x = self.reservoir(prop)
                prop = prop * self.args.res_input_decay

        z = self.W_ro(x)
        # clipping so movements can't be too large
        z = torch.clamp(z, -2, 2)
        self.s = self.s + z
        if extras:
            return self.s, {'z': [z]}
        return self.s

    def reset(self, res_state=None):
        if self.args.use_reservoir:
            self.reservoir.reset(res_state=res_state)
        # initial condition
        self.s = torch.zeros(self.args.Z)




# captures the hypothesis generator and simulator into a single class
class HypothesisNet(nn.Module):
    def __init__(self, args=DEFAULT_ARGS):
        super().__init__()
        args = fill_undefined_args(args, DEFAULT_ARGS, to_bunch=True)
        self.args = args

        if self.args.model_path is not None:
            self.load_state_dict(torch.load(args.model_path))
        else:
            if not hasattr(args, 'network_seed'):
                self.args.network_seed = random.randrange(1e6)
            self._init_vars()

    def _init_vars(self):
        rng_pt = torch.get_rng_state()
        torch.manual_seed(self.args.network_seed)
        self.hypothesizer = Hypothesizer(self.args)
        self.simulator = Simulator(self.args)
        if self.args.use_reservoir:
            self.reservoir = Reservoir(self.args)
            self.W_ro = nn.Linear(self.args.N, self.args.Z, bias=self.args.bias)
        else:
            self.W_ro = nn.Linear(self.args.D, self.args.Z, bias=self.args.bias)
        torch.set_rng_state(rng_pt)

    def forward(self, t, extras=False):
        fail_count = 0
        while True:
            prop = self.hypothesizer(self.s, t)
            sim = self.simulator(self.s, prop)

            # test the sim here
            #cos_ang = torch.dot(sim/sim.norm(), t/t.norm())
            cur_dist = torch.norm(t - self.s, dim=-1)
            prop_dist = torch.norm(t - sim, dim=-1)

            #dx_ratio = (cur_dist - torch.norm(t - sim)) / cur_dist
            if prop_dist < cur_dist:
                break

            # if cos_ang * dx_ratio >= 1:
            #     break
            fail_count += 1
            print('failed', cur_dist.item(), prop_dist.item())
            if fail_count >= 100:
                print('really failed here')
                pdb.set_trace()

        print('succeeded')
        pdb.set_trace()
        u = prop

        x = self.reservoir(u)
        # self.stride_step += 1
        # if self.stride_step % self.stride == 0:
        #     x = self.reservoir(u)
        #     self.stride_step = 0
        # else:
        #     x = self.reservoir(-1)
        #     if x.shape[0] != u.shape[0]:
        #         # to expand hidden layer to appropriate batch size
        #         mul = u.shape[0]
        #         x = x.repeat((mul, 1))

        z = self.W_ro(x)
        pdb.set_trace()
        fn = get_output_activation(self.args)
        z = fn(z)
        #z = nn.ReLU()(z)
        if self.network_delay == 0:
            return z, x, u
        else:
            z2 = self.delay_output[self.delay_ind]
            self.delay_output[self.delay_ind] = z
            self.delay_ind = (self.delay_ind + 1) % self.network_delay
            return z2, x, u


    def reset(self, res_state=None):
        self.reservoir.reset(res_state=res_state)
        self.s = torch.zeros(self.args.Z)
