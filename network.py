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

default_arglist = {
    'L': 1,
    'D': 5,
    'N': 20,
    'Z': 1,
    'res_init_type': 'gaussian',
    'res_init_params': {'std': 1.5},
    'res_burn_steps': 200,
    'res_noise': 0,
    'reservoir_path': None,
    'bias': True,
    'network_delay': 0,
    'out_act': 'exp',
    'stride': 1,
    'model_path': None
}
BASIC_ARGS = Bunch(**default_arglist)
SIMULATOR_ARGS = copy.deepcopy(BASIC_ARGS)
SIMULATOR_ARGS.reservoir_seed = 0

# reservoir network. shouldn't be trained
class Reservoir(nn.Module):
    def __init__(self, args=BASIC_ARGS):
        super().__init__()
        args = fill_undefined_args(copy.deepcopy(args), BASIC_ARGS)
        self.args = copy.deepcopy(args)

        if not hasattr(args, 'reservoir_seed'):
            self.args.reservoir_seed = random.randrange(1e6)
        if not hasattr(args, 'reservoir_x_seed'):
            self.args.reservoir_x_seed = np.random.randint(1e6)

        self.J = nn.Linear(args.N, args.N, bias=False)
        self.W_u = nn.Linear(args.D, args.N, bias=False)
        self.activation = torch.tanh
        self.tau_x = 10

        self.n_burn_in = self.args.res_burn_steps
        self.reservoir_x_seed = self.args.reservoir_x_seed

        self.noise_std = args.res_noise

        # check whether we want to load parts of reservoir from somewhere else
        if args.model_path is not None:
            pass
        elif args.reservoir_path is not None:
            J, W_u = load_rb(args.reservoir_path)
            self.J.weight.data = J
            self.W_u.weight.data = W_u
        else:
            self._init_J(args.res_init_type, args.res_init_params)

        self.reset()

    def _init_J(self, init_type, init_params):
        if init_type == 'gaussian':
            rng_pt = torch.get_rng_state()
            torch.manual_seed(self.args.reservoir_seed)
            self.J.weight.data = torch.normal(0, init_params['std'], self.J.weight.shape) / np.sqrt(self.args.N)
            self.W_u.weight.data = torch.normal(0, init_params['std'], self.W_u.weight.shape) / np.sqrt(self.args.N)
            torch.set_rng_state(rng_pt)
        else:
            raise NotImplementedError

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
        if res_state == 'zero':
            # reset to 0
            self.x = torch.zeros((1, self.args.N))
        elif res_state == 'random':
            # reset to totally random value without using reservoir seed
            self.x = torch.normal(0, 1, (1, self.args.N))
        elif res_state is None:
            # load specified hidden state from seed
            res_state = self.reservoir_x_seed
        
        if type(res_state) is int:
            # if any seed set, set the net to that seed and burn in
            rng_pt = torch.get_rng_state()
            torch.manual_seed(res_state)
            self.x = torch.normal(0, 1, (1, self.args.N))
            torch.set_rng_state(rng_pt)
        else:
            # load a particular hidden state
            # if there's an error here then highly possible that res_state has wrong form
            self.x = torch.from_numpy(res_state).float()

        if burn_in:
            self.burn_in(self.n_burn_in)

# doesn't have hypothesizer or simulator
class BasicNetwork(nn.Module):
    def __init__(self, args=BASIC_ARGS):
        super().__init__()
        args = fill_undefined_args(copy.deepcopy(args), BASIC_ARGS)
        self.args = args
        self.reservoir = Reservoir(args)

        self.stride = args.stride
        self.stride_step = 0

        self.W_f = nn.Linear(self.args.L, self.args.D, bias=args.bias)
        self.W_ro = nn.Linear(self.args.N, self.args.Z, bias=args.bias)

        self.network_delay = args.network_delay

        self.reset()

    def forward(self, o, extras=False):
        u = self.W_f(o.reshape(-1, self.args.L))

        self.stride_step += 1
        if self.stride_step % self.stride == 0:
            x = self.reservoir(u)
            self.stride_step = 0
        else:
            x = self.reservoir(-1)
            if x.shape[0] != u.shape[0]:
                # to expand hidden layer to appropriate batch size
                mul = u.shape[0]
                x = x.repeat((mul, 1))

        z = self.W_ro(x)
        fn = get_output_activation(self.args)
        z = fn(z)
        #z = nn.ReLU()(z)
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
        if self.network_delay != 0:
            self.delay_output = [None] * self.network_delay
            self.delay_ind = 0


# given state and action, predicts next state
class Simulator(nn.Module):
    def __init__(self, args=SIMULATOR_ARGS):
        args = fill_undefined_args(copy.deepcopy(args), SIMULATOR_ARGS)
        super().__init__()
        self.args = args

        # using 2 for now for 2D target, can generalize later
        # also only using 1 layer for now, can add more later
        self.W_sim = nn.Linear(self.args.L + self.args.D, 2)


    def forward(self, s, p):
        inp = torch.cat([s, p])
        pred = self.W_sim(inp)

        return pred

# given current state and task, samples a proposal
class Hypothesizer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.sample_std = .1

        # simple one layer linear network for now, can generalize later
        # L dimensions for state, 2 dimensions for task (desired state)
        self.W_sample = nn.Linear(self.args.L + self.args.T, self.args.D)

    def forward(self, t, s):
        inp = torch.cat([t, s], axis=1)
        inp = inp + torch.normal(torch.zeros_like(inp), self.sample_std)
        pred = self.W_sample(inp)

        return pred

# has hypothesizer
# doesn't have the simulator because the truth is just given to the network
class StateNet(nn.Module):
    def __init__(self, args=BASIC_ARGS):
        super().__init__()
        args = fill_undefined_args(copy.deepcopy(args), BASIC_ARGS)
        self.args = args

        self.hypothesizer = Hypothesizer(args)
        self.reservoir = Reservoir(args)

        self.W_ro = nn.Linear(self.args.N, self.args.Z, bias=self.args.bias)

        if args.model_path is not None:
            model = torch.load(args.model_path)
            self.load_state_dict(model)

        self.reset()

    def forward(self, t, extras=False):
        # when we are using batches, we get different shapes with initial self.s
        if len(t.shape) != len(self.s.shape):
            mul = t.shape[0]
            self.s = self.s.repeat((mul, 1))
        prop = self.hypothesizer(t, self.s)

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
        self.reservoir.reset(res_state=res_state)
        # initial condition
        self.s = torch.zeros(self.args.Z)


# captures the hypothesis generator and simulator into a single class
class HypothesisNet(nn.Module):
    def __init__(self, args=BASIC_ARGS):
        super().__init__()
        args = fill_undefined_args(copy.deepcopy(args), BASIC_ARGS)
        self.args = args

        self.hypothesizer = Hypothesizer(args)
        self.simulator = Simulator(args)
        self.reservoir = Reservoir(args)

        self.W_ro = nn.Linear(args.N, args.Z, bias=self.args.bias)

        self.reset()

    def forward(self, t):
        fail_count = 0
        while True:
            prop = self.hypothesizer(t, self.s)
            sim = self.simulator(prop, self.s)


            # test the sim here
            #cos_ang = torch.dot(sim/sim.norm(), t/t.norm())
            cur_dist = torch.norm(t - self.s)
            prop_dist = torch.norm(t - sim)

            #dx_ratio = (cur_dist - torch.norm(t - sim)) / cur_dist
            if prop_dist < cur_dist:
                break

            # if cos_ang * dx_ratio >= 1:
            #     break
            fail_count += 1
            print('failed', cos_ang,dx_ratio)
            pdb.set_trace()
            if fail_count >= 1000:
                print('really failed here')
                pdb.set_trace()

        print('succeeded')
        pdb.set_trace()
        u = prop

        self.stride_step += 1
        if self.stride_step % self.stride == 0:
            x = self.reservoir(u)
            self.stride_step = 0
        else:
            x = self.reservoir(-1)
            if x.shape[0] != u.shape[0]:
                # to expand hidden layer to appropriate batch size
                mul = u.shape[0]
                x = x.repeat((mul, 1))

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
