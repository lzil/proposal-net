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
    'H': 3,
    'use_reservoir': True,
    'res_init_type': 'gaussian',
    'res_init_params': {'std': 1.5},
    'res_burn_steps': 200,
    'res_noise': 0,
    'bias': True,
    'network_delay': 0,
    'out_act': 'none',
    'stride': 1,
    'model_path': None,

    'latent_decay': .9,
    'r_latency': 1,
    's_latency': 10,
    'h_latency': 10
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

        self.activation = torch.tanh
        self.tau_x = 10

        self.n_burn_in = self.args.res_burn_steps
        self.x_seed = self.args.res_x_seed

        self.noise_std = self.args.res_noise

        self.latency = self.args.r_latency
        self.latent_idx = 0
        self.latent_arr = [None] * self.latency
        self.latent_decay = self.args.latent_decay
        self.latent_out = 0

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
        self.latent_arr[self.latent_idx] = u
        if (self.latent_idx + 1) % self.latency != 0:
            # if it's not time to act, then update latent idx, decay old output, and return it
            self.latent_idx += 1
            self.latent_out = self.latent_out * self.latent_decay
            return self.latent_out
        # otherwise it's time to do some work! finally use latent_u
        self.latent_idx = 0
        cur_decay = 1
        for i in range(1, self.latency):
            cur_decay *= self.latent_decay
            self.latent_arr[i] *= cur_decay
        # sum of geometric series of length self.latency
        geom_sum = (1 - self.latent_decay ** self.latency) / (1 - self.latent_decay)
        u = sum(self.latent_arr) / geom_sum

        # actually doing the recurrent computation
        g = self.activation(self.J(self.x) + self.W_u(u))
        # adding any inherent reservoir noise
        if self.noise_std > 0:
            gn = g + torch.normal(torch.zeros_like(g), self.noise_std)
        else:
            gn = g
        # reservoir dynamics
        delta_x = (-self.x + gn) / self.tau_x
        self.x = self.x + delta_x

        self.latent_out = self.x
        return self.x

    def reset(self, res_state=None, burn_in=True):
        self.latent_idx = -1
        self.latent_arr = [None] * self.latency
        self.latent_out = 0

        if res_state is None:
            # load specified hidden state from seed
            res_state = self.x_seed

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
        if self.args.use_reservoir:
            x = self.reservoir(u)
            z = self.W_ro(x)
        else:
            z = self.W_ro(u)
        z = self.out_act(z)

        if self.network_delay > 0:
            z_delayed = self.delay_output[self.delay_ind]
            self.delay_output[self.delay_ind] = z
            self.delay_ind = (self.delay_ind + 1) % self.network_delay
            z = z_delayed

        # z = torch.clamp(z, -2, 2)

        if not extras:
            return z
        elif self.args.use_reservoir:
            return z, {'x': x, 'u': u}
        else:
            return z, {'u': u}


    def reset(self, res_state=None):
        if self.args.use_reservoir:
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

        self.activation = torch.tanh
        self.tau_x = 10

        # weights in the recurrent net
        self.W_sim = nn.Sequential(
            nn.Linear(self.args.L + self.args.D, 16, bias=self.args.bias),
            nn.Tanh(),
            nn.Linear(16, self.args.L, bias=self.args.bias)
        )

        # dealing with latent output
        self.latency = self.args.sim_latency
        self.latent_idx = 0
        self.latent_arr = [None] * self.latency
        self.latent_decay = self.args.latent_decay
        self.latent_out = 0

    def forward(self, s, p):
        # in case we're working with a batch of unknown size
        # if len(s.shape) + 1 == len(p.shape):
        #     s = s.unsqueeze(0)
        s, p = adj_batch_dims(s, p)
        inp = torch.cat([s, p], dim=-1)

        # update latent state
        self.latent_arr[self.latent_idx] = inp
        if (self.latent_idx + 1) % self.latency != 0:
            # if it's not time to act, then update latent idx, decay old output, and return it
            self.latent_idx += 1
            self.latent_out = self.latent_out * self.latent_decay
            return self.latent_out
        # otherwise it's time to do some work! finally use latent_u
        self.latent_idx = 0
        cur_decay = 1
        for i in range(1, self.latency):
            cur_decay *= self.latent_decay
            self.latent_arr[i] *= cur_decay
        # sum of geometric series of length self.latency
        geom_sum = (1 - self.latent_decay ** self.latency) / (1 - self.latent_decay)
        inp = sum(self.latent_arr) / geom_sum

        pred = self.W_sim(inp)
        self.latent_out = pred

        return pred

class RecurrentSimulator(nn.Module):
    def __init__(self, args=DEFAULT_ARGS):
        args = fill_undefined_args(args, DEFAULT_ARGS, to_bunch=True)
        super().__init__()
        self.args = args

        if not hasattr(self.args, 'sim_x_seed'):
            self.args.sim_x_seed = np.random.randint(1e6)

        self.x_seed = self.args.sim_x_seed
        self.activation = torch.tanh
        self.tau_x = 10

        # weights in the recurrent net
        self.J = nn.Linear(self.args.N_sim, self.args.N_sim, bias=False)
        self.S_u = nn.Linear(self.args.L + self.args.D, self.args.N_sim, bias=False)
        self.S_ro = nn.Linear(self.args.N_sim, self.args.L)

        # dealing with latent output
        self.latency = self.args.sim_latency
        self.latent_idx = -1
        self.latent_arr = [None] * self.latency
        self.latent_decay = self.args.latent_decay
        self.latent_out = 0

    def forward(self, s, p):
        # in case we're working with a batch of unknown size
        # if len(s.shape) + 1 == len(p.shape):
        #     s = s.unsqueeze(0)
        s, p = adj_batch_dims(s, p)
        inp = torch.cat([s, p], dim=-1)

        # update latent state
        self.latent_arr[self.latent_idx] = inp
        if (self.latent_idx + 1) % self.latency != 0:
            # if it's not time to act, then update latent idx, decay old output, and return it
            self.latent_idx += 1
            self.latent_out = self.latent_out * self.latent_decay
            return self.latent_out
        # otherwise it's time to do some work! finally use latent_u
        self.latent_idx = 0
        cur_decay = 1
        for i in range(1, self.latency):
            cur_decay *= self.latent_decay
            self.latent_arr[i] *= cur_decay
        # sum of geometric series of length self.latency
        geom_sum = (1 - self.latent_decay ** self.latency) / (1 - self.latent_decay)
        inp = sum(self.latent_arr) / geom_sum
        

        g = self.activation(self.J(self.x) + self.S_u(inp))
        # adding any inherent reservoir noise
        # if self.noise_std > 0:
        #     gn = g + torch.normal(torch.zeros_like(g), self.noise_std)
        # else:
        #     gn = g
        # reservoir dynamics
        delta_x = (-self.x + gn) / self.tau_x
        self.x = self.x + delta_x

        self.latent_out = self.x

        pred = self.W_sim(inp)

        return pred

    def burn_in(self, steps):
        for i in range(steps):
            g = self.activation(self.J(self.x))
            delta_x = (-self.x + g) / self.tau_x
            self.x = self.x + delta_x
        self.x.detach_()

    def reset(self, res_state=None, burn_in=True):
        self.latent_idx = -1
        self.latent_arr = [None] * self.latency
        self.latent_out = 0

        if res_state is None:
            # load specified hidden state from seed
            res_state = self.x_seed

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

class MemorySimulator(nn.Module):
    def __init__(self, args=DEFAULT_ARGS):
        args = fill_undefined_args(args, DEFAULT_ARGS, to_bunch=True)
        super().__init__()
        self.args = args

        if not hasattr(self.args, 'sim_x_seed'):
            self.args.sim_x_seed = np.random.randint(1e6)

        self.x_seed = self.args.sim_x_seed
        self.activation = torch.tanh
        self.tau_x = 10

        # weights in the recurrent net
        self.J = nn.Linear(self.args.N_sim, self.args.N_sim, bias=False)
        self.S_u = nn.Linear(self.args.L + self.args.D, self.args.N_sim, bias=False)
        self.S_ro = nn.Linear(self.args.N_sim, self.args.L)

        # dealing with latent output
        self.latency = self.args.sim_latency
        self.latent_idx = -1
        self.latent_arr = [None] * self.latency
        self.latent_decay = self.args.latent_decay
        self.latent_out = 0

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
        self.sample_std = .5

        # L dimensions for state, 2 dimensions for task (desired state)
        # self.W_1 = nn.Linear(self.args.L + self.args.T, self.args.H)
        # self.W_2 = nn.Linear(self.args.H, self.args.D)

        self.W_mu = nn.Linear(self.args.L + self.args.T, self.args.H)
        self.W_var = nn.Linear(self.args.L + self.args.T, self.args.H)
        self.W_H = nn.Linear(self.args.H, self.args.D)

        self.latent = Latent(self.args.h_latency, self.args.latent_decay, nargs=1)

        # # dealing with latent output
        # self.latency = self.args.h_latency
        # self.latent_idx = -1
        # self.latent_arr = [0] * self.latency
        # self.latent_decay = self.args.latent_decay
        # self.latent_comp = None
        # # [prop, kl]
        # self.latent_out = [0, 0]

    # taken loosely from https://github.com/AntixK/PyTorch-VAE/blob/master/models/vanilla_vae.py
    def _encode(self, inp):
        mu = self.W_mu(inp)
        logvar = self.W_var(inp)
        return [mu, logvar]

    def _reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def _decode(self, z):
        return self.W_H(z)

    def forward(self, s, t):
        s, t = adj_batch_dims(s, t)
        inp = torch.cat([s, t], dim=-1)

        do_update = self.latent.next_inp(inp)

        # if self.sample_std > 0:
        #     inp = inp + torch.normal(torch.zeros_like(inp), self.sample_std)
        # h = torch.tanh(self.W_1(inp))
        # prop = self.W_2(h)

        if do_update:

            mu, log_var = self._encode(inp)
            z = self._reparameterize(mu, log_var)
            prop = self._decode(z)

            kl = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

            self.latent.next_out(prop, kl)

        return self.latent.latent_out

        # replacing current out with computation done in the past
        # if self.latent_comp is None:
        #     self.latent_comp = [torch.zeros_like(prop), 0]
        # self.latent_out = self.latent_comp
        # self.latent_comp = [prop, kl]
        # return self.latent_out

    def reset(self):
        self.latent.reset()
        # self.latent_idx = -1
        # self.latent_arr = [0] * self.latency
        # self.latent_out = [0,0]
        # self.latent_comp = None

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
        prop, kl = self.hypothesizer(self.s, t)
        if self.args.use_reservoir:
            x = self.reservoir(prop)
            z = self.W_ro(x)
        else:
            z = self.W_ro(prop)
        # clipping so movements can't be too large
        z = torch.clamp(z, -2, 2)
        self.s = self.s + z
        if extras:
            return self.s, {'z': z, 'kl': kl}
        return self.s

    def reset(self, res_state=None):
        if self.args.use_reservoir:
            self.reservoir.reset(res_state=res_state)
        self.hypothesizer.reset()
        # initial condition
        self.s = torch.zeros(self.args.Z)




# captures the hypothesis generator and simulator into a single class
class HypothesisNet(nn.Module):
    def __init__(self, args=DEFAULT_ARGS):
        super().__init__()
        args = fill_undefined_args(args, DEFAULT_ARGS, to_bunch=True)
        self.args = args

        if not hasattr(self.args, 'network_seed'):
            self.args.network_seed = random.randrange(1e6)
        self._init_vars()
        if self.args.model_path is not None:
            self.load_state_dict(torch.load(args.model_path))

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
            prop, kld = self.hypothesizer(self.s, t)
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


# so I don't have to keep writing the code for latent dynamics (e.g. it takes some time for network to operate and give result)
class Latent:
    def __init__(self, latency, decay, nargs=0):
        self.nargs = nargs
        self.latency = latency
        self.latent_idx = -1
        self.latent_arr = [0] * self.latency
        self.latent_decay = decay
        self.latent_out = []
        self.latent_comp = None

    def next_inp(self, inp):
        # update latent state
        self.latent_arr[self.latent_idx] = inp
        if (self.latent_idx + 1) % self.latency != 0:
            # if it's not time to act, then update latent idx, decay old output, and return it
            self.latent_idx += 1
            self.latent_out[0] = self.latent_out[0] * self.latent_decay
            return False
        # otherwise it's time to do some work! finally use latent_u
        self.latent_idx = 0
        cur_decay = 1
        for i in range(1, self.latency):
            cur_decay *= self.latent_decay
            self.latent_arr[i] *= cur_decay
        # sum of geometric series of length self.latency
        geom_sum = (1 - self.latent_decay ** self.latency) / (1 - self.latent_decay)
        self.inp = sum(self.latent_arr) / geom_sum

        return True

    def next_out(self, out, *args):
        # replacing current out with computation done in the past
        if self.latent_comp is None:
            self.latent_comp = [torch.zeros_like(out)] + [0] * self.nargs
        self.latent_out = self.latent_comp
        self.latent_comp = [out] + list(args)

    def reset(self):
        self.latent_idx = -1
        self.latent_arr = [0] * self.latency
        self.latent_out = [0,0]
        self.latent_comp = None

