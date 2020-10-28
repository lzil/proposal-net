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
from helpers import get_output_activation, loss_confidence, loss_simulator

DEFAULT_ARGS = {
    'L': 2,
    'T': 2,
    'D': 5,
    'N': 50,
    'Z': 2,
    'H': 3,
    'use_reservoir': True,
    'res_init_std': 1.5,
    'res_burn_steps': 200,
    'res_noise': 0,
    'bias': True,
    'network_delay': 0,
    'out_act': 'none',
    'model_path': None,

    'h_type': 'variational',
    's_type': 'ff',

    'latent_decay': .9,
    'r_latency': 1,
    's_latency': 10,
    'h_latency': 10,
    's_steps': 5,

    'res_path': None,
    'sim_path': None
}

# reservoir network. shouldn't be trained
class Reservoir(nn.Module):
    def __init__(self, args=DEFAULT_ARGS):
        super().__init__()
        self.args = fill_undefined_args(args, DEFAULT_ARGS, to_bunch=True)

        if not hasattr(self.args, 'res_seed'):
            self.args.res_seed = random.randrange(1e6)
        if not hasattr(self.args, 'res_x_seed'):
            self.args.res_x_seed = np.random.randint(1e6)

        self.tau_x = 10
        self.x_seed = self.args.res_x_seed
        self.noise_std = self.args.res_noise

        self._init_vars()
        self.reset()

    def _init_vars(self):
        rng_pt = torch.get_rng_state()
        torch.manual_seed(self.args.res_seed)
        self.W_u = nn.Linear(self.args.D, self.args.N, bias=False)
        self.W_u.weight.data = torch.normal(0, self.args.res_init_std, self.W_u.weight.shape) / np.sqrt(self.args.D)
        self.J = nn.Linear(self.args.N, self.args.N, bias=False)
        self.J.weight.data = torch.normal(0, self.args.res_init_std, self.J.weight.shape) / np.sqrt(self.args.N)
        self.W_ro = nn.Linear(self.args.N, self.args.Z, bias=False)
        # print(self.J.weight.data[0])
        torch.set_rng_state(rng_pt)

        if self.args.res_path is not None:
            self.load_state_dict(torch.load(self.args.res_path))

    def burn_in(self, steps):
        for i in range(steps):
            g = torch.tanh(self.J(self.x))
            delta_x = (-self.x + g) / self.tau_x
            self.x = self.x + delta_x
        self.x.detach_()

    # extras currently doesn't do anything. maybe add x val, etc.
    def forward(self, u, extras=False):
        g = torch.tanh(self.J(self.x) + self.W_u(u))
        # adding any inherent reservoir noise
        if self.noise_std > 0:
            gn = g + torch.normal(torch.zeros_like(g), self.noise_std)
        else:
            gn = g
        delta_x = (-self.x + gn) / self.tau_x
        self.x = self.x + delta_x

        z = self.W_ro(self.x)

        return z

    def reset(self, res_state=None, burn_in=True):
        if res_state is None:
            # load specified hidden state from seed
            res_state = self.x_seed

        if type(res_state) is np.ndarray:
            # load an actual particular hidden state
            # if there's an error here then highly possible that res_state has wrong form
            self.x = torch.as_tensor(res_state).float()
        elif type(res_state) is torch.Tensor:
            self.x = res_state
        elif res_state == 'zero' or res_state == -1:
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
            print('not any of these types, something went wrong')
            pdb.set_trace()

        if burn_in:
            self.burn_in(self.args.res_burn_steps)

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
        else:
            self.W_ro = nn.Linear(self.args.D, self.args.Z, bias=self.args.bias)
        torch.set_rng_state(rng_pt)

    def forward(self, o, extras=False):
        # pdb.set_trace()
        # pass through the forward part
        u = self.W_f(o.reshape(-1, self.args.L))
        if self.args.use_reservoir:
            z = self.reservoir(u)
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
            return z, {'x': self.x, 'u': u}
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

        self.W_sim = nn.Sequential(
            nn.Linear(self.args.L + self.args.D, 16, bias=self.args.bias),
            nn.Tanh(),
            nn.Linear(16, self.args.L, bias=self.args.bias)
        )

        if args.sim_path is not None:
            self.load_state_dict(torch.load(args.sim_path))

    def forward(self, s, p):
        # in case we're working with a batch of unknown size
        # if len(s.shape) + 1 == len(p.shape):
        #     s = s.unsqueeze(0)
        # s, p = adj_batch_dims(s, p)
        inp = torch.cat([s, p], dim=-1)

        pred = self.W_sim(inp)
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

class VariationalHypothesizer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.sample_std = .5
        # number of proposals to make
        self.n_props = 3
        # how far apart proposals have to be in at least one dimension
        self.props_threshold = 0.1

        # 2x size hidden layer for mean and logvar
        self.encoder = nn.Sequential(
            nn.Linear(self.args.L + self.args.T, self.args.H * 2),
            nn.Tanh(),
            nn.Linear(self.args.H * 2, self.args.H * 2 )
        )
        self.decoder = nn.Sequential(
            nn.Linear(self.args.H * 2, self.args.H),
            nn.Tanh(),
            nn.Linear(self.args.H, self.args.D + 1)
        )
        # self.latent = Latent(self.args.h_latency, self.args.latent_decay, nargs=2)

    def forward(self, s, t):
        inp = torch.cat([s, t], dim=-1)

        # taken loosely from https://github.com/AntixK/PyTorch-VAE/blob/master/models/vanilla_vae.py
        # encoding
        h = self.encoder(inp)
        mu = h[:,:self.args.H]
        lvar = h[:,self.args.H:]
        # reparameterization
        std = torch.exp(0.5 * lvar)

        distro = torch.distributions.normal.Normal(mu, std)

        zs = []
        rands = []
        while(len(zs) < self.n_props):
            # generating the proposal
            eps = torch.randn_like(std)
            z = eps * std + mu
            # calc stats for previous z's and lprobs for network input
            cdfs = distro.cdf(z)
            lprobs = distro.log_prob(z)
            z_lprobs = torch.cat([z, lprobs], dim=-1)
            # test to see if it's too close to any previous z's
            is_valid = True
            for j in zs:
                # check if within 10% of all dimensions
                diffs = torch.abs(cdfs - j[1])
                is_valid = torch.all(diffs - self.props_threshold > 0)
                if not is_valid:
                    break

            if is_valid:
                zs.append([torch.cat([z, lprobs], dim=-1), cdfs])

        # don't bother with the cdfs
        zs = [z[0] for z in zs]

        # decoding
        prop_arr = []
        extras_arr = []
        for i in range(self.n_props):

            prop_conf = self.decoder(z_lprobs)
            prop = torch.tanh(prop_conf[:,:-1])
            # prop = prop_conf[:,:-1]
            conf = torch.sigmoid(prop_conf[:,-1])
            # conf = torch.sigmoid(-torch.max(h[:,self.args.H:], dim=-1)[0])
            # calc KL for loss
            kl = -0.5 * torch.sum(1 + lvar - mu ** 2 - lvar.exp(), dim=1)

            extras = {
                'conf': conf,
                'kl': kl
            }
            prop_arr.append(prop)
            extras_arr.append(extras)

        return prop_arr, extras_arr

class FFHypothesizer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.sample_std = .5

        # L dimensions for state, 2 dimensions for task (desired state)
        self.W_1 = nn.Linear(self.args.L + self.args.T, self.args.H)
        self.W_2 = nn.Linear(self.args.H, self.args.D)

    def forward(self, s, t):
        # s, t = adj_batch_dims(s, t)
        inp = torch.cat([s, t], dim=-1)


        # if self.sample_std > 0:
        #     inp = inp + torch.normal(torch.zeros_like(inp), self.sample_std)
        # h = torch.tanh(self.W_1(inp))
        # prop = self.W_2(h)

        prop = self.W_2(torch.tanh(self.W_1(inp)))
        self.latent.step_out(prop)

        return prop, torch.zeros(1), torch.zeros(1)

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

        self.latent = Latent(self.args.h_latency, self.args.latent_decay)
        self.reset()

    def _init_vars(self):
        if self.args.h_type == 'variational':
            self.hypothesizer = VariationalHypothesizer(self.args)
        elif self.args.h_type == 'ff':
            self.hypothesizer = FFHypothesizer(self.args)
        else:
            raise NotImplementedError

        if self.args.use_reservoir:
            self.reservoir = Reservoir(self.args)
        else:
            self.W_ro = nn.Linear(self.args.D, self.args.Z, bias=self.args.bias)

    def forward(self, t, extras=False):
        if self.s is None:
            self._init_states(t)
        t = self._adj_input_dim(t)

        if self.cur_h is None:
            prop, extras = self.hypothesizer(self.s, t)
            self.cur_h = prop, extras['kl']

        h_done = self.latent.step()
        if h_done:
            self.p = self.cur_h
            self.cur_h = None

        prop, kl = self.p
        if self.args.use_reservoir:
            z = self.reservoir(prop)
        else:
            z = self.W_ro(prop)
        # clipping so movements can't be too large
        z = torch.clamp(z, -2, 2)
        self.s = self.s + z
        if extras:
            return self.s, {'z': z, 'kl': sum(kl), 'prop': prop}
        return self.s

    def reset(self, res_state=None):
        if self.args.use_reservoir:
            self.reservoir.reset(res_state=res_state)
        # initial condition
        self.s = None
        self.p = None
        self.cur_h = None

    def _init_states(self, t):
        if len(t.shape) == 2:
            self.s = torch.zeros(t.shape[0], self.args.Z)
            self.p = [torch.zeros(t.shape[0], self.args.D), torch.zeros(1), torch.zeros(1)]
        elif len(t.shape) == 1:
            self.s = torch.zeros(1, self.args.Z)
            self.p = [torch.zeros(1, self.args.D), torch.zeros(1), torch.zeros(1)]
        else:
            print('input t failed somehow')
            pdb.set_trace()

    def _adj_input_dim(self, t):
        if len(t.shape) == 2:
            return t
        elif len(t.shape) == 1:
            return t.unsqueeze(0)
        else:
            print('input t failed somehow')
            pdb.set_trace()


# captures the hypothesis generator and simulator into a single class
class HypothesisNet(nn.Module):
    def __init__(self, args=DEFAULT_ARGS):
        super().__init__()
        args = fill_undefined_args(args, DEFAULT_ARGS, to_bunch=True)
        self.args = args

        if not hasattr(self.args, 'network_seed'):
            self.args.network_seed = random.randrange(1e6)
        self._init_vars()
        self.log_h_yes = Latent(1, 0.9)
        self.log_s_yes = Latent(1, 0.9)
        self.log_conf = Latent(1, 0.9)

        self.switch = False
        self.fails = 0

        self.reset()

    def _init_vars(self):
        rng_pt = torch.get_rng_state()
        torch.manual_seed(self.args.network_seed)
        # hypothesizer
        if self.args.h_type == 'variational':
            self.hypothesizer = VariationalHypothesizer(self.args)
        elif self.args.h_type == 'ff':
            self.hypothesizer = FFHypothesizer(self.args)
        else:
            raise NotImplementedError
        # simulator
        self.simulator = Simulator(self.args)
        # reservoir
        if self.args.use_reservoir:
            self.reservoir = Reservoir(self.args)
        else:
            self.W_ro = nn.Linear(self.args.D, self.args.Z, bias=self.args.bias)
        torch.set_rng_state(rng_pt)

        # in case reservoir seed/path overwrites
        if self.args.use_reservoir:
            self.reservoir._init_vars()

        if self.args.model_path is not None:
            self.load_state_dict(torch.load(self.args.model_path))


    def forward(self, t, extras=False):
        # print(self.reservoir.J.weight)
        # pdb.set_trace()
        if self.s is None:
            self._init_states(t)
        t = self._adj_input_dim(t)

        lconf = []
        lsim = []

        for i in range(len(t)):
            # number of predicted steps forward should not be too large (e.g. greater than H time) or this will break
            if self.old_s[i] is not None:
                self.old_s[i][0] -= 1
                if self.old_s[i][0] == 0:
                    lsim.append(loss_simulator(self.s[i], self.old_s[i][1]))
                    self.old_s[i] = None

            cur_s = self.s[i].unsqueeze(0)
            cur_t = t[i].unsqueeze(0)
            self.h_latent[i].add_input(cur_s, cur_t)

            if self.c[i] == 'h':
                if self.cur_h[i] is None:
                    # no hypothesis, so make a hypothesis now
                    # inp = self.h_latent[i].get_input()
                    # using very current data instead of lagged data
                    prop_arr, extras_arr = self.hypothesizer(cur_s, cur_t)
                    kls = [e['kl'][0] for e in extras_arr]
                    confs = [e['conf'][0] for e in extras_arr]
                    self.cur_h_arr[i] = list(zip(prop_arr, kls, confs))
                    self.cur_h_ind[i] = 0

                    # using lagged data
                    # self.cur_h[i] = self.hypothesizer(*inp)

                    # old simulation doesn't matter cus we now got a new hypothesis
                    self.cur_s[i] = None

                # did our hypothesis finish computing?
                # we should set this value to something small to start with
                h_done = self.h_latent[i].step()
                if h_done:
                    # the singular current proposal we are considering
                    self.cur_h[i] = self.cur_h_arr[i][self.cur_h_ind[i]]
                    prop, kl, conf = self.cur_h[i]
                    # 1 if we are confident, 0 if not and want to use simulator
                    if conf is None:
                        conf = torch.tensor(1.)
                    self.log_conf.add_input(conf.item())
                    eps = torch.rand_like(conf)
                    do_sim = eps > conf
                    if do_sim and not self.switch:
                        # not confident enough, do a simulation
                        # print('h refused')
                        self.c[i] = 's'
                        # stop the current movement (all 0s) because we're not really sure what to do next
                        self.p[i] = [self.p[i][0]*0, 0, 0]
                        self.log_h_yes.add_input(0)
                    else:
                        # confident! just do it
                        self.log_h_yes.add_input(1)
                        self.p[i] = self.cur_h[i]
                        self.cur_h[i] = None

                else:
                    # it didn't finish computing yet, so nothing's gonna happen
                    pass

            # don't use elif so we immediately move to simulation
            if self.c[i] == 's':
                cur_s = self.s[i].unsqueeze(0)
                # p doesn't need to be unsqueezed cus it's in a list already
                cur_prop = self.cur_h[i][0]

                # there's nothing in the current s so we gon process something
                if self.cur_s[i] is None:
                    if self.switch:
                        self.cur_s[i] = True
                    else:
                        # we want to calculate simulator wrt goals loss at some point
                        # instead of just calculating whether we get closer physically

                        pred_prop = self.simulator(cur_s, cur_prop)
                        # pred_cur = self.simulator(cur_s, self.p[i][0])
                        dist_prop = torch.norm(t[i] - pred_prop[0])
                        dist_cur = torch.norm(t[i] - cur_s)
                        self.cur_s[i] = dist_prop < dist_cur
                        self.s_prop[i] = pred_prop

                s_done = self.s_latent[i].step()
                if s_done:
                    # confidence loss compares confidence and whether it was approved or not
                    lconf.append(10 * loss_confidence(self.cur_h[i][2], self.cur_s[i]))
                    if self.cur_s[i]:
                        # it works! go with it
                        self.log_s_yes.add_input(1)
                        self.old_s[i] = [self.args.s_steps, self.s_prop[i][0]]
                        self.p[i] = self.cur_h[i]
                        self.cur_s[i] = None
                        self.s_prop[i] = None
                        # we can develop a new hypothesis now
                        self.cur_h[i] = None
                        self.cur_h_arr[i] = None

                    else:
                        # didn't work :(
                        self.log_s_yes.add_input(0)
                        # self.old_s_prop[i][self.old_s_prop_idx[i]] = pred_cur

                        # move on to the next proposal
                        self.cur_h_ind[i] += 1
                        # rejected all proposals, try again
                        if self.cur_h_ind[i] >= len(self.cur_h_arr[i]):
                            self.cur_h[i] = None
                            self.cur_h_arr[i] = None
                            self.fails += 1
                    
                    self.c[i] = 'h'

        # self.p consists of [proposal, kl, confidence]
        p = torch.cat([p[0] for p in self.p])
        kl = [p[1] for p in self.p]

        if self.args.use_reservoir:
            z = self.reservoir(p)
        else:
            z = self.W_ro(p)
        z = torch.clamp(z, -2, 2)
        self.s = self.s + z

        if extras:
            return self.s, {'z': z, 'prop': p, 'kl': sum(kl), 'lconf': sum(lconf), 'lsim': sum(lsim)}
        return self.s

    def reset(self, res_state=None):
        if self.args.use_reservoir:
            self.reservoir.reset(res_state=res_state)
        self.s = None
        self.p = None

    def _init_states(self, t):
        if len(t.shape) == 1:
            bs = 1
        elif len(t.shape) == 2:
            bs = t.shape[0]
        else:
            print('input t failed somehow')
            pdb.set_trace()

        self.s = torch.zeros(bs, self.args.Z) # the current state
        self.c = ['h'] * bs # whether we're currently on H or S mode
        self.p = [[torch.zeros(1, self.args.D), torch.tensor(0.), torch.tensor(0.)] for i in range(bs)] # the current proposal (to output)
        self.cur_h = [None] * bs # the current hypothesis (to process via S)
        self.cur_s = [None] * bs # whether we've approved the current hypothesis
        self.cur_h_arr = [None] * bs
        self.cur_h_ind = [0] * bs
        self.h_latent = [Latent(self.args.h_latency, self.args.latent_decay) for i in range(bs)] # h inputs
        self.s_latent = [Latent(self.args.s_latency, self.args.latent_decay) for i in range(bs)] # s inputs
        self.old_s = [None] * bs
        self.s_prop = [None] * bs

    def _adj_input_dim(self, t):
        if len(t.shape) == 2:
            return t
        elif len(t.shape) == 1:
            return t.unsqueeze(0)
        else:
            print('input t failed somehow')
            pdb.set_trace()


# in case we're working with a batch of unknown size
def adj_batch_dims(smaller, larger):
    if len(smaller.shape) + 1 == len(larger.shape):
        smaller = smaller.repeat((larger.shape[0], 1))
    return (smaller, larger)

def adj_inp_dims(inp):
    if inp.shape == 0:
        inp = inp.unsqueeze(0)


class Latent:
    def __init__(self, latency, decay):
        self.latency = latency
        self.decay = decay
        self.reset()

    def step(self):
        self.latent_idx += 1
        if self.latent_idx == self.latency:
            self.latent_idx = 0
            return True
        else:
            return False

    def add_input(self, *inp):
        # check to make sure dimensions are right
        if len(self.latent_arr) == 0:
            self.inp_len = len(inp)
        else:
            assert self.inp_len == len(inp)
        # update the state
        if self.inp_len == 1:
            self.latent_arr.append(inp[0])
        else:
            self.latent_arr.append(list(inp))

    def get_input(self, clear=True):
        n_inps = len(self.latent_arr)
        assert n_inps > 0
        cur_decay = 1
        if self.inp_len == 1:
            for i in range(n_inps-2, -1, -1):
                cur_decay *= self.decay
                self.latent_arr[i] = self.latent_arr[i] * cur_decay
            # sum of geometric series of length self.latency
            geom_sum = (1 - self.decay ** n_inps) / (1 - self.decay)
            inp = sum(self.latent_arr) / geom_sum
        else:
            for i in range(n_inps-2, -1, -1):
                cur_decay *= self.decay
                for j in range(self.inp_len):
                    self.latent_arr[i][j] = self.latent_arr[i][j] * cur_decay
            # sum of geometric series of length self.latency
            geom_sum = (1 - self.decay ** n_inps) / (1 - self.decay)
            inp = [sum(i) / geom_sum for i in list(zip(*self.latent_arr))]

        if clear:
            self.latent_arr = []

        return inp

    def reset(self):
        self.latent_idx = 0
        self.latent_arr = []

