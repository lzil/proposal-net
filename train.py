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
import copy

from network import BasicNetwork, StateNet, Reservoir, HypothesisNet

from utils import log_this, load_rb, fill_undefined_args, get_config
from helpers import get_optimizer, get_criterion, get_potential, goals_loss, update_goal_indices, get_x_y

class Trainer:
    def __init__(self, args):
        super().__init__()

        self.args = args

        if self.args.net == 'basic':
            self.net = BasicNetwork(self.args)
        elif self.args.net == 'state':
            self.net = StateNet(self.args)
        elif self.args.net == 'hypothesis':
            self.net = HypothesisNet(self.args)

        # picks which parameters to train and which not to train
        self.n_params = {}
        self.train_params = []
        self.not_train_params = []
        logging.info('Training the following parameters:')
        for k,v in self.net.named_parameters():
            # k is name, v is weight
            found = False
            # filtering just for the parts that will be trained
            for part in self.args.train_parts:
                if part in k:
                    logging.info(f'  {k}')
                    self.n_params[k] = (v.shape, v.numel())
                    self.train_params.append(v)
                    found = True
                    break
            if not found:
                self.not_train_params.append(k)
        logging.info('Not training:')
        for k in self.not_train_params:
            logging.info(f'  {k}')

        self.criterion = get_criterion(self.args)
        self.optimizer = get_optimizer(self.args, self.train_params)
        self.dset = load_rb(self.args.dataset)
        self.potential = get_potential(self.args)

        # if using separate training and test sets, separate them out
        if not self.args.same_test:
            np.random.shuffle(self.dset)
            cutoff = round(.9 * len(self.dset))
            self.train_set = self.dset[:cutoff]
            self.test_set = self.dset[cutoff:]
            logging.info(f'Using separate training ({cutoff}) and test ({len(self.dset) - cutoff}) sets.')
        else:
            self.train_set = self.dset
            self.test_set = self.dset
        
        self.log_interval = self.args.log_interval
        if not self.args.no_log:
            self.log = self.args.log
            self.run_id = self.args.log.run_id
            self.vis_samples = []
            self.csv_path = open(os.path.join(self.log.run_dir, f'losses_{self.run_id}.csv'), 'a')
            self.writer = csv.writer(self.csv_path, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            self.writer.writerow(['ix', 'avg_loss'])
            self.plot_checkpoint_path = os.path.join(self.log.run_dir, f'checkpoints_{self.run_id}.pkl')
            self.save_model_path = os.path.join(self.log.run_dir, f'model_{self.run_id}.pth')

    def log_model(self, ix=0):
        # saving all checkpoints takes too much space so we just save one model at a time, unless we explicitly specify it
        if self.args.log_checkpoint_models:
            self.save_model_path = os.path.join(self.log.checkpoint_dir, f'model_{ix}.pth')
        elif os.path.exists(self.save_model_path):
            os.remove(self.save_model_path)
        torch.save(self.net.state_dict(), self.save_model_path)

    def log_checkpoint(self, ix, x, y, z, total_loss, avg_loss):
        self.writer.writerow([ix, avg_loss])
        self.csv_path.flush()

        self.log_model(ix)

        # we can save individual samples at each checkpoint, that's not too bad space-wise
        self.vis_samples.append([ix, x, y, z, total_loss, avg_loss])
        if os.path.exists(self.plot_checkpoint_path):
            os.remove(self.plot_checkpoint_path)
        with open(self.plot_checkpoint_path, 'wb') as f:
            pickle.dump(self.vis_samples, f)

    def train_iteration(self, x, y):
        self.net.reset()
        self.optimizer.zero_grad()

        outs = []
        total_loss = torch.tensor(0.)

        # ins is actual input into the network
        # targets is desired output
        # outs is output of network
        if self.args.dset_type == 'goals':
            ins = []
            l_other = {'kl': 0, 'lconf': 0, 'lsim': 0, 'lfprop': 0, 'lp': 0}
            targets = x
            cur_idx = torch.zeros(x.shape[0], dtype=torch.long)
            for j in range(self.args.goals_timesteps):
                net_out, step_loss, cur_idx, extras = self.run_iter_goal(x, cur_idx)
                # what we need to record for logging
                ins.append(extras['in'])
                outs.append(net_out[-1].detach().numpy())
                total_loss += step_loss

                if 'kl' in extras and extras['kl'] is not None:
                    l_other['kl'] += extras['kl']
                if 'lconf' in extras and extras['lconf'] is not None:
                    l_other['lconf'] += extras['lconf']
                if 'lsim' in extras and extras['lsim'] is not None:
                    l_other['lsim'] += extras['lsim']
                if 'lp' in extras and extras['lp'] is not None:
                    l_other['lp'] += extras['lp']
                # if 'lfprop' in extras and extras['lfprop'] is not None:
                #     l_other['lfprop'] += extras['lfprop']

            ins = torch.cat(ins)

        else:
            ins = x
            targets = y
            for j in range(x.shape[1]):
                net_out, step_loss, extras = self.run_iter_traj(x[:,j], y[:,j])
                if np.isnan(step_loss.item()):
                    return -1, (net_out, extras)
                total_loss += step_loss
                outs.append(net_out[-1].item())

        total_loss.backward()
        self.optimizer.step()

        etc = {
            'ins': ins,
            'targets': targets,
            'outs': outs,
            'prop': extras['prop'],
        }
        etc.update(l_other)
        if self.args.dset_type == 'goals':
            etc['indices'] = cur_idx

        return total_loss, etc

    # runs an iteration where we want to match a certain trajectory
    def run_iter_traj(self, x, y):
        net_in = x.reshape(-1, self.args.L)
        net_out, extras = self.net(net_in, extras=True)
        net_target = y.reshape(-1, self.args.Z)
        step_loss = self.criterion(net_out, net_target)

        return net_out, step_loss, extras

    # runs an iteration where we want to hit a certain goal (dynamic input)
    def run_iter_goal(self, x, indices):
        x_goal = x[torch.arange(x.shape[0]),indices,:]
        
        net_in = x_goal.reshape(-1, self.args.L)
        net_out, extras = self.net(net_in, extras=True)
        # the target is actually the input
        step_loss, new_indices = goals_loss(net_out, x, indices, threshold=self.args.goals_threshold)
        # it'll be None if we just started, or if we're not doing variational stuff

        # non-goals related losses
        # if net_out.shape[0] != 1:
        #     pdb.set_trace()
        extras['lp'] = self.potential(net_out).sum()
        step_loss += extras['lp']
        if 'kl' in extras and extras['kl'] is not None:
            step_loss += extras['kl']
        if 'lconf' in extras and extras['lconf'] is not None:
            step_loss += extras['lconf']
        if 'lsim' in extras and extras['lsim'] is not None:
            step_loss += extras['lsim']
        # if 'lfprop' in extras and extras['lfprop'] is not None:
        #     step_loss += extras['lfprop']

        extras.update({'in': net_in})

        return net_out, step_loss, new_indices, extras

    def test(self, n=0):
        if n != 0:
            assert n <= len(self.test_set)
            batch_idxs = np.random.choice(len(self.test_set), n)
            batch = [self.test_set[i] for i in batch_idxs]
        else:
            batch = self.test_set

        x, y = get_x_y(batch, self.args.dataset)

        with torch.no_grad():
            self.net.reset()
            total_loss = torch.tensor(0.)

            if self.args.dset_type == 'goals':
                cur_idx = torch.zeros(x.shape[0], dtype=torch.long)
                for j in range(self.args.goals_timesteps):
                    _, step_loss, cur_idx, _ = self.run_iter_goal(x, cur_idx)
                    total_loss += step_loss

            else:
                for j in range(x.shape[1]):
                    _, step_loss, _ = self.run_iter_traj(x[:,j], y[:,j])
                    total_loss += step_loss

        etc = {}
        if self.args.dset_type == 'goals':
            etc['indices'] = cur_idx

        return total_loss.item() / len(batch), etc

    def train(self, ix_callback=None):
        ix = 0

        its_p_epoch = len(self.train_set) // self.args.batch_size
        logging.info(f'Training set size {len(self.train_set)} | batch size {self.args.batch_size} --> {its_p_epoch} iterations / epoch')

        # for convergence testing
        max_abs_grads = []
        running_min_error = float('inf')
        running_no_min = 0

        running_loss = 0.0
        # running_mag = 0.0
        ending = False
        for e in range(self.args.n_epochs):
            np.random.shuffle(self.train_set)
            epoch_idx = 0
            while epoch_idx < its_p_epoch:
                epoch_idx += 1
                batch = self.train_set[(epoch_idx-1) * self.args.batch_size:epoch_idx * self.args.batch_size]
                if len(batch) < self.args.batch_size:
                    break
                ix += 1

                x, y = get_x_y(batch, self.args.dataset)
                loss, etc = self.train_iteration(x, y)

                if ix_callback is not None:
                    ix_callback(loss, etc)

                if loss == -1:
                    logging.info(f'iteration {ix}: is nan. ending')
                    ending = True
                    break

                running_loss += loss.item()
                # mag = max([torch.max(torch.abs(p.grad)) for p in self.train_params])
                # running_mag += mag             

                if ix % self.log_interval == 0:
                    outs = etc['outs']
                    
                    z = np.stack(outs).squeeze()
                    # avg of the last 50 trials
                    avg_loss = running_loss / self.args.batch_size / self.log_interval
                    test_loss, test_etc = self.test(n=30)
                    # avg_max_grad = running_mag / self.log_interval
                    log_arr = [
                        f'iteration {ix}',
                        f'train loss {avg_loss:.3f}',
                        # f'max abs grad {avg_max_grad:.3f}',
                        f'test loss {test_loss:.3f}'
                    ]
                    # calculating average index reached for goals task
                    if self.args.dset_type == 'goals':
                        avg_index = test_etc['indices'].float().mean().item()
                        log_arr.append(f'avg index {avg_index:.3f}')
                    if self.args.net == 'hypothesis':
                        ha = self.net.log_h_yes.get_input()
                        sa = self.net.log_s_yes.get_input()
                        conf = self.net.log_conf.get_input()
                        lconf, lsim, kl, lp = etc['lconf'], etc['lsim'], etc['kl'], etc['lp']
                        log_arr.append(f'hyp_app {ha:.3f}')
                        log_arr.append(f'sim_app {sa:.3f}')
                        log_arr.append(f'conf {conf:.3f}')
                        log_arr.append(f'lconf {lconf:.3f}')
                        log_arr.append(f'lsim {lsim:.3f}')
                        log_arr.append(f'lp {lp:.3f}')
                        # log_arr.append(f'kl {kl:.3f}')
                    log_str = '\t| '.join(log_arr)
                    logging.info(log_str)

                    if not self.args.no_log:
                        self.log_checkpoint(ix, etc['ins'].numpy(), etc['targets'].numpy(), z, running_loss, avg_loss)
                    running_loss = 0.0
                    running_mag = 0.0

                    # convergence based on no avg loss decrease after patience samples
                    if self.args.conv_type == 'patience':
                        if test_loss < running_min_error:
                            running_no_min = 0
                            running_min_error = test_loss
                        else:
                            running_no_min += self.log_interval
                        if running_no_min > self.args.patience:
                            logging.info(f'iteration {ix}: no min for {args.patience} samples. ending')
                            ending = True
                    # elif self.args.conv_type == 'grad':
                    #     if avg_max_grad < self.args.grad_threshold:
                    #         logging.info(f'iteration {ix}: max absolute grad < {args.grad_threshold}. ending')
                    #         ending = True
                if ending:
                    break
            logging.info(f'Finished dataset epoch {e+1}')
            if ending:
                break

        if not self.args.no_log:
            # for later visualization of outputs over timesteps
            with open(os.path.join(self.log.run_dir, f'checkpoints_{self.run_id}.pkl'), 'wb') as f:
                pickle.dump(self.vis_samples, f)

            self.csv_path.close()

        final_loss, etc = self.test()
        logging.info(f'END | iterations: {(ix // self.log_interval) * self.log_interval} | test loss: {final_loss}')
        return final_loss, ix

def parse_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-L', type=int, default=2, help='latent input dimension')
    parser.add_argument('-T', type=int, default=2, help='task dimension')
    parser.add_argument('-D', type=int, default=5, help='intermediate dimension')
    parser.add_argument('-N', type=int, default=50, help='number of neurons in reservoir')
    parser.add_argument('-Z', type=int, default=2, help='output dimension')

    parser.add_argument('-H', type=int, default=3, help='hypothesizer hidden dimension')

    parser.add_argument('--net', type=str, default='hypothesis', choices=['basic', 'state', 'hypothesis'])
    parser.add_argument('--h_type', type=str, default='variational', choices=['ff', 'variational'])
    parser.add_argument('--s_type', type=str, default='ff', choices=['ff', 'recurrent'])

    parser.add_argument('-d', '--dataset', type=str, default='datasets/goals_2d_1.pkl')
    parser.add_argument('--same_test', action='store_true', help='use same set for test/train')
    parser.add_argument('--train_parts', type=str, nargs='+', default=[''])
    
    # make sure model_config path is specified if you use any paths! it ensures correct dimensions, bias, etc.
    # parser.add_argument('--model_config_path', type=str, default=None, help='config path corresponding to model load path')
    parser.add_argument('--model_path', type=str, default=None, help='start training from certain model. superseded by below')
    # parser.add_argument('--Wro_path', type=str, default=None, help='start training from certain Wro')
    # parser.add_argument('--Wf_path', type=str, default=None, help='start training from certain Wf')
    parser.add_argument('--res_path', type=str, default=None, help='saved, probably trained, reservoir. should be saved with seed tho')
    parser.add_argument('--sim_path', type=str, default=None, help='saved, probably trained, simulator')

    # seeds
    parser.add_argument('--seed', type=int, help='seed for everything else')
    parser.add_argument('--network_seed', type=int, help='seed for the network')
    parser.add_argument('--res_seed', type=int, help='seed for reservoir')
    parser.add_argument('--res_x_seed', type=int, default=0, help='seed for reservoir init hidden states')

    # res_seed will be overwritten by res_path every time
    
    parser.add_argument('--no_reservoir', action='store_true', help='leave out the reservoir completely')
    parser.add_argument('--no_bias', action='store_true')

    parser.add_argument('--res_init_std', type=float, default=1.5)
    
    parser.add_argument('--res_burn_steps', type=int, default=100, help='number of steps for reservoir to burn in')
    parser.add_argument('--network_delay', type=int, default=0)
    parser.add_argument('--res_noise', type=float, default=0)

    # so backprop doesn't completely go crazy we might want to truncate BPTT
    parser.add_argument('--latent_decay', type=float, default=.8, help='proportion to keep from the last state')
    parser.add_argument('--r_latency', type=int, default=1, help='how many operation steps it takes to move one step')
    # parser.add_argument('--r_input_decay', type=float, default=1, help='decay of res input for each step w/o input')
    parser.add_argument('--h_latency', type=int, default=5, help='how many operation steps it takes to move one step')
    parser.add_argument('--s_latency', type=int, default=2, help='how many operation steps it takes to move one step')
    parser.add_argument('--s_steps', type=int, default=5, help='number of forward steps to predict for s')
    
    parser.add_argument('--out_act', type=str, default='none', help='output activation')    

    # goals parameters
    parser.add_argument('--goals_timesteps', type=int, default=200, help='num timesteps to run seq goals dataset for')
    parser.add_argument('--goals_threshold', type=float, default=1, help='threshold for detection for seq goals')
    parser.add_argument('--goals_potential', type=str, default='none')

    # optimization parameters
    parser.add_argument('--optimizer', choices=['adam', 'sgd', 'rmsprop', 'lbfgs-scipy', 'lbfgs-pytorch'], default='adam')
    parser.add_argument('--loss', type=str, default='mse')

    # lbfgs-scipy arguments
    parser.add_argument('--maxiter', type=int, default=10000, help='limit to # iterations. lbfgs-scipy only')

    # adam arguments
    parser.add_argument('--batch_size', type=int, default=1, help='size of minibatch used')
    parser.add_argument('--conv_type', type=str, choices=['patience', 'grad'], default='patience', help='how to determine convergence. adam only')
    parser.add_argument('--patience', type=int, default=2000, help='stop training if loss doesn\'t decrease. adam only')
    # parser.add_argument('--grad_threshold', type=float, default=1e-4, help='stop training if grad is less than certain amount. adam only')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate. adam only')
    parser.add_argument('--n_epochs', type=int, default=10, help='number of epochs to train for. adam only')

    

    # parser.add_argument('-x', '--reservoir_x_init', type=str, default=None, help='other seed options for reservoir')

    parser.add_argument('--no_log', action='store_true')
    parser.add_argument('--log_interval', type=int, default=50)
    parser.add_argument('--log_checkpoint_models', action='store_true')

    parser.add_argument('--name', type=str, default='test')
    parser.add_argument('--param_path', type=str, default=None)
    parser.add_argument('--slurm_id', type=int, default=None)

    args = parser.parse_args()
    args.bias = not args.no_bias
    args.use_reservoir = not args.no_reservoir
    return args

def adjust_args(args):
    # don't use logging.info before we initialize the logger!! or else stuff is gonna fail

    # dealing with slurm. do this first!! before anything else other than seed setting, which we want to override
    if args.slurm_id is not None:
        from parameters import apply_parameters
        args = apply_parameters(args.param_path, args)

    # in case we are loading from a model
    # if we don't use this we might end up with an error when loading model
    if args.model_path is not None:
        config = get_config(args.model_path)
        args = fill_undefined_args(args, config, overwrite_none=True)
        enforce_same = ['N', 'D', 'L', 'Z', 'T', 'net', 'bias', 'use_reservoir']
        for v in enforce_same:
            if v in config and args.__dict__[v] != config[v]:
                print(f'Warning: based on config, changed {v} from {args.__dict__[v]} -> {config[v]}')
                args.__dict__[v] = config[v]

    # shortcut for specifying train everything including reservoir
    if args.train_parts == ['all']:
        args.train_parts = ['']

    # # output activation depends on the task / dataset used
    # if args.out_act is None:
    #         args.out_act = 'none'

    # set the dataset
    if 'goals' in args.dataset:
        args.dset_type = 'goals'
    elif 'copy' in args.dataset:
        args.dset_type = 'copy'
    else:
        args.dset_type = 'unknown'

    # use custom goals loss for goals dataset, override default loss fn
    if args.dset_type == 'goals':
        args.loss = 'goals'

    args.argv = sys.argv

    # setting seeds
    if args.seed is None:
        args.seed = random.randrange(1e6)
    if args.network_seed is None:
        args.network_seed = random.randrange(1e6)
    if args.res_seed is None:
        args.res_seed = random.randrange(1e6)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # initializing logging
    # do this last, because we will be logging previous parameters into the config file
    if not args.no_log:
        if args.slurm_id is not None:
            log = log_this(args, 'logs', os.path.join(args.name.split('_')[0], args.name.split('_')[1]), checkpoints=args.log_checkpoint_models)
        else:
            log = log_this(args, 'logs', args.name, checkpoints=args.log_checkpoint_models)

        logging.basicConfig(format='%(message)s', filename=log.run_log, level=logging.DEBUG)
        console = logging.StreamHandler()
        console.setLevel(logging.DEBUG)
        logging.getLogger('').addHandler(console)
        args.log = log
    else:
        logging.basicConfig(format='%(message)s', level=logging.DEBUG)
        logging.info('NOT LOGGING THIS RUN.')

    # logging, when loading models from paths
    if args.model_path is not None:
        logging.info(f'Using model path {args.model_path}')

    logging.info(f'Seeds:\n  general: {args.seed}\n  network: {args.network_seed}')

    return args


if __name__ == '__main__':
    args = parse_args()
    args = adjust_args(args)
    
    trainer = Trainer(args)
    logging.info(f'Initialized trainer. Using optimizer {args.optimizer}.')
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
                args.network_seed, args.res_x_seed, args.res_noise,
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



