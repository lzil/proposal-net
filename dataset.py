import numpy as np
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor as gpr
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel
import pickle
import os
import sys
import json
import pdb
import random

import matplotlib.pyplot as plt

import argparse

from motifs import gen_fn_motifs
from utils import load_rb

eps = 1e-6

# toy ready set go dataset
def create_dataset(args):
    name = args.name
    t_type = args.trial_type
    n_trials = args.n_trials
    t_len = args.trial_len
    trial_args = args.trial_args

    config = {}
    config['argv'] = sys.argv
    config['t_type'] = t_type
    config['n_trials'] = n_trials
    config['t_len'] = t_len

    trials = []

    if t_type.startswith('copy'):
        delay = get_args_val(trial_args, 'delay', 0, int)
        config['delay'] = delay
        
        dim = 1
        x = np.arange(0, t_len)
        ys = []

        if t_type == 'copy':
            n_freqs = get_args_val(trial_args, 'n_freqs', 15, int)
            f_range = get_args_val(trial_args, 'f_range', [2, 30], float, n_vals=2)
            amp = get_args_val(trial_args, 'amp', 1, float)
            start_zero = 'start_nonzero' not in trial_args
            config['n_freqs'] = n_freqs
            config['f_range'] = f_range
            config['amp'] = amp
            config['start_zero'] = start_zero

            fn = np.sin if start_zero else np.cos

            for n in range(n_trials):
                y = np.zeros_like(x)
                freqs = np.random.uniform(f_range[0], f_range[1], (n_freqs))
                amps = np.random.uniform(-amp, amp, (n_freqs))
                for i in range(n_freqs):
                    y = y + amps[i] * fn(1/freqs[i] * x)

                ys.append(y)

        elif t_type == 'copy_gp':
            interval = get_args_val(trial_args, 'interval', 10, int)
            scale = get_args_val(trial_args, 'scale', 1, float)
            kernel = RBF(length_scale=5)
            config['interval'] = interval
            config['scale'] = scale

            x_list = x[..., np.newaxis]
            x_filter = x_list[::interval]
            n_pts = x_filter.squeeze().shape[0]

            for n in range(n_trials):
                y_filter = np.zeros((n_pts, dim))
                y_filter[0] = 0
                for i in range(1, n_pts):
                    if 'smoothing' in trial_args:
                        y_filter[i] = np.random.multivariate_normal(y_filter[i-1]/2, cov=scale*np.eye(dim))
                    else:
                        y_filter[i] = np.random.multivariate_normal(np.zeros(dim), cov=scale*np.eye(dim))

                gp = gpr(kernel=kernel, normalize_y=False).fit(x_filter, y_filter)
                y_prediction, y_std = gp.predict(x_list, return_std=True)

                y = y_prediction.reshape(-1)
                ys.append(y)

        elif t_type == 'copy_motifs':
            assert args.motifs is not None
            motifs = load_rb(args.motifs)
            pause = get_args_val(trial_args, 'pause', 10, int)
            amp = get_args_val(trial_args, 'amp', .1, float)
            config['pause'] = pause
            config['amp'] = amp

            for n in range(n_trials):
                y = gen_fn_motifs(motifs, length=t_len, pause=pause, amp=amp, smoothing='cubic')
                ys.append(y)
        else:
            raise Exception

        for y in ys:
            z = np.zeros_like(y)
            if delay == 0:
                z = y
            else:
                z[delay:] = y[:-delay]

            trials.append((y, z, delay))

    elif t_type == 'integration':
        n_freqs = get_args_val(trial_args, 'n_freqs', 15, int)
        f_range = get_args_val(trial_args, 'f_range', [3, 30], float, n_vals=2)
        amp = get_args_val(trial_args, 'amp', 1, int)
        config['n_freqs'] = n_freqs
        config['f_range'] = f_range
        config['amp'] = amp

        for n in range(n_trials):
            x = np.arange(0, t_len)
            y = np.zeros_like(x).astype(np.float32)

            xp = t_len//2

            freqs = np.random.uniform(f_range[0], f_range[1], (n_freqs))
            amps = np.random.uniform(-amp, amp, (n_freqs))
            for i in range(n_freqs):
                y[:xp] = y[:xp] + amps[i] * np.cos(1/freqs[i] * x[:xp])

            y[:xp] = y[:xp] * np.cos(np.pi/2 * x[:xp] / x[xp])

            z_mag = np.sum(y)
            trial_range = np.arange(t_len)
            z = z_mag / 2 * norm.pdf(trial_range, loc=int(xp * 3/2), scale=2)

            trials.append((y, z, z_mag))

    elif t_type == 'goals':
        n_goals = get_args_val(trial_args, 'n_goals', 10, int)
        scale = get_args_val(trial_args, 'scale', 5, float)
        dim = get_args_val(trial_args, 'dim', 2, int)
        config['n_goals'] = n_goals
        config['scale'] = scale
        config['dim'] = dim

        for n in range(n_trials):
            trial = []
            for i in range(n_goals):
                trial.append(np.random.normal(loc=0, scale=scale, size=dim))

            trials.append(trial)

    return trials, config

def get_args_val(args, name, default, dtype, n_vals=1):
    if name in args:
        idx = args.index(name)
        if n_vals == 1:
            val = dtype(args[idx + 1])
        else:
            val = []
            for i in range(1, n_vals+1):
                val.append(dtype(args[idx + i]))
    else:
        val = default
    return val

def save_dataset(dset, name, config=None):
    fname = name + '.pkl'
    with open(os.path.join('datasets', fname), 'wb') as f:
        pickle.dump(dset, f)
    gname = name + '.json'
    if config is not None:
        with open(os.path.join('datasets', 'configs', gname), 'w') as f:
            json.dump(config, f, indent=2)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('mode', default='load', choices=['load', 'create'], help='do you want to create or load a dataset?')
    parser.add_argument('name', help='name of the dataset to either create or load')
    parser.add_argument('-t', '--trial_type', default='goals', help='the type of task')
    parser.add_argument('-a', '--trial_args', nargs='*', help='terms to specify parameters of trial type')
    parser.add_argument('-n', '--n_trials', type=int, default=1000, help='number of trials in this dataset')
    parser.add_argument('-l', '--trial_len', type=int, default=200, help='length of trial for copy task')
    parser.add_argument('--motifs', type=str, help='path to motifs for the copy task')
    args = parser.parse_args()

    if args.trial_args is None:
        args.trial_args = []

    if args.mode == 'create':
        dset, config = create_dataset(args)
        save_dataset(dset, args.name, config=config)
    elif args.mode == 'load':
        dset = load_rb(args.name)

        dset_len = len(dset)
        sample = random.sample(dset, 6)
        dset_range = range(len(sample[0][0]))
        fig, ax = plt.subplots(2,3,sharex=True, sharey=True, figsize=(8,4))
        for i, ax in enumerate(fig.axes):
            ax.plot(dset_range, sample[i][0], color='coral', label='ready/set', lw=2)
            ax.plot(dset_range, sample[i][1], color='dodgerblue', label='go', lw=2)
            ax.axvline(x=0, color='dimgray', alpha = 1)
            ax.axhline(y=0, color='dimgray', alpha = 1)
            ax.grid(True, which='major', lw=1, color='lightgray', alpha=0.4)
            ax.tick_params(axis='both', color='white')
            #ax.set_title(sample[i][2])
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['bottom'].set_visible(False)

        handles, labels = ax.get_legend_handles_labels()
        #fig.legend(handles, labels, loc='lower center')
        plt.show()
    # confirm ready set go works
    # for i in range(5):
    #     np.random.shuffle(dset)
    #     print(np.where(dset[i][0] == 1), np.argmax(dset[i][1]))
