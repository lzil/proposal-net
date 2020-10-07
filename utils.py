import os
import numpy as np

#import tensorflow as tf

import yaml
import logging
import time
import json
import csv
import pickle
# import pandas as pd
import pdb
import copy


# tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class LogObject(object):
    pass

# use yaml config files; note what is actually set via the config file
def add_yaml_args(args, config_file):
    if config_file:
        config = yaml.safe_load(open(config_file))
        dic = vars(args)
        # all(map(dic.pop, config))
        for c, v in config.items():
            dic[c] = v
            # if c in dic.keys():
            #     logging.info(f'{c} is set via config: {v}')
            # else:
            #     logging.warning(f'{c} is not set to begin with: {v}')
    return args

# fills an argument dictionary with keys from a default dictionary
# also works with dicts now
def fill_undefined_args(args, default_args, overwrite_none=False, to_bunch=False):
    # so we don't overwrite the original args
    args = copy.deepcopy(args)
    # takes care of default args not being a dict
    if type(default_args) is not dict:
        default_args = default_args.__dict__
    # only change the args dictionary
    if type(args) is dict:
        args_dict = args
    else:
        args_dict = args.__dict__
    for k in default_args.keys():
        if k not in args_dict:
            args_dict[k] = default_args[k]
        elif overwrite_none and args_dict[k] is None:
            args_dict[k] = default_args[k]

    if to_bunch:
        args = Bunch(**args_dict)
    return args


# produce run id and create log directory
# config is the configuration file for the run
# log_dir is usually 'logs'
# log_name is the parent name of the run/type of run
# checkpoints is whether we want to save checkpoints
# use_id is deprecated
def log_this(config, log_dir, log_name=None, checkpoints=False, use_id=True):
    run_id = str(int(time.time() * 100))[-7:]
    config.run_id = run_id
    print('\n=== Logging ===', flush=True)
    
    if log_name is None or len(log_name) == 0:
        log_name = run_id
    print(f'Run id: {run_id} with name {log_name}', flush=True)

    run_dir = os.path.join(log_dir, log_name)
    os.makedirs(run_dir, exist_ok=True)
    print(f'Log folder: {run_dir}', flush=True)

    log_path = os.path.join(run_dir, f'log_{run_id}.log')
    print(f'Log file: {log_path}', flush=True)

    if checkpoints:
        checkpoint_dir = os.path.join(run_dir, f'checkpoints_{run_id}')
        os.makedirs(checkpoint_dir, exist_ok=True)
        print(f'Logging checkpoints to: {checkpoint_dir}', flush=True)
    else:
        checkpoint_dir = None

    # might want to send stdout here later too
    path_config = os.path.join(run_dir, f'config_{run_id}.json')
    with open(path_config, 'w', encoding='utf-8') as f:
        json.dump(vars(config), f, indent=4)
        print(f'Config file saved to: {path_config}', flush=True)

    log = LogObject()
    log.checkpoint_dir = checkpoint_dir
    log.run_dir = run_dir
    log.run_log = log_path
    log.run_id = run_id

    print('===============\n', flush=True)
    return log


# https://code.activestate.com/recipes/52308-the-simple-but-handy-collector-of-a-bunch-of-named/?in=user-97991
class Bunch:
    def __init__(self, **kwds):
        self.__dict__.update(kwds)

def load_rb(path):
    with open(path, 'rb') as f:
        qs = pickle.load(f)
    return qs

def lrange(l, p=0.1):
    return np.linspace(0, (l-1) * p, l)

# get config dictionary from the model path
def get_config(model_path):
    head, tail = os.path.split(model_path)
    run_id = re.split('_|\.', tail)[1]
    config_path = os.path.join(head, 'config_'+run_id+'.json')
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

