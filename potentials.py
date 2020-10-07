import torch
import numpy as np


def none(x, npy=False):
    return 0 * x[0] * x[1]

def sin_xy(x, npy=False):
    if npy:
        z = .1 * x[0] * x[1] * (np.sin(x[0] * x[1]))
        z = np.clip(z, a_min=0, a_max=None)
    else:
        z = .1 * x[0] * x[1] * (torch.sin(x[0] * x[1]))
        z = torch.clamp(z, min=0)
    return z

def sin_sphere(x, npy=False):
    if npy:
        z = np.sin(x[0] ** 2 + x[1] ** 2)
        z = np.clip(z, a_min=0, a_max=None)
    else:
        z = torch.sin(x[0] ** 2 + x[1] ** 2)
        z = torch.clamp(z, min=0)
    return z

def gentle_slope(x, npy=False):
    z = .1 * (x[0] ** 2 + x[1] ** 2)
    if npy:
        z = np.clip(z, a_min=0, a_max=None)
    else:
        z = torch.clamp(z, min=0)
    return z