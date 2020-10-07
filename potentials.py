import torch
import numpy as np
from scipy.stats import multivariate_normal as mvn
from torch.distributions import multivariate_normal as tmvn

import pdb

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
    z = .01 * (x[0] ** 2 + x[1] ** 2)
    if npy:
        z = np.clip(z, a_min=0, a_max=None)
    else:
        z = torch.clamp(z, min=0)
    return z

def central_bump(x, npy=False):
    if npy:
        xx = np.array(x).transpose(1,2,0)
        z = mvn(np.zeros(len(x))).pdf(xx)
    else:
        dist = tmvn.MultivariateNormal(loc=torch.zeros_like(x), covariance_matrix=torch.eye(x.shape[0]))
        z = torch.exp(dist.log_prob(x))
    return 5 * z