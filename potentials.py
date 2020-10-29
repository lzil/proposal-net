import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

from scipy.stats import multivariate_normal as MVN
from torch.distributions.multivariate_normal import MultivariateNormal as TMVN

import pdb

def p_none(x, npy=False):
    if npy:
        z = np.zeros(x.shape[:-1])
    else:
        z = torch.zeros(x.shape[:-1])
    return z

def p_sin(x, npy=False):
    if npy:
        z = np.sin(np.linalg.norm(x, axis=-1))
        z = np.clip(z, a_min=0, a_max=None)
    else:
        z = torch.sin(torch.linalg.norm(x, dim=-1))
        z = torch.clamp(z, min=0)
    return 10*z

def p_slope(x, npy=False):
    if npy:
        z = np.linalg.norm(x, axis=-1)
    else:
        z = torch.linalg.norm(x, dim=-1)
    return .3 * z

def p_bump(x, npy=False):
    ndims = x.shape[-1]
    if npy:
        z = MVN(np.zeros(ndims)).pdf(x)
    else:
        dist = TMVN(torch.zeros(ndims), torch.eye(ndims))
        z = torch.exp(dist.log_prob(x.float()))
    return 100 * z

if __name__ == '__main__':

    lim = 10
    

    x = np.outer(np.linspace(-lim, lim, 1000), np.ones(1000))
    y = x.copy().T

    ins = np.stack([x, y]).transpose(1, 2, 0)
    if not npy:
        x = torch.tensor(x)
        y = torch.tensor(y)
        ins = torch.tensor(ins)

    p = 'sin'
    npy = True

    if p == 'slope':
        z = p_slope(ins, npy)
    elif p == 'bump':
        z = p_bump(ins, npy)
    elif p == 'sin':
        z = p_sin(ins, npy)
    else:
        z = p_none(ins, npy)

    if not npy:
        z = z.detach().numpy()

    # Creating figyre
    fig = plt.figure(figsize =(14, 9))
    ax = plt.axes(projection ='3d')

    # Creating plot
    ax.plot_surface(x, y, z, cmap='plasma')
    ax.set_zlim3d(-lim, lim)

    # Hide grid lines
    ax.grid(False)

    plt.axis('off')
    # z100 = np.zeros(100)
    # a100 = np.linspace(-5, 5, 100)
    # ax.plot3D(z100, z100, a100, color='black')
    # ax.plot3D(z100, a100, z100, color='black')

    # ax.plot3D(a100, z100, z100, color='black')

    # show plot
    plt.show()
