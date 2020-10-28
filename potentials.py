import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

from scipy.stats import multivariate_normal as MVN
from torch.distributions.multivariate_normal import MultivariateNormal as TMVN

import pdb

def none(x, npy=False):
    return 0 * x[0] * x[1]

def sin_sphere(x, npy=False):
    if npy:
        z = np.sin(np.sqrt(x[0] ** 2 + x[1] ** 2))
        z = np.clip(z, a_min=0, a_max=None)
    else:
        z = torch.sin(torch.sqrt(x[0] ** 2 + x[1] ** 2))
        z = torch.clamp(z, min=0)
    return z

def gentle_slope(x, npy=False):
    z = .02 * (x[0] ** 2 + x[1] ** 2)
    return z

def central_bump(x, npy=False):
    if npy:
        x_adj = x.transpose(1, 2, 0)
        z = MVN(np.zeros(len(x))).pdf(x_adj)
    else:
        dist = TMVN(torch.zeros(len(x)), torch.eye(x.shape[0]))
        x_adj = x.permute(1, 2, 0).float()
        z = torch.exp(dist.log_prob(x_adj))
    return 10 * z



if __name__ == '__main__':

    lim = 10
    npy = True

    x = np.outer(np.linspace(-lim, lim, 1000), np.ones(1000))
    y = x.copy().T

    ins = np.stack([x, y])
    if not npy:
        x = torch.tensor(x)
        y = torch.tensor(y)
        ins = torch.tensor(ins)

    p = 'central_bump'

    if p == 'gentle_slope':
        z = gentle_slope(ins, npy)
    elif p == 'central_bump':
        z = central_bump(ins, npy)
    elif p == 'sin_sphere':
        z = sin_sphere(ins, npy)

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
