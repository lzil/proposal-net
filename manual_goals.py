import numpy as np

dim = 2

n_goals = 5


def across_goals(n_goals, dim, scale):
    rng = np.random.default_rng()
    mains = rng.multivariate_normal(np.zeros(dim), (scale ** 2) * np.eye(dim), size=n_goals)

    goals = []
    for i in range(n_goals):
        goals.append(mains[i])
        goals.append(-mains[i])

    return goals