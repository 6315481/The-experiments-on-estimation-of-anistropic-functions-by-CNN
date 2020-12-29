import numpy as np


def anistrophic_decrease(X, sigma):
    num = X.shape[0]
    dim = X.shape[1]

    y = np.zeros(num)

    normalizer = 0
    phi = np.sqrt(2) * np.cos(2 * np.pi * X)
    for i in range(dim):
        normalizer = normalizer + 2 ** sigma[i] 
        y = y + np.prod(phi[:, 0:(i+1)], axis=1) / normalizer

    return y    


def mix_decrease(X, sigma):
    num = X.shape[0]
    dim = X.shape[1]

    y = np.zeros(num)

    normalizer = 1
    phi = np.sqrt(2) * np.cos(2 * np.pi * X)
    for i in range(dim):
        normalizer = normalizer * 2 ** sigma[i] 
        y = y + np.prod(phi[:, 0:(i+1)], axis=1) / normalizer

    return y   

