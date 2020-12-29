import numpy as np
import torch
import pickle
from torch.optim import SGD
from torch.nn import Sequential, Conv1d, Module, Linear, ReLU, MSELoss
from dilated_CNN import DilatedConvNet
from functions import anistrophic_decrease, mix_decrease
from experiment_module import exec_experiment
import os


n = 128
num_layers = 1
out_channels = 10
kernel_size = 10
dilation = 1
test_losses = {}
dim_list = [100, 200, 500, 1000]
p_list = [1.5, 2.0, 2.5, 5.0]

n_experiment = 5

for p in p_list:
    test_losses[p] = [] 
    for dim in dim_list:
        m = []
        for i in range(n_experiment):
            sigma = np.arange(1, dim+1) ** p
            m.append(exec_experiment(n, dim, sigma, num_epoch=1000))
        test_losses[p].append(np.mean(m))

print(test_losses)
print('sum_inv: {}'.format(np.sum(1 / sigma)))

with open('anistrophic_sparse.pickle', 'wb') as f:
    pickle.dump(test_losses, f)

