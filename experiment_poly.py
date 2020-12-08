import numpy as np
import torch
from torch.optim import SGD
from torch.nn import Sequential, Conv1d, Module, Linear, ReLU, MSELoss
from dilated_CNN import DilatedConvNet
from functions import generate_poly_increasing
import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'




   
def score(model, X_test, y_test):
    y_pred = model(X_test)
    return MSELoss()(y_pred[:, 0, 0], y_test).item()
    
def generate_experiment_data(generate_function, n, dim, p, test_size=0.5):
    X, y = generate_function(n, dim, p)
    X_test, y_test = generate_function(int(test_size * n), dim, p)
    X, y = torch.Tensor(X).cuda(), torch.Tensor(y).cuda()
    X_test, y_test = torch.Tensor(X_test).cuda(), torch.Tensor(y_test).cuda()
    return X, y, X_test, y_test

n = 100
dim = 10000
p = 2
X, y, X_test, y_test = generate_experiment_data(generate_poly_increasing, n, dim, p)

num_layers = 4
out_channels = 4
kernel_size = 4
dilation = 1
model = DilatedConvNet(num_layers, out_channels, kernel_size, dilation).cuda()
loss_function = MSELoss()
optimizer = SGD(model.parameters(), 0.01, momentum=0.5)

for epoch in range(1000):
    optimizer.zero_grad()
    y_pred = model(X)
    loss = loss_function(y_pred[:, 0, 0], y)

    loss.backward()
    optimizer.step()

    print(loss.item())


print(score(model, X_test, y_pred))