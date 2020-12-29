import numpy as np
import torch
from torch.optim import SGD
from torch.nn import Sequential, Conv1d, Module, Linear, ReLU, MSELoss
from dilated_CNN import DilatedConvNet
from functions import anistrophic_decrease, mix_decrease
import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'




def exec_experiment(n, dim, sigma, func=anistrophic_decrease, num_layers=1, out_channels=10, kernel_size=10, dilation=1, batch_size=32, num_epoch=200):
    np.random.seed(seed=42)
    X, X_test = np.random.uniform(size=[n, dim]), np.random.uniform(size=[n, dim])
    y, y_test = func(X, sigma), func(X_test, sigma)
    y, y_test = y / np.max(y), y_test / np.max(y_test)
    X, y = torch.Tensor(X.reshape((n, 1, dim))).cuda(), torch.Tensor(y).cuda()
    X_test, y_test = torch.Tensor(X_test.reshape((n, 1, dim))).cuda(), torch.Tensor(y_test).cuda()


    model = DilatedConvNet(num_layers, out_channels, kernel_size, dilation).cuda()
    loss_function = MSELoss()
    optimizer = SGD(model.parameters(), 0.01, momentum=0.5)

    num_batch = int(n / batch_size)

    for epoch in range(num_epoch):
        for i in range(num_batch):
            optimizer.zero_grad()

            left = (i-1) * batch_size
            right = i * batch_size
            y_pred = model(X[left:right])

            loss = loss_function(y_pred[:, 0, 0], y[left:right])
            loss.backward()
            optimizer.step()
        
        if  epoch % 5 == 0:
            print(loss.item())

    y_pred = model(X_test)
    loss = loss_function(y_pred[:, 0, 0], y_test)
    print('test loss : {}'.format(loss.item()))

    return loss.item()


