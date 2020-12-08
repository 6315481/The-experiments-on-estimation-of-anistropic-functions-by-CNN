import numpy as np
import torch
from torch import transpose
from torch.nn import Sequential, Conv1d, Module, Linear, ReLU, MSELoss


class DilatedConvNet(Module):

    def __init__(self, num_layers, out_channels, kernel_size, dilation=1):
        super(DilatedConvNet,self).__init__()

        layers = [Conv1d(in_channels=1, kernel_size=kernel_size, out_channels=out_channels, dilation=1)]
        for i in range(num_layers-1):
            layers.append(Conv1d(in_channels=out_channels, kernel_size=kernel_size, out_channels=out_channels, dilation=dilation ** (i+1)))
        self.CNN_layers = Sequential(*layers)

        self.FNN_layers = Sequential(Linear(out_channels, 200), ReLU(), Linear(200, 1))

    def forward(self, x):
        x = self.CNN_layers(x)
        x = transpose(x, 1, 2)
        return self.FNN_layers(x)
    
    

