import torch
import torch.nn as nn

"""
Args:
"""
class FullyConnected(nn.Module):
    def __init__(self, in_features, out_features, bias=True, norm=None, act=None):
        super(FullyConnected, self).__init__()
        
        # Convolution
        self.fc = nn.Linear(in_features=in_features, out_features=out_features, bias=True)

        # Normalization
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(out_features)
        elif norm == 'in':
            self.norm = nn.InstanceNorm2d(out_features)
        elif norm is None:
            self.norm = None
        
        # Activation
        if act == 'relu':
            self.act = nn.ReLU()
        elif act == 'lrelu':
            self.act = nn.LeakyReLU(negative_slope=0.2)
        elif act == 'tanh':
            self.act = nn.Tanh()
        elif act == 'sigmoid':
            self.act = nn.Sigmoid()
        elif act is None:
            self.act = None

    def forward(self, x):

        y = self.fc(x)

        if self.norm is not None:
            y = self.norm(y)
        
        if self.act is not None:
            y = self.act(y)

        return y