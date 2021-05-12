""" Taken from https://pytorch.org/tutorials/intermediate/spatial_transformer_tutorial.html and modified """

import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatialTransformer3d_NOTCOMPLETED(nn.Module):
    def __init__(self):
        super(SpatialTransformer3d_NOTCOMPLETED, self).__init__()

        self.localization = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 3 * 3, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )

        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def forward(self, x):
        xs = self.localization(x)
        print('xs', xs.size())
        xs = xs.view(-1, 10 * 3 * 3)
        print('xs', xs.size())
        theta = self.fc_loc(xs)
        print('theta', theta.size())
        theta = theta.view(-1, 2, 3)
        print('theta', theta.size())

        grid = F.affine_grid(theta, x.size())
        print('grid', grid.size())
        x = F.grid_sample(x, grid)
        return x


