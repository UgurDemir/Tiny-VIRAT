import torch
import torch.nn as nn

from collections import OrderedDict

from deepnn.layer.conv3d import ConvST3d

"""
Standard residual decoder
Args:
    cnorm: normalization layer after convolution
    unorm: normalization layer after upconv layer
    actc: activation function for encoding step
    actd: activation function for decoding step
    scse: Skip connection from start to end [True|False]
"""
class G3(nn.Module):
    def __init__(self, ngf=128, actd='relu', acto='tanh', cnorm='bn', unorm='bn'):
        super(G3, self).__init__()

        self.conv1 = ConvST3d(3, ngf, (3,3,3), padding=(1,1,1), act=actd, norm=cnorm)
        self.uconv1 = ConvST3d(ngf, ngf, (3,3,3), padding=(1,1,1), scale_factor=[1,2,2], scale_mode='trilinear', act=actd, norm=unorm)
        self.conv2 = ConvST3d(ngf, ngf, (3,3,3), padding=(1,1,1), act=actd, norm=cnorm)
        self.uconv2 = ConvST3d(ngf, ngf, (3,3,3), padding=(1,1,1), scale_factor=[1,2,2], scale_mode='trilinear', act=actd, norm=unorm)
        self.conv3 = ConvST3d(ngf, ngf, (3,3,3), padding=(1,1,1), act=actd, norm=cnorm)
        self.conv4 = ConvST3d(ngf, 3, (3,3,3), padding=(1,1,1), act=acto)

    def forward(self, x):
        c1 = self.conv1(x)
        u1 = self.uconv1(c1)

        c2 = self.conv2(u1)
        u2 = self.uconv2(u1+c2)

        c3 = self.conv3(u2)
        c4 = self.conv4(c3+u2)

        return c4


