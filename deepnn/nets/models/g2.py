import torch
import torch.nn as nn

from collections import OrderedDict

from deepnn.layer.conv3d import ConvST3d

"""
Standard decoder
Args:
    cnorm: normalization layer after convolution
    unorm: normalization layer after upconv layer
    actc: activation function for encoding step
    actd: activation function for decoding step
"""
class G2(nn.Module):
    def __init__(self, ngf=128, actd='relu', acto='tanh', cnorm='bn', unorm='bn'):
        super(G2, self).__init__()

        seq = [
            ('conv1', ConvST3d(3, ngf, (3,3,3), padding=(1,1,1), act=actd, norm=cnorm)),
            ('uconv1', ConvST3d(ngf, ngf, (3,3,3), padding=(1,1,1), scale_factor=[1,2,2], scale_mode='trilinear', act=actd, norm=unorm)),
            ('conv2', ConvST3d(ngf, ngf, (3,3,3), padding=(1,1,1), act=actd, norm=cnorm)),
            ('uconv2', ConvST3d(ngf, ngf, (3,3,3), padding=(1,1,1), scale_factor=[1,2,2], scale_mode='trilinear', act=actd, norm=unorm)),
            ('conv3', ConvST3d(ngf, ngf, (3,3,3), padding=(1,1,1), act=actd, norm=cnorm)),
            ('conv4', ConvST3d(ngf, 3, (3,3,3), padding=(1,1,1), act=acto))
        ]

        self.model = nn.Sequential(OrderedDict(seq))

    def forward(self, x):
        return self.model(x)




