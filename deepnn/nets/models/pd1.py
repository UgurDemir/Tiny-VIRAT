import torch
import torch.nn as nn

from collections import OrderedDict

from deepnn.layer.conv3d import ConvST3d
from deepnn.layer.fc import FullyConnected

class PD1(nn.Module):
    def __init__(self, ndf=16, actc='lrelu', acto='sigmoid', norm='spec', conv_mode='3D'):
        super(PD1, self).__init__()

        feat = [
            ConvST3d(3,   ndf,  (3,3,3), padding=(1,1,1), stride=(1,1,1), act=actc, norm=norm, conv_mode=conv_mode),
            ConvST3d(3,   ndf,  (3,3,3), padding=(1,1,1), stride=(1,1,1), act=actc, norm=norm, conv_mode=conv_mode),
            ConvST3d(3,   ndf,  (3,3,3), padding=(1,1,1), stride=(1,1,1), act=actc, norm=norm, conv_mode=conv_mode)
        ]

        blocks = [
            Block(ndf, ndf, 2, stride=[1,2,2], norm=norm, act=actc, conv_mode=conv_mode),
            Block(ndf, ndf, 1, stride=[1,2,2], norm=norm, act=actc, conv_mode=conv_mode),
            Block(ndf, ndf, 1, stride=[1,2,2], norm=norm, act=actc, conv_mode=conv_mode)
        ]

        self.convout = ConvST3d(ndf, 1, (3,3,3), padding=(1,1,1), stride=(2,1,1), act=acto, norm=None, conv_mode=conv_mode)

        self.avgpool = nn.AvgPool3d(kernel_size=[1,2,2], stride=[1,2,2])
        self.feat_proj = nn.ModuleList(feat)
        self.layers = nn.ModuleList(blocks)

    def forward(self, x, scale=None, fadein=0.0):
        scale = scale if scale is not None else len(self.layers)
        
        # Initial conv block to increase number of channel
        y = self.feat_proj[scale-1](x)

        # Encoding and upscaling. Core Block
        for i in range(scale-1,-1,-1):
            y = self.layers[i](y)
            if fadein > 0.0 and i == (scale - 1):
                assert scale > 1
                y_prev = self.feat_proj[scale-2](x)
                y_prev = self.avgpool(y_prev)
                y = y_prev * fadein + y * (1.0 - fadein)

        y = self.convout(y)

        return y

class Block(nn.Module):
    def __init__(self, inch, outch, nlayer, stride=[1,1,1], norm='spec', act='lrelu', conv_mode='3D'):
        super(Block, self).__init__()
        seq = []
        for _ in range(nlayer):
            seq += [ConvST3d(inch,  outch, (3,3,3), padding=(1,1,1), stride=stride, act=act, norm=norm, conv_mode=conv_mode)]

        self.model = nn.Sequential(*seq)

    def forward(self, x):
        y = self.model(x)
        return y