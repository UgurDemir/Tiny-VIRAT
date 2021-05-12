import torch
import torch.nn as nn

from collections import OrderedDict

from deepnn.layer.conv3d import ConvST3d
from deepnn.layer.fc import FullyConnected

class PD2(nn.Module):
    def __init__(self, mode, scale_factor, ndf=16, actc='lrelu', acto='sigmoid', norm='spec', conv_mode='3D'):
        super(PD2, self).__init__()
        assert mode in ['3D', 'ST']
        assert scale_factor in [4 , 8]

        if mode == '3D':
            feat = [
                ConvST3d(3,   ndf*4,  (3,3,3), padding=(1,1,1), stride=(1,1,1), act=actc, norm=norm, conv_mode=conv_mode),
                ConvST3d(3,   ndf*2,  (3,3,3), padding=(1,1,1), stride=(1,1,1), act=actc, norm=norm, conv_mode=conv_mode),
                ConvST3d(3,   ndf,  (3,3,3), padding=(1,1,1), stride=(1,1,1), act=actc, norm=norm, conv_mode=conv_mode)
            ]

            blocks = [
                nn.Sequential(
                    ConvST3d(ndf*4,  ndf*4, (3,3,3), padding=(1,1,1), stride=[1,2,2], act=actc, norm=norm, conv_mode=conv_mode),
                    ConvST3d(ndf*4,  ndf*4, (3,3,3), padding=(1,1,1), stride=[2,2,2], act=actc, norm=norm, conv_mode=conv_mode),
                    ConvST3d(ndf*4,  ndf*4, (3,3,3), padding=(1,1,1), stride=[1,2,2], act=actc, norm=norm, conv_mode=conv_mode),
                    ConvST3d(ndf*4,  ndf, (3,3,3), padding=(1,1,1), stride=[1,2,2], act=actc, norm=norm, conv_mode=conv_mode)
                ),
                nn.Sequential(
                    ConvST3d(ndf*2,  ndf*2, (3,3,3), padding=(1,1,1), stride=[1,1,1], act=actc, norm=norm, conv_mode=conv_mode),
                    ConvST3d(ndf*2,  ndf*4, (3,3,3), padding=(1,1,1), stride=[1,1,1], act=actc, norm=norm, conv_mode=conv_mode),
                ),
                ConvST3d(ndf,  ndf*2, (3,3,3), padding=(1,1,1), stride=[1,1,1], act=actc, norm=norm, conv_mode=conv_mode)
            ]
            if scale_factor == 4:
                feat, blocks = feat[:-1], blocks[:-1]

        elif mode == 'ST':
            feat = [
                ConvST3d(3,   ndf*4,  (3,3,3), padding=(1,1,1), stride=(1,1,1), act=actc, norm=norm, conv_mode=conv_mode),
                ConvST3d(3,   ndf*4,  (3,3,3), padding=(1,1,1), stride=(1,1,1), act=actc, norm=norm, conv_mode=conv_mode),
                ConvST3d(3,   ndf*2,  (3,3,3), padding=(1,1,1), stride=(1,1,1), act=actc, norm=norm, conv_mode=conv_mode),
                ConvST3d(3,   ndf*2,  (3,3,3), padding=(1,1,1), stride=(1,1,1), act=actc, norm=norm, conv_mode=conv_mode),
                ConvST3d(3,   ndf,  (3,3,3), padding=(1,1,1), stride=(1,1,1), act=actc, norm=norm, conv_mode=conv_mode),
                ConvST3d(3,   ndf,  (3,3,3), padding=(1,1,1), stride=(1,1,1), act=actc, norm=norm, conv_mode=conv_mode)
            ]

            blocks = [
                nn.Sequential(
                    ConvST3d(ndf*4,  ndf*4, (3,3,3), padding=(1,1,1), stride=[1,2,2], act=actc, norm=norm, conv_mode=conv_mode),
                    ConvST3d(ndf*4,  ndf*4, (3,3,3), padding=(1,1,1), stride=[2,2,2], act=actc, norm=norm, conv_mode=conv_mode),
                    ConvST3d(ndf*4,  ndf*4, (3,3,3), padding=(1,1,1), stride=[1,2,2], act=actc, norm=norm, conv_mode=conv_mode),
                    ConvST3d(ndf*4,  ndf, (3,3,3), padding=(1,1,1), stride=[1,2,2], act=actc, norm=norm, conv_mode=conv_mode)
                ),    
                ConvST3d(ndf*4,  ndf*4, (3,3,3), padding=(1,1,1), stride=[1,1,1], act=actc, norm=norm, conv_mode=conv_mode),
                ConvST3d(ndf*2,  ndf*4, (3,3,3), padding=(1,1,1), stride=[1,1,1], act=actc, norm=norm, conv_mode=conv_mode),
                ConvST3d(ndf*2,  ndf*2, (3,3,3), padding=(1,1,1), stride=[1,1,1], act=actc, norm=norm, conv_mode=conv_mode),
                ConvST3d(ndf,  ndf*2, (3,3,3), padding=(1,1,1), stride=[1,1,1], act=actc, norm=norm, conv_mode=conv_mode),
                ConvST3d(ndf,  ndf, (3,3,3), padding=(1,1,1), stride=[1,1,1], act=actc, norm=norm, conv_mode=conv_mode)
            ]
            if scale_factor == 4:
                feat, blocks = feat[:-2], blocks[:-2]

        
        self.convout = ConvST3d(ndf, 1, (3,3,3), padding=(1,1,1), stride=(1,1,1), act=acto, norm=None, conv_mode=conv_mode)

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
                nn.functional.interpolate(y_prev, size=[y.size(-3), y.size(-2), y.size(-1)], mode='trilinear', align_corners=False)
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