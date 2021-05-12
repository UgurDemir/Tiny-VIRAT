import torch
import torch.nn as nn

from collections import OrderedDict

from deepnn.layer.conv3d import ConvST3d
from deepnn.layer.fc import FullyConnected

class D4(nn.Module):
    # [nlayer = 6] : 112x112--> 4x4
    # [nlayer = 5] : 112x112--> 7x7
    # [nlayer = 4] : 112x112--> 14x14 
    def __init__(self, ndf=16, nlayer=5, actc='lrelu', acto='sigmoid', norm='spec', conv_mode='3D'):
        super(D4, self).__init__()

        seq = [
            ('conv1', ConvST3d(3, ndf, (3,3,3), padding=(1,1,1), stride=(1,1,1), act=actc, norm=norm, conv_mode=conv_mode)),
        ]

        nin = ndf
        nout = ndf
        for i in range(nlayer-2):
            nout = nin * (2**i)
            seq += [('conv%d'%(i+2), ConvST3d(nin, nout, (3,3,3), padding=(1,1,1), stride=(1,2,2), act=actc, norm=norm, conv_mode=conv_mode))]
            nin = nout

        seq += [('convout', ConvST3d(nout, 1, (3,3,3), padding=(1,1,1), stride=(1,2,2), act=acto, norm=None, conv_mode=conv_mode))]

        self.model = nn.Sequential(OrderedDict(seq))

    def forward(self, x):
        y = self.model(x)
        return y