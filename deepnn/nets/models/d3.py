import torch
import torch.nn as nn

from collections import OrderedDict

from deepnn.layer.conv3d import ConvST3d
from deepnn.layer.fc import FullyConnected

class D3(nn.Module):
    def __init__(self, ndf=16, nfc=256, nframe=16, nout=16, actc='lrelu', acto='sigmoid', norm='bn', conv_mode='ST'):
        super(D3, self).__init__()
        self.nout = nout

        seq = [
            ('conv1', ConvST3d(3, ndf, (3,3,3), padding=(1,1,1), stride=(1,2,2), act=actc, norm=norm, conv_mode=conv_mode)),
            ('conv2', ConvST3d(ndf, ndf, (3,3,3), padding=(1,1,1), stride=(1,2,2), act=actc, norm=norm, conv_mode=conv_mode)),
            ('conv3', ConvST3d(ndf, ndf*2, (3,3,3), padding=(1,1,1), stride=(1,2,2), act=actc, norm=norm, conv_mode=conv_mode)),
            ('conv4', ConvST3d(ndf*2, ndf*4, (3,3,3), padding=(1,1,1), stride=(1,2,2), act=actc, norm=norm, conv_mode=conv_mode)),
            ('conv5', ConvST3d(ndf*4, ndf*8, (3,3,3), padding=(1,1,1), stride=(1,2,2), act=actc, norm=norm, conv_mode=conv_mode))
        ]

        self.nfeat = 4*4 * nframe * ndf*8
        lin = [
            ('fc1', FullyConnected(self.nfeat, nfc, act='relu')),
            ('fc2', FullyConnected(nfc, self.nout, act=acto))
        ]

        self.model = nn.Sequential(OrderedDict(seq))
        self.fc = nn.Sequential(OrderedDict(lin))

    def forward(self, x):
        y = self.model(x)
        y = y.view(-1, self.nfeat)
        y = self.fc(y)
        return y