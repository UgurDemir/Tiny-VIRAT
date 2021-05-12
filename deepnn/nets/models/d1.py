import torch
import torch.nn as nn

from collections import OrderedDict

from deepnn.layer.conv3d import ConvST3d

class D1(nn.Module):
    def __init__(self, ndf=16, actc='relu', acto='sigmoid', norm='bn'):
        super(D1, self).__init__()

        seq = [
            ('conv1', ConvST3d(3, ndf, (3,3,3), padding=(1,1,1), stride=(1,2,2), act=actc, norm=norm)),
            ('conv2', ConvST3d(ndf, ndf*2, (3,3,3), padding=(1,1,1), stride=(2,2,2), act=actc, norm=norm)),
            ('conv3', ConvST3d(ndf*2, ndf*4, (3,3,3), padding=(1,1,1), stride=(2,2,2), act=actc, norm=norm)),
            ('conv4', ConvST3d(ndf*4, ndf*8, (3,3,3), padding=(1,1,1), stride=(2,2,2), act=actc, norm=norm)),
            ('conv5', ConvST3d(ndf*8, ndf*8, (3,3,3), padding=(1,1,1), stride=(2,2,2), act=actc, norm=norm)),

            ('conv6', ConvST3d(ndf*8, 1, (1,4,4), stride=(1,4,4), act=acto)),
        ]

        self.model = nn.Sequential(OrderedDict(seq))

    def forward(self, x):
        b = x.size(0)
        y = self.model(x)
        y = y.view(b, 1)
        return y
