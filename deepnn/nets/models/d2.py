import torch
import torch.nn as nn

from collections import OrderedDict

from deepnn.layer.conv3d import ConvST3d

class D2(nn.Module):
    def __init__(self, ndf=16, actc='relu', acto='sigmoid', norm='bn', conv_mode='ST'):
        super(D2, self).__init__()

        seq = [
            ('conv1', ConvST3d(3, ndf, (3,3,3), padding=(1,1,1), stride=(1,2,2), act=actc, norm=norm, conv_mode=conv_mode)),
            ('conv2', ConvST3d(ndf, ndf*2, (3,3,3), padding=(1,1,1), stride=(1,2,2), act=actc, norm=norm, conv_mode=conv_mode)),
            ('conv3', ConvST3d(ndf*2, ndf*4, (3,3,3), padding=(1,1,1), stride=(1,2,2), act=actc, norm=norm, conv_mode=conv_mode)),
            ('conv4', ConvST3d(ndf*4, ndf*8, (3,3,3), padding=(1,1,1), stride=(1,2,2), act=actc, norm=norm, conv_mode=conv_mode)),
            ('conv5', ConvST3d(ndf*8, ndf*8, (3,3,3), padding=(1,1,1), stride=(1,2,2), act=actc, norm=norm, conv_mode=conv_mode)),

            ('conv6', ConvST3d(ndf*8, 1, (1,4,4), stride=(1,4,4), act=acto, conv_mode=conv_mode)),
        ]

        self.model = nn.Sequential(OrderedDict(seq))

    def forward(self, x):
        y = self.model(x)
        return y
