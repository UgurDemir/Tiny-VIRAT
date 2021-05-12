import torch
import torch.nn as nn

from collections import OrderedDict

import deepnn.nets.builder as lbuild

class C3DGAN(nn.Module):
    def __init__(self, ngf=64, norm='bn', use_dropout=False, 
                inch=3, outch=3, intemp=1,
                padding='reflect', uplayer='iconv', actc='ReLU', actd='LeakyReLU', acto='Tanh'):
        super(C3DGAN, self).__init__()

        cact = lbuild.act(actc)
        dact = lbuild.act(actd)
        oact = lbuild.act(acto)

        seq = [
            ('conv1', nn.Conv3d(3, 32, kernel_size=(3,3,3), padding=(1,1,1))),
            ('conv1_act', cact()),
            ('pool1', nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))),

            ('conv2', nn.Conv3d(32, 32, kernel_size=(3,3,3), padding=(1,1,1))),
            ('conv2_act', cact()),
            ('pool2', nn.MaxPool3d(kernel_size=(2, 1, 1), stride=(2, 1, 1))),

            ('conv3a',nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))),
            ('conv3a_act', cact()),
            ('conv3b',nn.Conv3d(64, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))),
            ('conv3b_act', cact()),
            ('pool3', nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))),

            ('conv4a',nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))),
            ('conv4a_act', cact()),
            ('conv4b',nn.Conv3d(128, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))),
            ('conv4b_act', cact()),
            ('pool4', nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))),

            ('up1', nn.ConvTranspose3d(128, 64, kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1), output_padding=(0,1,1))),
            ('up1_act', dact()),
            ('up2', nn.ConvTranspose3d(64, 32, kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1), output_padding=(0,1,1))),
            ('up2_act', dact()),
            ('up3', nn.ConvTranspose3d(32, 16, kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1), output_padding=(0,1,1))),
            ('up3_act', dact()),

            ('out', nn.Conv3d(16, 3, kernel_size=(2,3,3), padding=(0,1,1))),
            ('out_act', oact())

        ]

        self.model = nn.Sequential(OrderedDict(seq))

    
    # Args:
    #   x: input video segment with shape [batch, channel, temporal, height, width]
    # Output:
    #   y: Super-resolved image with shape [batch, channel, height, width]    
    def forward(self, x):
        y = self.model(x)
        return y.squeeze(dim=2)