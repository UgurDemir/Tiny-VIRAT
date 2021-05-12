import torch
import torch.nn as nn

from collections import OrderedDict

import deepnn.nets.builder as lbuild

class C3D_DISC(nn.Module):
    def __init__(self, actc='ReLU', acto='Softmax'):
        super(C3D_DISC, self).__init__()

        cact = lbuild.act(actc)
        oact = lbuild.act(acto)

        feat = [
            ('conv1', nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))),
            ('conv1_act', cact()),
            ('pool1', nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))),

            ('conv2', nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))),
            ('conv2_act', cact()),
            ('pool2', nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))),

            ('conv3a',nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))),
            ('conv3a_act', cact()),
            ('conv3b',nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))),
            ('conv3b_act', cact()),
            ('pool3', nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))),

            ('conv4a',nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))),
            ('conv4a_act', cact()),
            ('conv4b',nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))),
            ('conv4b_act', cact()),
            ('pool4', nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))),

            ('conv5a',nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))),
            ('conv5a_act', cact()),
            ('conv5b',nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))),
            ('conv5b_act', cact()),
            ('pool5', nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 1, 1)))

        ]

        linear = [
            ('fc1',nn.Linear(8192, 4096)),
            ('fc1_act', cact()),
            ('do1', nn.Dropout(p=0.5)),
            ('fc2',nn.Linear(4096, 4096)),
            ('fc2_act', cact()),
            ('do2', nn.Dropout(p=0.5)),
            ('fc3',nn.Linear(4096, 487)),
            ('fc3_act', oact())
        ]

        self.feature = nn.Sequential(OrderedDict(feat))
        self.fc = nn.Sequential(OrderedDict(linear))

      
    def forward(self, x):
        y = self.feature(x)
        y = y.view(-1, 8192)
        y = self.fc(y)
        return y