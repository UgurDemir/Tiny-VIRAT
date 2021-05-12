import torch
import torch.nn as nn

from collections import OrderedDict

from deepnn.layer.conv3d import ConvST3d

class G1(nn.Module):
    def __init__(self, ngf=16, actc='relu', actd='relu', acto='tanh'):
        super(G1, self).__init__()

        seq = [
            ('convin', ConvST3d(3, ngf, (5,5,5), padding=(2,2,2), act=actc)),

            ('d1_uconv1', ConvST3d(ngf, ngf, (3,3,3), padding=(1,1,1), scale_factor=[2,2,2], scale_mode='trilinear', act=actd)),
            ('d1_conv1', ConvST3d(ngf, ngf, (3,3,3), padding=(1,1,1), act=actd)),
            ('d1_uconv2', ConvST3d(ngf, ngf*2, (3,3,3), padding=(1,1,1), scale_factor=[2,2,2], scale_mode='trilinear', act=actd)),
            ('d1_conv2', ConvST3d(ngf*2, ngf*2, (3,3,3), padding=(1,1,1), act=actd)),

            ('e1_conv1', ConvST3d(ngf*2, ngf*2, (3,3,3), padding=(1,1,1), stride=(2,1,1), act=actc)),
            ('e1_conv2', ConvST3d(ngf*2, ngf, (3,3,3), padding=(1,1,1), stride=(2,2,2), act=actc)),
            ('e1_conv3', ConvST3d(ngf, int(ngf/2), (3,3,3), padding=(1,1,1), stride=(2,2,2), act=actc)),

            ('d2_uconv1', ConvST3d(int(ngf/2), ngf, (3,3,3), padding=(1,1,1), scale_factor=[1,2,2], scale_mode='trilinear', act=actd)),
            ('d2_conv1', ConvST3d(ngf, ngf, (3,3,3), padding=(1,1,1), act=actd)),
            ('d2_uconv2', ConvST3d(ngf, ngf*2, (3,3,3), padding=(1,1,1), scale_factor=[2,2,2], scale_mode='trilinear', act=actd)),
            ('d2_conv2', ConvST3d(ngf*2, ngf*2, (3,3,3), padding=(1,1,1), act=actd)),

            ('convout', ConvST3d(ngf*2, 3, (3,3,3), padding=(1,1,1), act=acto))
        ]

        self.model = nn.Sequential(OrderedDict(seq))

    def forward(self, x):
        return self.model(x)




