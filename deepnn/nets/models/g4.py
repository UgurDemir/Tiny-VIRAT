# Some part of the code are taken from https://github.com/xinntao/ESRGAN/blob/master/block.py
import torch
import torch.nn as nn

from collections import OrderedDict

from deepnn.layer.conv3d import ConvST3d

"""
ESRGAN (winner of PRIM 2018) Residual in Residual Dense Block
Args:
    cnorm: normalization layer after convolution
    unorm: normalization layer after upconv layer
    actc: activation function for encoding step
    actd: activation function for decoding step
    scse: Skip connection from start to end [True|False]
"""
class G4(nn.Module):
    def __init__(self, ngf=64, nbf=32, nblock=5, nup=2, conv_mode='ST', act='lrelu', acto=None, norm=None):
        super(G4, self).__init__()

        self.convin = ConvST3d(3, ngf, (3,3,3), padding=(1,1,1), act=None, norm=None, conv_mode=conv_mode)
        self.res_hall = nn.Sequential(OrderedDict([('rrdb%d'%(i+1), RRDB(inch=ngf, nbf=nbf, norm=norm, act=act, conv_mode=conv_mode)) for i in range(nblock)]))
        self.convbout = ConvST3d(ngf, ngf, (3,3,3), padding=(1,1,1), act=None, norm=norm, conv_mode=conv_mode)

        up = [('upconv%d'%(i+1), ConvST3d(ngf, ngf, (3,3,3), padding=(1,1,1), scale_factor=[1,2,2], scale_mode='trilinear', act=act, norm=None, conv_mode=conv_mode)) for i in range(nup)]
        self.up_hall = nn.Sequential(OrderedDict(up))
        self.conv1 = ConvST3d(ngf, ngf, (3,3,3), padding=(1,1,1), act=act, norm=None, conv_mode=conv_mode)
        self.conv2 = ConvST3d(ngf, 3, (3,3,3), padding=(1,1,1), act=acto, norm=None, conv_mode=conv_mode)

    def forward(self, x):
        c1 = self.convin(x)
        y = self.res_hall(c1)
        y = self.convbout(y)
        y = y + c1
        y = self.up_hall(y)
        y = self.conv1(y)
        y = self.conv2(y)
        return y

class ResidualDenseBlock(nn.Module):
    """
    Residual Dense Block
    style: 5 convs
    """

    def __init__(self, inch, outch, res_scale=0.2, kernel_size=(3,3,3), norm=None, act='lrelu', conv_mode='ST'):
        super(ResidualDenseBlock, self).__init__()
        self.res_scale = res_scale
        
        self.conv1 = ConvST3d(inch,           outch, kernel_size, padding=(1,1,1), act=act, norm=norm, conv_mode=conv_mode)
        self.conv2 = ConvST3d(inch + outch,   outch, kernel_size, padding=(1,1,1), act=act, norm=norm, conv_mode=conv_mode)
        self.conv3 = ConvST3d(inch + outch*2, outch, kernel_size, padding=(1,1,1), act=act, norm=norm, conv_mode=conv_mode)
        self.conv4 = ConvST3d(inch + outch*3, outch, kernel_size, padding=(1,1,1), act=act, norm=norm, conv_mode=conv_mode)
        self.conv5 = ConvST3d(inch+outch*4,   inch,  kernel_size, padding=(1,1,1), act=None, norm=norm, conv_mode=conv_mode)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(torch.cat((x, x1), 1))
        x3 = self.conv3(torch.cat((x, x1, x2), 1))
        x4 = self.conv4(torch.cat((x, x1, x2, x3), 1))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5.mul(self.res_scale) + x


class RRDB(nn.Module):
    """
    Residual in Residual Dense Block
    """

    def __init__(self, inch, nbf, res_scale=0.2, kernel_size=(3,3,3), norm=None, act='lrelu', conv_mode='ST'):
        super(RRDB, self).__init__()
        self.res_scale = res_scale

        self.RDB1 = ResidualDenseBlock(inch=inch, outch=nbf, res_scale=res_scale, kernel_size=kernel_size, norm=norm, act=act, conv_mode=conv_mode)
        self.RDB2 = ResidualDenseBlock(inch=inch, outch=nbf, res_scale=res_scale, kernel_size=kernel_size, norm=norm, act=act, conv_mode=conv_mode)
        self.RDB3 = ResidualDenseBlock(inch=inch, outch=nbf, res_scale=res_scale, kernel_size=kernel_size, norm=norm, act=act, conv_mode=conv_mode)

    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out.mul(self.res_scale) + x

