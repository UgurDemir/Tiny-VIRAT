import torch
import torch.nn as nn

from collections import OrderedDict

from deepnn.layer.conv3d import ConvST3d

class PG1(nn.Module):
    def __init__(self, ngf=32, actenc='lrelu', actdec='lrelu', acto='tanh', norm='spec', conv_mode='3D'):
        super(PG1, self).__init__()
        
        self.convin = ConvST3d(3, ngf, (3,3,3), padding=(1,1,1), act=actenc, norm=norm, conv_mode=conv_mode)

        blocks = [
            Block(inch=ngf, outch=ngf, scale_factor=[1,2,2], norm=norm, actenc=actenc, actup=actdec, conv_mode=conv_mode),
            Block(inch=ngf, outch=ngf, scale_factor=[1,2,2], norm=norm, actenc=actenc, actup=actdec, conv_mode=conv_mode),
            Block(inch=ngf, outch=ngf, scale_factor=[1,2,2], norm=norm, actenc=actenc, actup=actdec, conv_mode=conv_mode)
        ]

        proj = [
            RGBBlock(ngf, 2, stride=(2,1,1), act=actenc, acto=acto, norm=norm, conv_mode=conv_mode),
            RGBBlock(ngf, 1, stride=(2,1,1), act=actenc, acto=acto, norm=norm, conv_mode=conv_mode),
            RGBBlock(ngf, 1, stride=(1,1,1), act=actenc, acto=acto, norm=norm, conv_mode=conv_mode)
        ]

        self.layers = nn.ModuleList(blocks)
        self.rgb_proj = nn.ModuleList(proj)

    # Args:
    #   x: input low res image
    #   scale: 1: 2x | 2: 4x | 3: 8x
    #   fadein: fadein factor for smooth transition between consequtive scales
    #           out = current_rgb * (1-fadein) + previous_rgb * fadein

    def forward(self, x, scale=None, fadein=0.0):
        scale = scale if scale is not None else len(self.layers)

        # Initial conv block to increase number of channel
        y = self.convin(x)

        # Encoding and upscaling. Core Block
        for i in range(scale):
            feat_prev = y
            y = self.layers[i](y)

        # RGB Projection
        y = self.rgb_proj[scale-1](y)

        # Fade in the new layer
        if fadein > 0.0:
            assert scale > 1
            y_prev = self.rgb_proj[scale-2](feat_prev)
            y_prev_up = nn.functional.interpolate(y_prev, scale_factor=[2, 2, 2], mode='trilinear', align_corners=False)

            y = y_prev_up * fadein + y * (1.0 - fadein)


        return y

class Block(nn.Module):
    def __init__(self, inch=32, outch=32, scale_factor=[1,2,2], scale_mode='trilinear', norm=None, actenc='lrelu', actup='lrelu', conv_mode='3D'):
        super(Block, self).__init__()
        self.enc = DenseBlock(inch, outch, res_scale=0.2, kernel_size=(3,3,3), norm=norm, act=actenc, conv_mode=conv_mode)
        self.up = ConvST3d(outch, outch, (3,3,3), padding=(1,1,1), scale_factor=scale_factor, scale_mode=scale_mode, act=actup, norm=norm, conv_mode=conv_mode)
        self.dec = DenseBlock(outch, outch, res_scale=0.2, kernel_size=(3,3,3), norm=norm, act=actup, conv_mode=conv_mode)

    def forward(self, x):
        y = self.enc(x)
        y = self.up(y)
        y = self.dec(y)

        return y

class RGBBlock(nn.Module):
    def __init__(self, inch, nlayer, stride, act='lrelu', acto='tanh', norm=None, conv_mode='3D'):
        super(RGBBlock, self).__init__()
        seq = []
        for i in range(nlayer):
            outch = inch#//(2**(i+1))
            seq += [('l%d'%i, ConvST3d(inch, outch, kernel_size=(3,3,3), stride=stride ,padding=(1,1,1), act=act, norm=norm, conv_mode=conv_mode))]
            inch = outch
        
        seq += [('l%d'%nlayer, ConvST3d(inch, 3, kernel_size=(1,1,1), act=acto, norm=None, conv_mode='3D'))]

        self.seq = nn.Sequential(OrderedDict(seq))

    def forward(self, x):
        return self.seq(x)


class DenseBlock(nn.Module):
    """
    Dense Block
    style: 3 convs
    """

    def __init__(self, inch, outch, res_scale=0.2, kernel_size=(3,3,3), norm=None, act='lrelu', conv_mode='ST'):
        super(DenseBlock, self).__init__()
        self.res_scale = res_scale
        
        self.conv1 = ConvST3d(inch,           outch, kernel_size, padding=(1,1,1), act=act, norm=norm, conv_mode=conv_mode)
        self.conv2 = ConvST3d(inch + outch,   outch, kernel_size, padding=(1,1,1), act=act, norm=norm, conv_mode=conv_mode)
        self.conv3 = ConvST3d(inch + outch*2, outch, kernel_size, padding=(1,1,1), act=act, norm=None, conv_mode=conv_mode)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(torch.cat((x, x1), 1))
        x3 = self.conv3(torch.cat((x, x1, x2), 1))
        return x3.mul(self.res_scale) + x
