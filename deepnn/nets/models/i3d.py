# Modified from https://github.com/piergiaj/pytorch-i3d/blob/master/pytorch_i3d.py

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

import os
import sys
from collections import OrderedDict


class InceptionI3d(nn.Module):
    def __init__(self, num_classes=157, spatial_squeeze=True, name='inception_i3d', in_channels=3, dropout_keep_prob=0.5):
        super(InceptionI3d, self).__init__()
        self.num_classes = num_classes
        self.spatial_squeeze = spatial_squeeze

        seq = [
            ('Conv3d_1a_7x7', Unit3D(in_channels=in_channels, output_channels=64, kernel_shape=[7, 7, 7], stride=(2, 2, 2), padding=(3,3,3))),
            
            #('MaxPool3d_2a_3x3', MaxPool3dSamePadding(kernel_size=[1, 3, 3], stride=(1, 2, 2), padding=0)),
            ('Conv3d_2b_1x1', Unit3D(in_channels=64, output_channels=64, kernel_shape=[1, 1, 1], padding=0)),
            ('Conv3d_2c_3x3', Unit3D(in_channels=64, output_channels=192, kernel_shape=[3, 3, 3], padding=1)),
            
            ('MaxPool3d_3a_3x3', MaxPool3dSamePadding(kernel_size=[1, 3, 3], stride=(1, 2, 2), padding=0)),
            ('Mixed_3b', InceptionModule(192, [64,96,128,16,32,32])),
            ('Mixed_3c', InceptionModule(256, [128,128,192,32,96,64])),
            
            ('MaxPool3d_4a_3x3', MaxPool3dSamePadding(kernel_size=[3, 3, 3], stride=(2, 2, 2), padding=0)),
            ('Mixed_4b', InceptionModule(128+192+96+64, [192,96,208,16,48,64])),
            ('Mixed_4c', InceptionModule(192+208+48+64, [160,112,224,24,64,64])),
            ('Mixed_4d', InceptionModule(160+224+64+64, [128,128,256,24,64,64])),
            ('Mixed_4e', InceptionModule(128+256+64+64, [112,144,288,32,64,64])),
            ('Mixed_4f', InceptionModule(112+288+64+64, [256,160,320,32,128,128])),

            ('MaxPool3d_5a_2x2', MaxPool3dSamePadding(kernel_size=[2, 2, 2], stride=(2, 2, 2), padding=0)),
            ('Mixed_5b', InceptionModule(256+320+128+128, [256,160,320,32,128,128])),
            ('Mixed_5c', InceptionModule(256+320+128+128, [384,192,384,48,128,128])),

            ('avg_pool', nn.AvgPool3d(kernel_size=[2, 7, 7], stride=(1, 1, 1))),
            ('dropout', nn.Dropout(dropout_keep_prob)),
        ]

        self.logits = Unit3D(in_channels=384+384+128+128, output_channels=self.num_classes, kernel_shape=[1, 1, 1], padding=0, act=None, norm=None, use_bias=True)

        self.model = nn.Sequential(OrderedDict(seq))

    def forward(self, x):
        y = self.model(x)
        y = self.logits(y)
        if self.spatial_squeeze:
            y = y.squeeze(3).squeeze(3).squeeze(2)
        return y

    def replace_logits(self, num_classes):
        self.num_classes = num_classes
        self.logits = Unit3D(in_channels=384+384+128+128, output_channels=self.num_classes, kernel_shape=[1, 1, 1], padding=0, act=None, norm=None, use_bias=True)

class MaxPool3dSamePadding(nn.MaxPool3d):
    
    def compute_pad(self, dim, s):
        if s % self.stride[dim] == 0:
            return max(self.kernel_size[dim] - self.stride[dim], 0)
        else:
            return max(self.kernel_size[dim] - (s % self.stride[dim]), 0)

    def forward(self, x):
        # compute 'same' padding
        (batch, channel, t, h, w) = x.size()
        #print t,h,w
        out_t = np.ceil(float(t) / float(self.stride[0]))
        out_h = np.ceil(float(h) / float(self.stride[1]))
        out_w = np.ceil(float(w) / float(self.stride[2]))
        #print out_t, out_h, out_w
        pad_t = self.compute_pad(0, t)
        pad_h = self.compute_pad(1, h)
        pad_w = self.compute_pad(2, w)
        #print pad_t, pad_h, pad_w

        pad_t_f = pad_t // 2
        pad_t_b = pad_t - pad_t_f
        pad_h_f = pad_h // 2
        pad_h_b = pad_h - pad_h_f
        pad_w_f = pad_w // 2
        pad_w_b = pad_w - pad_w_f

        pad = (pad_w_f, pad_w_b, pad_h_f, pad_h_b, pad_t_f, pad_t_b)
        #print x.size()
        #print pad
        x = F.pad(x, pad)
        return super(MaxPool3dSamePadding, self).forward(x)
    

class Unit3D(nn.Module):

    def __init__(self, in_channels,
                 output_channels,
                 kernel_shape=(1, 1, 1),
                 stride=(1, 1, 1),
                 padding=0,
                 act='ReLU',
                 norm='BatchNorm3d',
                 use_bias=False):
        
        """Initializes Unit3D module."""
        super(Unit3D, self).__init__()
        
        self.output_channels = output_channels
        self.kernel_shape = kernel_shape
        self.stride = stride
        self.use_bias = use_bias
        #self.name = name
        #self.padding = padding
        
        self.conv3d = nn.Conv3d(in_channels=in_channels,
                                out_channels=self.output_channels,
                                kernel_size=self.kernel_shape,
                                stride=self.stride,
                                padding=0, # we always want padding to be 0 here. We will dynamically pad based on input size in forward function
                                bias=self.use_bias)
        
        
        self.bn = nn.__dict__[norm](self.output_channels, eps=0.001, momentum=0.01) if norm is not None else None
        self.act = nn.__dict__[act]() if act is not None else None
        

    def compute_pad(self, dim, s):
        if s % self.stride[dim] == 0:
            return max(self.kernel_shape[dim] - self.stride[dim], 0)
        else:
            return max(self.kernel_shape[dim] - (s % self.stride[dim]), 0)

            
    def forward(self, x):
        # compute 'same' padding
        (batch, channel, t, h, w) = x.size()
        #print t,h,w
        out_t = np.ceil(float(t) / float(self.stride[0]))
        out_h = np.ceil(float(h) / float(self.stride[1]))
        out_w = np.ceil(float(w) / float(self.stride[2]))
        #print out_t, out_h, out_w
        pad_t = self.compute_pad(0, t)
        pad_h = self.compute_pad(1, h)
        pad_w = self.compute_pad(2, w)
        #print pad_t, pad_h, pad_w

        pad_t_f = pad_t // 2
        pad_t_b = pad_t - pad_t_f
        pad_h_f = pad_h // 2
        pad_h_b = pad_h - pad_h_f
        pad_w_f = pad_w // 2
        pad_w_b = pad_w - pad_w_f

        pad = (pad_w_f, pad_w_b, pad_h_f, pad_h_b, pad_t_f, pad_t_b)
        #print x.size()
        #print pad
        x = F.pad(x, pad)
        #print x.size()        

        x = self.conv3d(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.act is not None:
            x = self.act(x)
        return x



class InceptionModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(InceptionModule, self).__init__()

        self.b0 = Unit3D(in_channels=in_channels, output_channels=out_channels[0], kernel_shape=[1, 1, 1], padding=0)
        #                 name=name+'/Branch_0/Conv3d_0a_1x1')
        self.b1a = Unit3D(in_channels=in_channels, output_channels=out_channels[1], kernel_shape=[1, 1, 1], padding=0)
        #                  name=name+'/Branch_1/Conv3d_0a_1x1')
        self.b1b = Unit3D(in_channels=out_channels[1], output_channels=out_channels[2], kernel_shape=[3, 3, 3])
        #                  name=name+'/Branch_1/Conv3d_0b_3x3')
        self.b2a = Unit3D(in_channels=in_channels, output_channels=out_channels[3], kernel_shape=[1, 1, 1], padding=0)
        #                  name=name+'/Branch_2/Conv3d_0a_1x1')
        self.b2b = Unit3D(in_channels=out_channels[3], output_channels=out_channels[4], kernel_shape=[3, 3, 3])
        #                  name=name+'/Branch_2/Conv3d_0b_3x3')
        self.b3a = MaxPool3dSamePadding(kernel_size=[3, 3, 3],
                                stride=(1, 1, 1), padding=0)
        self.b3b = Unit3D(in_channels=in_channels, output_channels=out_channels[5], kernel_shape=[1, 1, 1], padding=0)
        #                  name=name+'/Branch_3/Conv3d_0b_1x1')
        #self.name = name

    def forward(self, x):    
        b0 = self.b0(x)
        b1 = self.b1b(self.b1a(x))
        b2 = self.b2b(self.b2a(x))
        b3 = self.b3b(self.b3a(x))
        return torch.cat([b0,b1,b2,b3], dim=1)


