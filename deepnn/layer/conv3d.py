import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm as spec_norm

"""
Args:
    mode: [3D | ST]
"""
class ConvST3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=(1,1,1), padding=(0,0,0), 
                norm=None, act=None, scale_factor=[1,1,1], scale_mode='nearest', conv_mode='ST', **kwargs):
        super(ConvST3d, self).__init__()
        self.scale_factor = scale_factor
        self.scale_mode = scale_mode
        kt, kh, kw = kernel_size
        pt, ph, pw = padding
        st, sh, sw = stride
        
        # Convolution
        if conv_mode == 'ST':
            # Spatial conv
            self.conv_spatial = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, 
                                kernel_size=(1, kh, kw), stride=(1,sh, sw), padding=(0,ph,pw), **kwargs)

            # Temporal conv
            if kt != 1:
                self.conv_temporal = nn.Conv3d(in_channels=out_channels, out_channels=out_channels, 
                                kernel_size=(kt, 1, 1), stride=(st,1,1), padding=(pt,0,0), **kwargs)
            else:
                self.conv_temporal = None
        elif conv_mode == '3D':
            self.conv_spatial = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, 
                                kernel_size=kernel_size, stride=stride, padding=padding, **kwargs)
            self.conv_temporal = None

        # Normalization
        if norm == 'bn':
            self.norm = nn.BatchNorm3d(out_channels)
        elif norm == 'in':
            self.norm = nn.InstanceNorm3d(out_channels)
        elif norm == 'spec':
            self.norm = None
            if self.conv_spatial is not None: 
                self.conv_spatial = spec_norm(self.conv_spatial)
            if self.conv_temporal is not None:
                self.conv_temporal = spec_norm(self.conv_temporal)
        elif norm is None:
            self.norm = None
        
        # Activation
        if act == 'relu':
            self.act = nn.ReLU()
        elif act == 'lrelu':
            self.act = nn.LeakyReLU(negative_slope=0.2)
        elif act == 'tanh':
            self.act = nn.Tanh()
        elif act == 'sigmoid':
            self.act = nn.Sigmoid()
        elif act is None:
            self.act = None

    def forward(self, x):
        y = x
        st, sh, sw = self.scale_factor
        if st != 1 or sh != 1 or sw != 1:
            y = torch.nn.functional.interpolate(x, scale_factor=self.scale_factor, mode=self.scale_mode, align_corners=False)

        y = self.conv_spatial(y)

        if self.conv_temporal is not None:
            y = self.conv_temporal(y)

        if self.norm is not None:
            y = self.norm(y)
        
        if self.act is not None:
            y = self.act(y)

        return y