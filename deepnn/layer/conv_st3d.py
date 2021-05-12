import torch
import torch.nn as nn

class ConvST3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super(ConvST3d, self).__init__()
        kt, kh, kw = kernel_size
        self.conv_spatial = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, kh, kw), **kwargs)
        if kt != 1:
            self.conv_temporal = nn.Conv3d(in_channels=out_channels, out_channels=out_channels, kernel_size=(kt, 1, 1), **kwargs)
        else:
            self.conv_temporal = None

    def forward(self, x):
        y = self.conv_spatial(x)
        if self.conv_temporal is not None:
            y = self.conv_temporal(y)
        return y