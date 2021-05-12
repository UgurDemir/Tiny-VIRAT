import torch
import torch.nn as nn

class InterpolatedConv2D(nn.Module):
	"""
		Args:
			mode: Upsampling method [nearest|bilinear]
	"""
	def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, scale_factor=2, mode='nearest'):
		super(InterpolatedConv2D, self).__init__()
		#self.up = nn.Upsample(scale_factor=scale_factor, mode=mode)
		self.scale_factor = scale_factor
		self.mode = mode
		self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
		self.weight = self.conv.weight
		self.bias = self.conv.bias

	def forward(self, x):
		#y = self.up(x)
		y = nn.functional.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)

		y = self.conv(y)

		return y

	def __repr__(self):
		return 'InterpolatedConv2D'

class InterpolatedConv3D(nn.Module):
	def __init__(self, scale_factor=[2,2,2], mode='nearest', **kwargs):
		super(InterpolatedConv3D, self).__init__()
		self.scale_factor = scale_factor
		self.mode = mode
		self.conv = nn.Conv3d(**kwargs)

	def forward(self, x):
		#_,_, t, h, w = x.size()
		#st, sh, sw = self.scale_factor
		#out_size = [int(st*t), int(sh*h), int(sw*w)]
		y = torch.nn.functional.interpolate(x, scale_factor=self.scale_factor, mode=self.scale_mode, align_corners=False)

		y = self.conv(y)
		return y