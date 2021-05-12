import torch
import torch.nn as nn

class PassLayer(nn.Module):
	def __init__(self, *args, **kwargs):
		super(PassLayer, self).__init__()

	def forward(self, x):
		return x

	def __repr__(self):
		return 'PassLayer'
