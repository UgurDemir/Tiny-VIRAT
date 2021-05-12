import torch
import torch.nn as nn

class FeatureExtractor(nn.Module):
	def __init__(self, slicenets):
		super(FeatureExtractor, self).__init__()
		self.slicenets = slicenets

		i = 0
		for name, nets in slicenets:
			for l in nets:
				self.add_module(str(i), l)
				i+=1

		for name, param in self.named_parameters():
			param.requires_grad = False

	def forward(self, x):
		outmap = {}
		h = x#.clone()
		for name, snet in self.slicenets:
			h = snet(h)
			outmap[name] = h
		return outmap
