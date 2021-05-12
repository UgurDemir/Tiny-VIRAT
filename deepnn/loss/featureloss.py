import torch
import torch.nn as nn
from ..feature.vgg_feature import VGGFeatureExtractor
from ..feature.i3d_feature import I3DFeatureExtractor
from ..loss import build as build_loss

class FeatLoss(nn.Module):
	def __init__(self, layers=[], model='vgg16', loss='l2', **kwargs):
		super(FeatLoss, self).__init__()
		self.layers = layers
		self.model_name = model

		if model == 'vgg16':
			self.model = VGGFeatureExtractor([l[0] for l in self.layers], **kwargs)
		if model == 'i3d':
			self.model = I3DFeatureExtractor([l[0] for l in self.layers], **kwargs)
		else:
			raise Exception('Model is not supported')

		self.loss = build_loss(name=loss)

	def forward(self, x, y):
		f1 = self.model(x)
		f2 = self.model(y)

		total = 0
		for l, wt in self.layers:
			total += self.loss(f1[l], f2[l]) * wt
			del f1[l], f2[l]
		return total