import torch
import numpy as np

class ZeroMeanUnitVarNormalization():
	def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
		mean = np.array(mean).astype(np.float32)
		self.m = torch.from_numpy(mean).resize_(mean.size,1,1)

		std = np.array(std).astype(np.float32)
		self.s = torch.from_numpy(std).resize_(std.size,1,1)

	def cuda(self):
		self.m = self.m.cuda()
		self.s = self.s.cuda()
		return self

	def normalize(self, x):
		x.sub_(self.m).div_(self.s)

	def denormalize(self, x):
		#x.mul_(self.s).add_(self.m)
		return (x *self.s) + self.m
