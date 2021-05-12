import torch
import torch.nn as nn
from torch.autograd import Variable

"""
Adapted from https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
"""
class GANLoss(nn.Module):
	def __init__(self, loss_type='BCELoss', target_real_label=1.0, target_fake_label=0.0):
		super(GANLoss, self).__init__()
		self.real_label = target_real_label
		self.fake_label = target_fake_label
		self.real_label_var = None
		self.fake_label_var = None
		self.loss = nn.__dict__[loss_type]()

	def get_target(self, input, isreal):
		target_tensor = None

		if isreal:
			create_label = ((self.real_label_var is None) or
							(self.real_label_var.numel() != input.numel()) or
							(self.real_label_var.device != input.device))
			if create_label:
				self.real_label_var = input.new_full(input.size(), self.real_label, requires_grad=False)
			target_tensor = self.real_label_var
		else:
			create_label = ((self.fake_label_var is None) or
							(self.fake_label_var.numel() != input.numel()) or
							(self.fake_label_var.device != input.device))

			if create_label:
				self.fake_label_var = input.new_full(input.size(), self.fake_label, requires_grad=False)
			target_tensor = self.fake_label_var

		return target_tensor

	def __call__(self, input, isreal):
		target = self.get_target(input, isreal)
		return self.loss(input, target)
