import torch
import torch.nn as nn

from collections import OrderedDict

from deepnn.loss.ganloss import GANLoss
import deepnn.nets.builder as lbuild

class PatchDiscriminator(nn.Module):
	def __init__(self, ndf, n_layer, inch=3, norm='bn', actc='lrelu', use_sigmoid=True):
		super(PatchDiscriminator, self).__init__()
		self.advloss = GANLoss(use_lsgan=False)
		#self.cuda_objects += [self.advloss]

		cact = lbuild.act(actc)
		norml = lbuild.norm(norm)

		seq = [
			('l1_conv', nn.Conv2d(in_channels=inch, out_channels=ndf, kernel_size=4, stride=2, padding=1, bias=False)),
			('l1_act', cact())
		]

		nin = ndf
		for i in range(2, n_layer):
			nout = (2**(i-1))*ndf
			seq += [
				('l%s_conv'%str(i), nn.Conv2d(in_channels=nin, out_channels=nout, kernel_size=4, stride=2, padding=1, bias=False)),
				('l%s_norm'%str(i), norml(nout)),
				('l%s_act'%str(i), cact())
			]
			nin = nout

		seq += [
			('l%s_conv'%str(n_layer), nn.Conv2d(in_channels=nin, out_channels=1, kernel_size=4, padding=1, bias=False))
		]

		if use_sigmoid: seq += [('l%s_act'%str(n_layer), nn.Sigmoid())]

		self.model = nn.Sequential(OrderedDict(seq))

	def forward(self, x):
		return self.model(x)

	def calc_loss(self, pred, label):
		dloss = self.advloss(pred, label)
		return dloss

	def adv_loss(self, pred_fake):
		return self.calc_loss(pred_fake, True)

	def backpass(self, rec, real):
		# Accuracy
		acc_fake, acc_real, acc = self._acc(pred_fake, pred_real)

		return {'train_lossD': d_loss.data[0], 'train_dfake': d_fake_loss.data[0],
				'train_dreal': d_real_loss.data[0],
				'train_acc_fake':acc_fake, 'train_acc_real':acc_real, 'train_acc':acc}

	def validate(self, pred, target):
		rec = pred
		real = target
		pred_fake = self.model.forward(rec.detach())
		d_fake_loss = self.advloss(pred_fake, False)
		pred_real = self.model.forward(real)
		d_real_loss = self.advloss(pred_real, True)
		d_loss = (d_fake_loss + d_real_loss) * 0.5

		# Accuracy
		acc_fake, acc_real, acc = self._acc(pred_fake, pred_real)

		return {'val_lossD': d_loss.data[0], 'val_dfake': d_fake_loss.data[0],
				'val_dreal': d_real_loss.data[0],
				'val_acc_fake':acc_fake, 'val_acc_real':acc_real, 'val_acc':acc}

	

	def _acc(self, pred_fake, pred_real):
		acc_fake = float((pred_fake.data.resize_(pred_fake.size()[0],1) < 0.5).sum()) / float(pred_fake.size()[0])
		acc_real = float((pred_real.data.resize_(pred_real.size()[0],1) > 0.5).sum()) / float(pred_real.size()[0])
		acc = (acc_fake + acc_real) / 2.0

		return acc_fake, acc_real, acc
