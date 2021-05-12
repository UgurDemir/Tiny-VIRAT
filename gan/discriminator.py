import torch

from deepnn.nets.network_wrapper import NetworkWrapper
from deepnn.loss.ganloss import GANLoss

class DiscriminatorWrapper(NetworkWrapper):
	def __init__(self, name, loss, **conf):
		super(DiscriminatorWrapper, self).__init__(name=name, **conf)
		self.wtgan = loss['wt']
		self.mode = loss['mode']

		self.gan_loss = GANLoss(loss_type=loss['loss_type'], target_real_label=loss['real'], target_fake_label=loss['fake'])
		

	def calc_loss(self, pred_fake, pred_real):
		dinfo = {}
		# Standart Vanilla GAN
		if self.mode == 'std':
			d_fake_loss = self.gan_loss(pred_fake, isreal=False) * 0.5
			d_real_loss = self.gan_loss(pred_real, isreal=True) * 0.5
			dloss = (d_fake_loss + d_real_loss) * self.wtgan
		# Relativistic GAN
		elif self.mode == 'rel':
			d_fake_loss = self.gan_loss(pred_fake - torch.mean(pred_real), isreal=False) * 0.5
			d_real_loss = self.gan_loss(pred_real - torch.mean(pred_fake), isreal=True) * 0.5
			dloss = (d_fake_loss + d_real_loss) * self.wtgan
		else:
			raise NotImplementedError('Unknown GAN mode: ' + str(self.mode))

		dinfo['d_loss'] = dloss.item()
		dinfo['dfake_loss'] = d_fake_loss.item()
		dinfo['dreal_loss'] = d_real_loss.item()

		return dloss, dinfo


	def adv_loss(self, pred_fake, pred_real):
		# Standart Vanilla GAN
		if self.mode == 'std':
			l_adv = self.gan_loss(pred_fake, isreal=True)
		# Relativistic GAN
		elif self.mode == 'rel':
			l_adv_fake = self.gan_loss(pred_fake - torch.mean(pred_real), isreal=True) * 0.5
			l_adv_real = self.gan_loss(pred_real - torch.mean(pred_fake), isreal=False) * 0.5
			l_adv = l_adv_fake + l_adv_real
		else:
			raise NotImplementedError('Unknown GAN mode: ' + str(self.mode))

		
		return l_adv
