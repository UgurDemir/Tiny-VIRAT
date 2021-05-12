import torch
import torch.nn as nn

from deepnn.nets.network_wrapper import NetworkWrapper
from deepnn.loss.featureloss import FeatLoss
from deepnn.loss.gramloss import GramLoss
from deepnn.loss.styleloss import StyleLoss
from deepnn.loss.maskedloss import MaskedLoss
from deepnn.loss import total_variation_loss


class GeneratorWrapper(NetworkWrapper):
	def __init__(self, name, loss=None, **conf):
		super(GeneratorWrapper, self).__init__(name=name, **conf)
		
		if loss is not None:
			# Adversarial Loss
			self.wtadv = loss['adv']['wt'] if 'adv' in loss else 0.0
			self.wtadv_seg = loss['adv_seg']['wt'] if 'adv_seg' in loss else 0.0

			# Action Loss
			if 'ac' in loss:
				self.wtac = loss['ac']['wt']
				self.crit_ac = nn.__dict__[loss['ac']['loss']]()
			else:
				self.wtac = 0.0

			# Feature Loss
			if 'feat' in loss:
				self.wtfeat = loss['feat'].pop('wt')
				self.featloss = FeatLoss(**loss['feat'])
			else:
				self.featloss = None
				self.wtfeat = 0.0

			# Gram Loss
			if 'gram' in loss:
				self.wtgram = loss['gram'].pop('wt')
				self.gramloss = GramLoss(**loss['gram'])
			else:
				self.gramloss = None
				self.wtgram = 0.0
			
			# Style Loss
			if 'style' in loss:
				self.wtstyle = loss['style'].pop('wt')
				self.styleloss = StyleLoss(**loss['style'])
			else:
				self.styleloss = None
				self.wtstyle = 0.0

			# Total Variation Loss
			self.wttv = loss['tv'].pop('wt') if 'tv' in loss else 0.0

			# Reconstruction Loss
			if 'l1' in loss:
				self.recloss = nn.L1Loss()
				self.maskloss = MaskedLoss(loss='l1')
				self.wtrec = loss['l1']['wt']
				self.wtmask = loss['l1']['wtmask'] if 'wtmask' in loss['l1'] else 0.0
			elif 'l2' in loss:
				self.recloss = nn.MSELoss()
				self.maskloss = MaskedLoss(loss='l2')
				self.wtrec = loss['l2']['wt']
				self.wtmask = loss['l2']['wtmask'] if 'wtmask' in loss['l2'] else 0.0		
			else:
				self.recloss = None
				self.maskloss = None
				self.wtrec = 0.0
			
			if 'l1_wloc' in loss:
				self.recloss = nn.L1Loss(reduction='none')
				self.wtrec_loc = loss['l1_wloc']['wt']
				self.detach_loc = loss['l1_wloc']['detach_loc']
			else:
				self.wtrec_loc = 0.0

			# Localization Loss
			if 'loc' in loss:
				self.wtloc = loss['loc']['wt']
				self.crit_loc = nn.__dict__[loss['loc']['loss']]()
			else:
				self.wtloc = 0.0

			# Localization Regularization
			if 'loc_reg' in loss:
				self.wtlocreg = loss['loc_reg']['wt']
				self.loc_reg_type = loss['loc_reg'].get('loss', 'l1')
			else:
				self.wtlocreg = 0.0

	def calc_loss(self, rec, target, loc_pred=None, disc=None, disc_kwargs={}, neta=None, action_labels=None):
		info = {}
		total_loss = 0

		# Adversarial Loss
		# Disc
		if self.wtadv > 0.0 and disc is not None:
			pred_fake4g = disc(rec, **disc_kwargs)
			pred_real4g = disc(target, **disc_kwargs) if disc.mode == 'rel' else None
			adv = disc.adv_loss(pred_fake=pred_fake4g, pred_real=pred_real4g)

			adv_loss = adv * self.wtadv
			total_loss += adv_loss
			info['g_adv'] = adv_loss.item()

		# Action Loss
		if self.wtac > 0.0 and neta is not None:
			y = neta(rec)
			ac = self.crit_ac(y, action_labels)
			
			ac_loss = ac * self.wtac
			total_loss += ac_loss
			info['g_ac'] = ac_loss.item()

		# Feature Loss
		if self.wtfeat > 0.0:
			# If gt and rec are single image
			if self.featloss.model_name == 'i3d':
				l_feat = self.featloss(rec, target) * self.wtfeat
			# If gt and rec are video frames
			elif self.featloss.model_name == 'vgg16':
				b, c, t, h, w = rec.size()
				rec_batch = rec.permute(0,2,1,3,4).contiguous().view(b*t, c, h, w)
				target_batch = target.permute(0,2,1,3,4).contiguous().view(b*t, c, h, w)
				l_feat = self.featloss(rec_batch, target_batch) * self.wtfeat
			else:
				raise NotImplementedError('Feature loss is not implemented for tensors with different dimensions')
			total_loss += l_feat
			info['g_feat'] = l_feat.item()

		# Gram Loss
		if self.wtgram > 0.0:
			# If gt and rec are single image
			if self.gramloss.model_name == 'i3d':
				l_gram = self.gramloss(rec, target) * self.wtgram
			# If gt and rec are video frames
			elif self.gramloss.model_name == 'vgg16':
				b, c, t, h, w = rec.size()
				rec_batch = rec.permute(0,2,1,3,4).contiguous().view(b*t, c, h, w)
				target_batch = target.permute(0,2,1,3,4).contiguous().view(b*t, c, h, w)
				l_gram = self.gramloss(rec_batch, target_batch) * self.wtgram
			else:
				raise NotImplementedError('Gram loss is not implemented for tensors with different dimensions')
			total_loss += l_gram
			info['g_gram'] = l_gram.item()

		# Style Loss
		if self.wtstyle > 0.0:
			# If gt and rec are single image
			if self.styleloss.model_name == 'i3d':
				l_style = self.styleloss(rec, target) * self.wtstyle
			# If gt and rec are video frames
			elif self.styleloss.model_name == 'vgg16':
				b, c, t, h, w = rec.size()
				rec_batch = rec.permute(0,2,1,3,4).contiguous().view(b*t, c, h, w)
				target_batch = target.permute(0,2,1,3,4).contiguous().view(b*t, c, h, w)
				l_style = self.styleloss(rec_batch, target_batch) * self.wtstyle
			else:
				raise NotImplementedError('Style loss is not implemented for tensors with different dimensions')
			total_loss += l_style
			info['g_style'] = l_style.item()

		# Total Variation Loss
		if self.wttv > 0.0:
			# If gt and rec are single image
			if rec.dim() == 4:
				l_tv = total_variation_loss(rec) * self.wttv
			# If gt and rec are video frames
			elif rec.dim() == 5:
				b, c, t, h, w = rec.size()
				rec_batch = rec.permute(0,2,1,3,4).contiguous().view(b*t, c, h, w)
				l_tv = total_variation_loss(rec_batch) * self.wttv
			else:
				raise NotImplementedError('Style loss is not implemented for tensors with different dimensions')
			total_loss += l_tv
			info['g_tv'] = l_tv.item()

		# Reconstruction Loss
		if self.wtrec > 0.0:
			l_rec = self.recloss(rec, target) * self.wtrec
			total_loss += l_rec
			info['g_rec'] = l_rec.item()

		# Localization weighted reconstruction loss
		if self.wtrec_loc > 0.0:
			if self.detach_loc:
				loc_weight = loc_pred.detach()
			else:
				loc_weight = loc_pred

			l_locwrec = (self.recloss(rec, target) * loc_weight).mean()
			total_loss += l_locwrec * self.wtrec_loc
			info['g_recloc'] = l_locwrec.item()

		
		# Localizer training
		if self.wtloc > 0.0:
			ac_pred_l = neta(rec.detach()*loc_pred)
			l_loc = self.crit_loc(ac_pred_l, action_labels) * self.wtloc
			total_loss += l_loc
			info['g_loc'] = l_loc.item()

		# Localizer regularization (Sparsity)
		if self.wtlocreg > 0.0:
			if self.loc_reg_type == 'l1':
				l_locreg = loc_pred.abs().mean() * self.wtlocreg
			elif self.loc_reg_type == 'chexp':
				l_locreg = torch.exp(loc_pred.mean(dim=2).mean(dim=2).mean(dim=2)).mean()
			elif self.loc_reg_type == 'exp':
				l_locreg = torch.exp(loc_pred).mean()

			total_loss += l_locreg
			info['g_loc_reg'] = l_locreg.item()

		info['g_loss'] = total_loss.item()
		return total_loss, info
