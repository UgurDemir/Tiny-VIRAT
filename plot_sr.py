import os
import argparse
from collections import OrderedDict

import numpy as np
import matplotlib.pyplot as plt

from deepnn.util.plotfig import plot_hist


def plot_seg(log_path, out_path='./', save=False, display=True):
	logmap = np.load(log_path)['data'].item()

	print(logmap.keys())
	
	# Plot1 psnr(train vs val), ssim(train vs val)
	plot_hist(logmap,
		psnr={'src':['t_psnr','v_psnr'], 'label':['train','eval'],'c':['r-','b-'],'mode':'epoch'},
		ssim={'src':['t_ssim','v_ssim'], 'label':['train','eval'],'c':['r-','b-'],'mode':'epoch'}
	)
	if save: plt.savefig(os.path.join(out_path,'figure_1.png'))

	# Plot2 lossG(train vs val), lossD[_X](train vs val)
	argmap = OrderedDict({
		'1_loss_gen':{'src':['t_g_loss', 'v_g_loss'], 'label':['train', 'eval'], 'c':['r-', 'b-'], 'mode':'epoch'},
		'2_loss_disc':{'src':['t_d_loss','v_d_loss'], 'label':['train','val'], 'c':['r-','b-'], 'mode':'epoch'},
		'3_loss_seg':{'src':['t_dseg_loss','v_dseg_loss'], 'label':['train','val'], 'c':['r-','b-'], 'mode':'epoch'},	
	})
	plot_hist(logmap, **argmap)
	if save: plt.savefig(os.path.join(out_path,'figure_2.png'))

	# Plot3 lossG(train vs val), lossD[_X](train vs val)
	argmap = OrderedDict({
		'1_loss_gen':{'src':['t_g_loss', 'v_g_loss'], 'label':['train', 'eval'], 'c':['r-', 'b-'], 'mode':'iter'},
		'2_loss_disc':{'src':['t_d_loss','v_d_loss'], 'label':['train','val'], 'c':['r-','b-'], 'mode':'iter'},
		'3_loss_seg':{'src':['t_dseg_loss','v_dseg_loss'], 'label':['train','val'], 'c':['r-','b-'], 'mode':'iter'},
	})
	plot_hist(logmap, **argmap)
	if save: plt.savefig(os.path.join(out_path,'figure_3.png'))

	# Plot4 lossG in detail
	argmap = OrderedDict({
		'1_train_gen':{'src':['t_g_loss', 't_g_rec', 't_g_feat', 't_g_adv', 't_g_seg'],
						'label':['lossG', 'lrec', 'lfeat', 'ladv', 'lseg'],
						'c':['r-', 'k--', 'c--', 'g--', 'y--'], 'mode':'epoch'},
		'2_val_gen':{'src':['v_g_loss', 'v_g_rec', 'v_g_feat', 'v_g_adv', 'v_g_seg'],
							'label':['lossG', 'lrec', 'lfeat', 'ladv', 'lseg'],
							'c':['b-', 'k--', 'c--', 'g--', 'y--'], 'mode':'epoch'}
	})
	plot_hist(logmap, **argmap)
	if save: plt.savefig(os.path.join(out_path,'figure_4.png'))
	
	"""
	# Plot5 lossD in detail
	argmap = OrderedDict({
		'1_train_disc':{'src':['t_d_loss', 't_dfake_loss', 't_dreal_loss'],
						'label':['lossD', 'dfake', 'dreal'],
						'c':['r-', 'c--', 'g--'], 'mode':'epoch'},
		'2_val_disc':{'src':['v_d_loss', 'v_dfake_loss', 'v_dreal_loss'],
						'label':['lossD', 'dfake', 'dreal'],
						'c':['b-', 'c--', 'g--'], 'mode':'epoch'},
		'1_train_desc':{'src':['t_dseg_loss', 't_dseg_fake_loss', 't_dseg_real_loss'],
						'label':['lossD', 'dfake', 'dreal'],
						'c':['r-', 'c--', 'g--'], 'mode':'epoch'},
		'2_val_desc':{'src':['v_dseg_loss', 'v_dseg_fake_loss', 'v_dseg_real_loss'],
						'label':['lossD', 'dfake', 'dreal'],
						'c':['b-', 'c--', 'g--'], 'mode':'epoch'},
	})
	if plot_hist(logmap, **argmap) and save: plt.savefig(os.path.join(out_path,'figure_5.png'))
	"""
	
	# Plot6 netD accuracy
	p = plot_hist(logmap,**{
		'1_train_disc': {'src':['t_disc_acc', 'v_disc_acc' ,'t_disc_acc_fake', 't_disc_acc_real'],
					'label':['train_acc', 'val_acc', 'acc_fake', 'acc_real'],
					'c':['r-','b-', 'c--', 'g--'], 'mode':'epoch'},
		'2_val_disc': {'src':['v_disc_acc', 'v_disc_acc_fake', 'v_disc_acc_real'],
					'label':['acc', 'acc_fake', 'acc_real'],
					'c':['b-', 'c--', 'g--'], 'mode':'epoch'},
		'3_dseg': {'src':['t_dseg_acc','v_dseg_acc'],
					'label':['train_acc', 'val_acc'],
					'c':['r-', 'b-'], 'mode':'epoch'}
	})
	if p and save: plt.savefig(os.path.join(out_path,'figure_6.png'))

	# Plot7 learning rates
	plot_hist(logmap, **{
		'1_lr_gen': {'src':['lr_G'], 'label':['gen'], 'c':['r-'], 'mode':'epoch'},
		'2_lr_disc': {'src':['lr_D'], 'label':['disc'], 'c':['r-'], 'mode':'epoch'},
		'3_lr_dseg': {'src':['lr_Dseg'], 'label':['dseg'], 'c':['r-'], 'mode':'epoch'},
	})
	if save: plt.savefig(os.path.join(out_path,'figure_7.png'))

	if display: plt.show()
	
# Main
parser = argparse.ArgumentParser()
parser.add_argument('-p', type=str, default="./_results/log/hist.npz", help='Log file path')
parser.add_argument('-o', type=str, default="./_results/log", help='Out file path')
parser.add_argument('-d', action='store_true', help='Display plots')
parser.add_argument('-s', action='store_true', help='Save plots')

if __name__ == '__main__':
	opt = parser.parse_args()
	print(opt)

	plot_seg(log_path=opt.p, out_path=opt.o, save=opt.s, display=opt.d)