import numpy as np
import os
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
#import seaborn as sns
#import sys

def save_param_dict_density_plt(param_dict, epoch, it, logdir):
	atleastoneplotted = False
	for pname, params in param_dict.items():
		if len(params) == 0: continue

		bplotdata = []
		bplotlabel = []
		for n,lt in params:
			l = lt.numpy()
			bplotdata.append(l.flatten())
			bplotlabel.append(n)

		# Box plot
		plt.figure(figsize=(15, 10))
		plt.boxplot(bplotdata, labels=bplotlabel, sym='')
		args = plt.xticks()
		plt.xticks(*args,rotation=90)
		plt.grid(True)

		plt.tight_layout(rect=[0, 0, 0.85, 1.0])
		boxdir = os.path.join(logdir,'dist',pname,'box')
		if not os.path.exists(boxdir): os.makedirs(boxdir)
		plt.savefig(os.path.join(boxdir,'%d_%d_box.png'%(epoch,it)))
		atleastoneplotted = True

	if atleastoneplotted: plt.close('all')


def save_density_plt(params, epoch, it, logdir, tag):
	if len(params) == 0: return
	bplotdata = []
	bplotlabel = []
	ncol = 4
	f, axarr = plt.subplots(len(params)/ncol+1,ncol, figsize=(15, 10))
	#f, axarr = plt.subplots(len(params)/ncol+1,ncol)
	i, j = 0, 0
	for n,lt in params:
		l = lt.numpy()
		bplotdata.append(l.flatten())
		bplotlabel.append(n)

		try:
			# Matplotlib
			density = stats.kde.gaussian_kde(l.flatten())
			#x = np.linspace(l.min()*0.8, l.max()*0.8, 100)
			#x = np.linspace(-0.02, 0.02, 100)
			mean = l.mean()
			std = l.std()*1.5
			x = np.linspace(mean-std, mean+std, 100)
			y = density(x)
			axarr[i,j].plot(x, y)
			axarr[i,j].set_title(n)

			# Seaborn
			#sns.distplot(l.flatten(), rug=False, hist=False, norm_hist=True, ax=axarr[i,j])
			#sns.kdeplot(l.flatten())
		except Exception as ex:
			print(ex)

		j += 1
		if j%ncol == 0:
			j = 0
			i += 1
	plt.tight_layout()
	densedir = os.path.join(logdir,'dist',tag,'density')
	if not os.path.exists(densedir): os.makedirs(densedir)
	plt.savefig(os.path.join(densedir,'%d_%d_dens.png'%(epoch,it)))

	# Box plot
	plt.figure(figsize=(15, 10))
	plt.boxplot(bplotdata, labels=bplotlabel, sym='')
	args = plt.xticks()
	plt.xticks(*args,rotation=90)
	plt.grid(True)

	plt.tight_layout(rect=[0, 0, 0.85, 1.0])
	boxdir = os.path.join(logdir,'dist',tag,'box')
	if not os.path.exists(boxdir): os.makedirs(boxdir)
	plt.savefig(os.path.join(boxdir,'%d_%d_box.png'%(epoch,it)))

	plt.close('all')
