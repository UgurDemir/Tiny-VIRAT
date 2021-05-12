import torch
import torch.nn as nn

from collections import OrderedDict

import deepnn.nets.builder as lbuild

class ResBlock(nn.Module):
	def __init__(self, nfeat, pad, cact, norml):
		super(ResBlock, self).__init__()

		seq = [
			('r1_pad', pad(1)),
			('r1_conv', nn.Conv2d(in_channels=nfeat, out_channels=nfeat, kernel_size=3, stride=1)),
			('r1_norm', norml(nfeat)),
			('r1_act', cact()),

			('r2_pad', pad(1)),
			('r2_conv', nn.Conv2d(in_channels=nfeat, out_channels=nfeat, kernel_size=3, stride=1)),
			('r2_norm', norml(nfeat)),
		]

		self.model = nn.Sequential(OrderedDict(seq))

	def forward(self, x):
		return x + self.model(x)


class ResNet(nn.Module):
	def __init__(self, n_downs, n_blocks, n_up, ngf=64, norm='bn', use_dropout=False, inch=3, outch=3,
                padding='reflect', uplayer='iconv', actc='lrelu', actd='lrelu', acto='tanh'):
		super(ResNet, self).__init__()

		cact = lbuild.act(actc)
		dact = lbuild.act(actd)
		oact = lbuild.act(acto)

		pad = lbuild.pad(padding)
		norml = lbuild.norm(norm)

		deconv = lbuild.deconv(uplayer,
			iconv_args={'kernel_size':3, 'padding':1},
			tconv_args={'kernel_size':3, 'stride':2, 'padding':1, 'output_padding':1})

		if use_dropout: raise Exception('Dropout is not implemented yet')

		# Input layers
		seq = [
			('in_pad', pad(3)),
			('in_conv', nn.Conv2d(in_channels=inch, out_channels=ngf, kernel_size=7)),
			('in_norm', norml(ngf)),
			('in_act', cact())
		]

		# Downsampling layers
		nin = ngf
		for i in range(1, n_downs+1):
			nout = ngf*(2**i)
			seq += [
				('down%s_conv'%str(i), nn.Conv2d(in_channels=nin, out_channels=nout, kernel_size=3, stride=2, padding=1)),
				('down%s_norm'%str(i), norml(nout)),
				('down%s_act'%str(i), cact())
			]
			nin = nout

		# Residual blocks
		for i in range(n_blocks):
			seq += [
				('resblock%s'%str(i+1), ResBlock(nin, pad=pad, cact=cact, norml=norml))
			]

		# Upsampling layers
		for i in range(1, n_up+1):
			nout = ngf*(2**(n_downs-i))
			seq += [
				('up%s_deconv'%i, deconv(in_channels=nin, out_channels=nout)),
				('up%s_norm'%i, norml(nout)),
				('up%s_act'%i, dact())
			]
			nin = nout

		# Out layer
		seq += [
			('out_pad', pad(3)),
			('out_conv', nn.Conv2d(in_channels=nin, out_channels=outch, kernel_size=7)),
			('out_act', oact())
		]

		self.model = nn.Sequential(OrderedDict(seq))

	def forward(self, x):
		return self.model(x)
