import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

from deepnn.layer.iconv  import InterpolatedConv2D, InterpolatedConv3D
from deepnn.layer.conv_st3d import ConvST3d

from .models.c3d_gan import C3DGAN

from .models.d1 import D1
from .models.d2 import D2
from .models.d3 import D3
from .models.d4 import D4
from .models.pd1 import PD1
from .models.pd2 import PD2
from .models.g1 import G1
from .models.g2 import G2
from .models.g3 import G3
from .models.g4 import G4
from .models.g5 import G5
from .models.g6 import G6
from .models.g8 import G8
from .models.pg1 import PG1
from .models.pg2 import PG2
from .models.pg3 import PG3
from .models.pg4 import PG4
from .models.pg5 import PG5
from .models.locpg2 import LOCPG2
from .models.locpg5 import LOCPG5

from .models.i3d import InceptionI3d
import deepnn.nets.models.wide_resnet as wide_resnet
import deepnn.nets.models.densenet as densenet
import deepnn.nets.models.resnet_act as resnet_act

def build_net(arch, **args):
	if arch == 'c3d_gan':
		n = C3DGAN
	elif arch == 'wglobal_d':
		raise Exception('Unimplemented discriminative network architecture. Available soon.')
	elif arch == 'wpatch_d':
		raise Exception('Unimplemented discriminative network architecture. Available soon.')
	elif arch == 'multi_d':
		raise Exception('Unimplemented discriminative network architecture. Available soon.')
	elif arch == 'local_d':
		raise Exception('Unimplemented discriminative network architecture. Available soon.')
	
	elif arch == 'd1':
		n = D1
	elif arch == 'd2':
		n = D2
	elif arch == 'd3':
		n = D3
	elif arch == 'd4':
		n = D4
	elif arch == 'pd1':
		n = PD1
	elif arch == 'pd2':
		n = PD2
	elif arch == 'g1':
		n = G1
	elif arch == 'g2':
		n = G2
	elif arch == 'g3':
		n = G3
	elif arch == 'g4':
		n = G4
	elif arch == 'g5':
		n = G5
	elif arch == 'g6':
		n = G6
	elif arch == 'g8':
		n = G8
	elif arch == 'pg1':
		n = PG1
	elif arch == 'pg2':
		n = PG2
	elif arch == 'pg3':
		n = PG3
	elif arch == 'pg4':
		n = PG4
	elif arch == 'pg5':
		n = PG5
	elif arch == 'locpg2':
		n = LOCPG2
	elif arch == 'locpg5':
		n = LOCPG5

	elif arch == 'i3d':
		n = InceptionI3d
	elif arch == 'wide_resnet':
		n = wide_resnet.resnet50
	elif arch.startswith('resnet'):
		depth = arch[6:]
		if depth == '10':
			n = resnet_act.resnet10
		elif depth == '18':
			n = resnet_act.resnet18
		elif depth == '34':
			n = resnet_act.resnet34
		elif depth == '50':
			n = resnet_act.resnet50
		elif depth == '101':
			n = resnet_act.resnet101
		elif depth == '152':
			n = resnet_act.resnet152
		elif depth == '200':
			n = resnet_act.resnet200
		else:
			raise Exception('Unknown ResNet depth')
	elif arch.startswith('densenet'):
		depth = arch[8:]
		if depth == '121':
			n = densenet.densenet121
		elif depth == '169':
			n = densenet.densenet169
		elif depth == '201':
			n = densenet.densenet201
		elif depth == '264':
			n = densenet.densenet264
		else:
			raise Exception('Unknown DenseNet depth')
	else:
		raise Exception('Unimplemented network architecture')

	return n(**args)


def build_optim(model_params, method, **args):
	return optim.__dict__[method](model_params, **args)

def build_scheduler(optimizer, method, **args):
    return lr_scheduler.__dict__[method](optimizer, **args)

def act(name):
	layer = nn.__dict__[name]
	args = {}

	if name == 'LeakyReLU':
		args['negative_slope'] = 0.2

	def f(**kwargs):
		kwargs.update(args)
		return layer(**kwargs)

	return f

def pad(name):
	l = nn.__dict__[name]
	def f(padding):
		return l(padding=padding)

	return f

def norm(norm_layer):
	l = nn.__dict__[norm_layer]
	def f(nf):
		return l(nf)

	return f

def deconv(name, iconv_args={}, tconv_args={}):
    if name == 'iconv':
        l = InterpolatedConv2D
        args = iconv_args
    elif name == 'tconv':
        l = nn.ConvTranspose2d
        args = tconv_args
    else: raise Exception(name + ' deconv layer is not implemented')

    def f(**kwargs):
        kwargs.update(args)
        return l(**kwargs)

    return f

def conv(name):
	if name == 'ConvST3d':
		l = ConvST3d
	elif name == 'InterpolatedConv3D':
		l = InterpolatedConv3D
	else:
		l = nn.__dict__[name]

	def f(**kwargs):
		return l(kwargs)

	return f