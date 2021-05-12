import torch
import torch.nn as nn

import os
from os.path import join
from functools import reduce

from .builder import build_net, build_optim, build_scheduler

class NetworkWrapper(nn.Module):
	def __init__(self, name, model, optim=None, lr_schedule=None, gpu=None, parallel_gpu=None, resume=None, load=None, **kwargs):
		super(NetworkWrapper, self).__init__()
		self.name = name
		self.gpu = gpu
		self.parallel_gpu = parallel_gpu

		self.model = build_net(**model)
		self.to(torch.device(self.gpu))
		
		self.optimizer = build_optim(self.model.parameters(), **optim) if optim is not None else None
		self.scheduler = build_scheduler(self.optimizer, **lr_schedule) if lr_schedule is not None else None
		
		if load is not None:
			self.load(load, only_model=True)

		if resume is not None:
			self.load(resume)

	def forward(self, *x, **kwargs):
		if self.parallel_gpu is not None:
			return nn.parallel.data_parallel(self.model, x, self.parallel_gpu, self.gpu)
		else:
			return self.model(*x, **kwargs)

	def scheduler_step(self):
		self.scheduler.step()

	def get_lr(self):
		return {'lr_'+self.name: self.scheduler.get_lr()[0]}

	def state_dict(self):
		sd = {'model': self.model.state_dict()}
		if self.optimizer is not None:
			sd['optimizer'] = self.optimizer.state_dict()
		return sd

	def load_state_dict(self, d):
		self.model.load_state_dict(d['model'])
		if 'optimizer' in d and self.optimizer is not None:
			self.optimizer.load_state_dict(d['optimizer'])

	def load(self, filename, only_model=False):
		if not os.path.isfile(filename): raise Exception('File not found {}'.format(filename))
		print("=> loading checkpoint '{}'".format(filename))
		checkpoint = torch.load(filename) # TODO: Add gpu enabled check
		if only_model:
			self.model.load_state_dict(checkpoint[self.name]['model'], strict=False)
		else:
			self.load_state_dict(checkpoint[self.name])

	def save(self, filename='checkpoint.pth.tar', extras={}):
		state = {}
		state[self.name] = self.state_dict()
		state.update(extras)
		torch.save(state, filename)

	def disable_grads(self):
		for p in self.parameters():
			p.requires_grad = False

	def enable_grads(self):
		for p in self.parameters():
			p.requires_grad = True

	def get_parameters(self, weights=True, bias=False, grads=False, conv=True, norm=False):
		params = []
		for n, l in self.model.named_parameters():
			nsplit = n.split('.')
			wb = nsplit[-1]
			ltype = nsplit[-2]

			if wb == 'bias' and not bias: continue
			#if wb == 'weight' and not weights: continue
			if ltype == 'conv' and not conv: continue
			if ltype == 'norm' and not norm: continue

			lname = reduce(lambda x, y: x+'_'+y, nsplit[:-1])
			lname = lname.replace('_model','').replace('model_','').replace('model','')

			if weights: params.append((lname, l.data.cpu()))
			if grads and l.grad is not None: params.append((lname+'_grad', l.grad.data.cpu()))

		return params

	def significant_parameters(self):
		return self.get_parameters(weights=True)

	def significant_grads(self):
		return self.get_parameters(weights=False, grads=True)
