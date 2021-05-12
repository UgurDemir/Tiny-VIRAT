import numpy as np
import matplotlib.pyplot as plt
from deepnn.log.history_logger import HistorySaver
from collections import OrderedDict
import os
from os.path import join
import argparse

def plot_hist(logmap, **figs):
	ti = 1
	length = 0
	for t in figs.keys():
		if atleastOne(figs[t]['src'], logmap.keys()):
			length += 1

	fig = plt.figure(figsize=(10,length*4))
	for title in sorted(figs.keys()):
		if not atleastOne(figs[title]['src'],logmap.keys()): continue
		pltid = length * 100 + 10 + ti
		ti += 1
		plt.subplot(pltid)
		plt.title(title)
		plt.grid(True)
		for i, key in enumerate(figs[title]['src']):
			if key not in logmap.keys(): continue
			mode = figs[title]['mode'] if 'mode' in figs[title] else 'iter'
			step = figs[title]['step'] if 'step' in figs[title] else 50
			clr = figs[title]['c'][i] if 'c' in figs[title] else 'r-'
			lbl = figs[title]['label'][i] if 'label' in figs[title] else None
			if mode == 'epoch':
				x, y = get_xy_epoch(logmap[key])
			elif mode == 'iter':
				x, y = get_xy_raw(logmap[key])
			elif mode == 'step':
				x, y = get_xy_step(logmap[key], step)
			plt.plot(x, y, clr, lw=3.0, label=lbl)
			#plt.xlim(xmin=0)
			plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

	if length>0:
		plt.tight_layout(rect=[0, 0, 0.85, 1.0])
		return True
	else:
		return False

def get_xy_epoch(data_map):
	y = []
	x = []
	for e in sorted(data_map.keys()):
		y.append(np.array(data_map[e]).mean())
		x.append(e+1)
	return np.array(x), np.array(y)

def get_xy_raw(data_map):
	y = []
	x = []
	for e in sorted(data_map.keys()):
		y += data_map[e]
		dx = np.linspace(e, e+1, num=len(data_map[e]), endpoint=False).tolist()
		x += dx
	return np.array(x), np.array(y)

def get_xy_step(data_map, step):
	raise Exception('Not implemented')
	y = []
	x = []
	for e in sorted(data_map.keys()):
		ed = data_map[e]
		start = 0
		for j in range(step,len(ed),step):
			y.append(np.array(ed[start:j]).mean())
			start = j
		dx = np.linspace(e, e+1, num=len(range(step,len(ed),step)), endpoint=False).tolist()
		x += dx
	return np.array(x), np.array(y)

def atleastOne(taglist, keylist):
	for t in taglist:
		if t in keylist:
			return True
	return False

def contains(taglist, keylist):
	for t in taglist:
		if t not in keylist:
			return False
	return True