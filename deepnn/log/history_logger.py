import numpy as np
import os.path
import time

class HistorySaver(object):
	def __init__(self, logname, start_epoch=-1):
		self.histname = logname
		self.record = {}
		self.metadata = {}
		if start_epoch != -1:
			try:
				self.load()
				self._clear_after_epoch(start_epoch)
			except Exception as e:
				print(e)

	def load(self):
		d = np.load(self.histname+'.npz', allow_pickle=True)
		self.record = d['data'].item()
		if 'metadata' in d.files:
			self.metadata = d['metadata'].item()
		else:
			self.metadata = {}
		return self

	def _clear_after_epoch(self, epoch):
		for k in self.record:
			for e in self.record[k].keys():
				if e >= epoch: del self.record[k][e]

	def _add_record(self, epoch, k, v):
		if epoch not in self.record[k]:
			self.record[k][epoch] = []

		self.record[k][epoch].append(v)

	def add(self, epoch, history):
		for k in history.keys():
			if k not in self.record.keys():
				self.record[k] = {}
			self._add_record(epoch, k, history[k])

	def put_meta(self, key, value):
		self.metadata[key] = value

	def save(self):
		np.savez(self.histname, data=self.record, metadata=self.metadata)

	def get_record(self):
		return self.record

	def get_meta(self):
		return self.metadata
