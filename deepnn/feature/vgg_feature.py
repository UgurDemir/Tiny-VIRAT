from torchvision import models
from .extractor import FeatureExtractor
import torch.nn as nn

class VGGFeatureExtractor(FeatureExtractor):
	def __init__(self, layers):
		feats = models.vgg16(pretrained=True).features
		lid = {
			'conv_1_1':1, 'relu_1_1':2, 'conv_1_2':3, 'relu_1_2':4, 'pool_1':5,
			'conv_2_1':6, 'relu_2_1':7, 'conv_2_2':8, 'relu_2_2':9, 'pool_2':10,
			'conv_3_1':11, 'relu_3_1':12, 'conv_3_2':13, 'relu_3_2':14, 'conv_3_3':15, 'relu_3_3':16, 'pool_3':17,
			'conv_4_1':18, 'relu_4_1':19, 'conv_4_2':20, 'relu_4_2':21, 'conv_4_3':22, 'relu_4_3':23, 'pool_4':24,
			'conv_5_1':25, 'relu_5_1':26, 'conv_5_2':27, 'relu_5_2':28, 'conv_5_3':29, 'relu_5_3':30, 'pool_5':31
		}

		layer_id_list = []
		for l in layers:
			ind = lid[l]
			layer_id_list.append((l,ind))

		slicenets= []
		start = 0
		for l in sorted(layer_id_list, key=lambda x: x[1]):
			net_slice = nn.Sequential()
			for i in range(start, l[1]):
				net_slice.add_module(str(i), feats[i])
			slicenets.append((l[0], net_slice))
			start = l[1]

		super(VGGFeatureExtractor, self).__init__(slicenets)
