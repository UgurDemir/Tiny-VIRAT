import torch.nn as nn

from .extractor import FeatureExtractor
from ..nets.models.i3d import InceptionI3d


class I3DFeatureExtractor(FeatureExtractor):
    def __init__(self, layers, model_path):
        feats = InceptionI3d().model
        lid = {
            'conv1a':1,
            'conv2b':2, 'conv2c':3,
            'pool3a':4, 'incep3b':5,  'incep3c':6,
            'pool4a':7, 'incep4b':8,  'incep4c':9, 'incep4d':10,  'incep4e':11, 'incep4f':12,
            'pool5a':13, 'incep5b':14,  'incep5c':15,
            'avgpool':16, 'drop':17
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
            print(l[0])
            print(net_slice)
            print('---------------')
            start = l[1]


        super(I3DFeatureExtractor, self).__init__(slicenets)
        
