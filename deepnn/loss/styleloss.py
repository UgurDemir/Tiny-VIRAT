import torch
import torch.nn as nn
from ..feature.vgg_feature import VGGFeatureExtractor
from ..feature.i3d_feature import I3DFeatureExtractor
from ..loss import build as build_loss
import deepnn.loss.gramloss as gramloss

class StyleLoss(nn.Module):
    def __init__(self, wt_feat, wt_gram, feat_layers=[], gram_layers=[], model='vgg16', loss='l2', **kwargs):
        super(StyleLoss, self).__init__()
        self.model_name = model
        self.feat_layers = feat_layers
        self.gram_layers = gram_layers
        self.wt_feat = wt_feat
        self.wt_gram = wt_gram

        layers = list(set(feat_layers + gram_layers))
        if model == 'vgg16':
            self.model = VGGFeatureExtractor([l[0] for l in layers], **kwargs)
        if model == 'i3d':
            self.model = I3DFeatureExtractor([l[0] for l in layers], **kwargs)
        else:
            raise Exception('Model is not supported')

        self.loss = build_loss(name=loss)

    def forward(self, x, y):
        # Feature Loss
        f1, f2 = self.model(x), self.model(y)
        feat_loss = 0.0
        for l, wt in self.feat_layers:
            feat_loss += self.loss(f1[l], f2[l]) * wt

        # Gram Loss
        gram_loss = 0.0     
        for l, wt in self.gram_layers:
            g1, g2 = gramloss.gram_matrix(f1[l]), gramloss.gram_matrix(f2[l])
            gram_loss += self.loss(g1, g2)

        total = feat_loss * self.wt_feat + gram_loss * self.wt_gram

        return total
