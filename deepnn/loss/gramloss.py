import torch
import torch.nn as nn
from ..feature.vgg_feature import VGGFeatureExtractor
from ..feature.i3d_feature import I3DFeatureExtractor
from ..loss import build as build_loss

class GramLoss(nn.Module):
    def __init__(self, layers=[], model='vgg16', loss='l2', **kwargs):
        super(GramLoss, self).__init__()
        self.layers = layers
        self.model_name = model

        if model == 'vgg16':
            self.model = VGGFeatureExtractor([l[0] for l in self.layers], **kwargs)
        if model == 'i3d':
            self.model = I3DFeatureExtractor([l[0] for l in self.layers], **kwargs)
        else:
            raise Exception('Model is not supported')

        self.loss = build_loss(name=loss)

    def forward(self, x, y):
        f1 = self.model(x)
        f2 = self.model(y)

        total = 0
        for l, wt in self.layers:
            g1 = gram_matrix(f1[l])
            g2 = gram_matrix(f2[l])
            total += self.loss(g1, g2) * wt
            del f1[l], f2[l]
        return total

def gram_matrix(feat):
    # https://github.com/pytorch/examples/blob/master/fast_neural_style/neural_style/utils.py
    if feat.dim() == 4:
        (b, ch, h, w) = feat.size()
        feat = feat.view(b, ch, h * w)
        numel = ch * h * w
    elif feat.dim() == 5:
        (b, ch, t, h, w) = feat.size()
        feat = feat.view(b, ch, t * h * w)
        numel = ch * t * h * w
    feat_t = feat.transpose(1, 2)
    gram = torch.bmm(feat, feat_t) / numel
    return gram
