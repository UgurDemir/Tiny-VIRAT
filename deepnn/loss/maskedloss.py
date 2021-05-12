import torch
import torch.nn as nn
from ..loss import build as build_loss

class MaskedLoss(nn.Module):
    def __init__(self, loss='l2'):
        super(MaskedLoss, self).__init__()
        self.loss = build_loss(name=loss)

    # Loss is calculated where the mask is 1
    def forward(self, x, y, mask):
        xm = x * mask
        ym = y * mask
        l = self.loss(xm, ym)

        return l