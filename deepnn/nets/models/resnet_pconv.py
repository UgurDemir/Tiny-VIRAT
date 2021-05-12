import torch
import torch.nn as nn

from collections import OrderedDict

import deepnn.nets.builder as lbuild
from deepnn.layer.pconv import PartialConv2d as PConv

class PartialResBlock(nn.Module):
    def __init__(self, nfeat, pad, cact, norml):
        super(PartialResBlock, self).__init__()

        seq = [
            PConv(in_channels=nfeat, out_channels=nfeat, kernel_size=3, stride=1, padding=1),
            norml(nfeat),
            cact(),

            PConv(in_channels=nfeat, out_channels=nfeat, kernel_size=3, stride=1, padding=1),
            norml(nfeat),
        ]

        self.model = nn.ModuleList(seq)

    def forward(self, x, mask):
        y = x
        for l in self.model:
            if isinstance(l, PConv):
                y, mask = l(y, mask)
            else:
                y = l(y)

        return x + y, mask


class ResNetPartial(nn.Module):
    def __init__(self, n_downs, n_blocks, n_up, ngf=64, norm='bn', use_dropout=False, inch=3, outch=3,
                padding='reflect', uplayer='iconv', actc='lrelu', actd='lrelu', acto='tanh'):
        super(ResNetPartial, self).__init__()

        cact = lbuild.act(actc)
        dact = lbuild.act(actd)
        oact = lbuild.act(acto)

        pad = lbuild.pad(padding)
        norml = lbuild.norm(norm)

        if use_dropout: raise Exception('Dropout is not implemented yet')
        if uplayer != 'iconv': raise Exception('only iconv is implemented for uplayer')
        
        # Input layers
        seq = [
            PConv(in_channels=inch, out_channels=ngf, kernel_size=7, padding=3),
            norml(ngf),
            cact()
        ]

        # Downsampling layers
        nin = ngf
        for i in range(1, n_downs+1):
            nout = ngf*(2**i)
            seq += [
                PConv(in_channels=nin, out_channels=nout, kernel_size=3, stride=2, padding=1),
                norml(nout),
                cact()
            ]
            nin = nout

        # Residual blocks
        for i in range(n_blocks):
            seq += [
                PartialResBlock(nin, pad=pad, cact=cact, norml=norml)
            ]

        # Upsampling layers
        up_seq = []
        for i in range(1, n_up+1):
            nout = ngf*(2**(n_downs-i))
            up_seq += [
                #deconv(in_channels=nin, out_channels=nout),
                PConv(in_channels=nin, out_channels=nout, kernel_size=3, stride=1, padding=1),
                norml(nout),
                dact()
            ]
            nin = nout

        # Out layer
        out_seq = [
            PConv(in_channels=nin, out_channels=outch, kernel_size=7, padding=3),
            oact()
        ]

        self.down_model = nn.ModuleList(seq)
        self.up_model = nn.ModuleList(up_seq)
        self.out_model = nn.ModuleList(out_seq)

    def forward(self, x, mask):
        y = x
        i = 1
        # Down + ResBlock
        for l in self.down_model:
            if isinstance(l, PConv) or isinstance(l, PartialResBlock):
                y, mask = l(y, mask)
                i += 1
            else:
                y = l(y)

        # Up
        for l in self.up_model:
            if isinstance(l, PConv):
                y = nn.functional.interpolate(y, scale_factor=2, mode='nearest')
                mask = nn.functional.interpolate(mask, scale_factor=2, mode='nearest')
                y, mask = l(y, mask)
            else:
                y = l(y)

        # Out
        for l in self.out_model:
            if isinstance(l, PConv):
                y, mask = l(y, mask)
            else:
                y = l(y)

        return y
