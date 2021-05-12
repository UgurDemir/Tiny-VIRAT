# Self Attention Generator
import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm
import torch.nn.functional as F

from collections import OrderedDict

from deepnn.layer.conv3d import ConvST3d

class G5(nn.Module):
    """Generator."""

    def __init__(self, ngf=64, nbf=32, nblock=5, nup=2, attn_resblock_idx=None, attn_upblock_idx=None, attn_before_up=False, conv_mode='3D', act='lrelu', acto='tanh', norm='spec'):
        super(G5, self).__init__()
        self.attn_before_up = attn_before_up

        # Initial convolution
        self.convin = ConvST3d(3, ngf, (3,3,3), padding=(1,1,1), act=None, norm=norm, conv_mode=conv_mode)
        
        # RRDB blocks
        rh = []
        for i in range(nblock):
            rh.append(('rrdb%d'%(i+1), RRDB(inch=ngf, nbf=nbf, norm=norm, act=act, conv_mode=conv_mode)))
            if attn_resblock_idx is not None and attn_resblock_idx == i:
                rh.append(('rb_attn', SelfAttention(ngf)))
        self.res_hall = nn.Sequential(OrderedDict(rh))
        
        # Convolution btw residual blocks and upsampling
        self.convbout = ConvST3d(ngf, ngf, (3,3,3), padding=(1,1,1), act=None, norm=norm, conv_mode=conv_mode)

        if self.attn_before_up:
            self.attn = SelfAttention(ngf)

        # Upsampling layers
        up = []
        for i in range(nup):
            up.append(('upconv%d'%(i+1), ConvST3d(ngf, ngf, (3,3,3), padding=(1,1,1), scale_factor=[1,2,2], scale_mode='trilinear', act=act, norm=norm, conv_mode=conv_mode)))
            if attn_upblock_idx is not None and attn_upblock_idx == i:
                up.append(('up_attn', SelfAttention(ngf)))
        self.up_hall = nn.Sequential(OrderedDict(up))
        
        # Output convolutions
        self.conv1 = ConvST3d(ngf, ngf, (3,3,3), padding=(1,1,1), act=act, norm=norm, conv_mode=conv_mode)
        self.conv2 = ConvST3d(ngf, 3, (3,3,3), padding=(1,1,1), act=acto, norm=norm, conv_mode=conv_mode)

        self.apply(init_weights)

    def forward(self, x):
        c1 = self.convin(x)
        y = self.res_hall(c1)
        y = self.convbout(y)
        
        y = y + c1

        if self.attn_before_up:
            y = self.attn(y)

        y = self.up_hall(y)
        y = self.conv1(y)
        y = self.conv2(y)
        return y

def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d or type(m) == nn.Conv3d:
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.)


class ResidualDenseBlock(nn.Module):
    """
    Residual Dense Block
    style: 5 convs
    """

    def __init__(self, inch, outch, res_scale=0.2, kernel_size=(3,3,3), norm=None, act='lrelu', conv_mode='ST'):
        super(ResidualDenseBlock, self).__init__()
        self.res_scale = res_scale
        
        self.conv1 = ConvST3d(inch,           outch, kernel_size, padding=(1,1,1), act=act, norm=norm, conv_mode=conv_mode)
        self.conv2 = ConvST3d(inch + outch,   outch, kernel_size, padding=(1,1,1), act=act, norm=norm, conv_mode=conv_mode)
        self.conv3 = ConvST3d(inch + outch*2, outch, kernel_size, padding=(1,1,1), act=act, norm=norm, conv_mode=conv_mode)
        self.conv4 = ConvST3d(inch + outch*3, outch, kernel_size, padding=(1,1,1), act=act, norm=norm, conv_mode=conv_mode)
        self.conv5 = ConvST3d(inch+outch*4,   inch,  kernel_size, padding=(1,1,1), act=None, norm=norm, conv_mode=conv_mode)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(torch.cat((x, x1), 1))
        x3 = self.conv3(torch.cat((x, x1, x2), 1))
        x4 = self.conv4(torch.cat((x, x1, x2, x3), 1))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5.mul(self.res_scale) + x


class RRDB(nn.Module):
    """
    Residual in Residual Dense Block
    """

    def __init__(self, inch, nbf, res_scale=0.2, kernel_size=(3,3,3), norm=None, act='lrelu', conv_mode='ST'):
        super(RRDB, self).__init__()
        self.res_scale = res_scale

        self.RDB1 = ResidualDenseBlock(inch=inch, outch=nbf, res_scale=res_scale, kernel_size=kernel_size, norm=norm, act=act, conv_mode=conv_mode)
        self.RDB2 = ResidualDenseBlock(inch=inch, outch=nbf, res_scale=res_scale, kernel_size=kernel_size, norm=norm, act=act, conv_mode=conv_mode)
        self.RDB3 = ResidualDenseBlock(inch=inch, outch=nbf, res_scale=res_scale, kernel_size=kernel_size, norm=norm, act=act, conv_mode=conv_mode)

    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out.mul(self.res_scale) + x

"""Taken form https://github.com/voletiv/self-attention-GAN-pytorch/blob/master/sagan_models.py and modified"""
class SelfAttention(nn.Module):
    def __init__(self, inch, C1=4, C2=2, act='relu', norm='spec'):
        super(SelfAttention, self).__init__()
        self.C1 = C1
        self.C2 = C2

        self.f = ConvST3d(inch, inch//C1, (1,1,1), act=act, norm=norm, conv_mode='3D')
        self.g = ConvST3d(inch, inch//C1, (1,1,1), act=act, norm=norm, conv_mode='3D')
        self.h = ConvST3d(inch, inch//C2, (1,1,1), act=act, norm=norm, conv_mode='3D')
        self.cov_out = ConvST3d(inch//C2, inch, (1,1,1), act=act, norm=norm, conv_mode='3D')
        
        self.softmax  = nn.Softmax(dim=-1)
        self.maxpool = nn.MaxPool3d(kernel_size=2, stride=2, padding=0)

        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        _, ch, t, h, w = x.size()

        theta = self.f(x).view(-1, ch//self.C1, h*w*t)
        
        phi = self.g(x)
        phi = self.maxpool(phi).view(-1, ch//self.C1, h*w*t//8)

        g = self.h(x)
        g = self.maxpool(g).view(-1, ch//self.C2, h*w*t//8)

        attn = torch.bmm(theta.permute(0, 2, 1), phi)
        attn = self.softmax(attn)

        attn_g = torch.bmm(g, attn.permute(0,2,1))
        attn_g = attn_g.view(-1, ch//self.C2, t, h, w)
        attn_g = self.cov_out(attn_g)

        out = x + self.gamma * attn_g

        return out