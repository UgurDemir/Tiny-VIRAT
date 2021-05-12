# STATIC ENCODER + PROGRESSIVE DECODER
import torch
import torch.nn as nn

from collections import OrderedDict

from deepnn.layer.conv3d import ConvST3d

class PG5(nn.Module):
    """
    Args:s
        mode: [3D | ST] 3D: Increase spatial and temporal size together. ST: Increase spatial then temporal (seperately)
    """
    def __init__(self, nch=32, actenc='lrelu', actdec='lrelu', acto='tanh', norm='spec', conv_mode='3D'):
        super(PG5, self).__init__()
        
        self.encoder = nn.Sequential(*[
            RGB2FeatProjector1(nch, 1, stride=(1,1,1), act=actenc, acto=actenc, norm=norm, conv_mode=conv_mode),
            EncoderBlock1(nch, 2*nch),
            EncoderBlock1(2*nch, 4*nch),
            EncoderBlock1(4*nch, 8*nch),
        ])

        self.decoder_blocks = nn.ModuleList([
            DecoderBlock1(8*nch, 4*nch), # 0
            DecoderBlock1(4*nch, 2*nch), # 1
            DecoderBlock1(2*nch, nch)    # 2
        ])

        self.feat2rgb = nn.ModuleList([
            Feat2RGBProjector1(4*nch, 1, stride=(1,1,1), act=actenc, acto=acto, norm=norm, conv_mode=conv_mode),
            Feat2RGBProjector1(2*nch, 1, stride=(1,1,1), act=actenc, acto=acto, norm=norm, conv_mode=conv_mode),
            Feat2RGBProjector1(nch, 1, stride=(1,1,1), act=actenc, acto=acto, norm=norm, conv_mode=conv_mode)
        ])

    # Args:
    #   x: input low res image
    #   scale: 1: 2x | 2: 4x | 3: 8x
    #   fadein: fadein factor for smooth transition between consequtive scales
    #           out = current_rgb * (1-fadein) + previous_rgb * fadein
    def forward(self, x, scale=None, fadein=0.0):
        scale = scale if scale is not None else len(self.decoder_blocks)

        # Encoder
        e = self.encoder(x)

        # Decoder
        y = e
        for i in range(scale):
            y_prev = y
            y = self.decoder_blocks[i](y)
        y = self.feat2rgb[scale-1](y)

        if fadein > 0 and scale > 1:
            _,_, t, h, w = y.shape
            y_prev_rgb = self.feat2rgb[scale-2](y_prev)
            y_prev_up = nn.functional.interpolate(y_prev_rgb, size=[t,h,w], mode='trilinear', align_corners=False)
            y = y_prev_up * fadein + y * (1.0 - fadein)

        return y


class EncoderBlock1(nn.Module):
    def __init__(self, nin, nout, res_scale=0.2, act='lrelu', norm='bn', conv_mode='3D'):
        super(EncoderBlock1, self).__init__()
        #self.rrdb = RRDB(nin, nout, res_scale=0.2, kernel_size=(3,3,3), norm=None, act=act, conv_mode=conv_mode)
        #self.conv1 = ConvST3d(nin, nout, (3,3,3), stride=stride, padding=(1,1,1), act=act, norm=norm, conv_mode=conv_mode)
        kernel_size = (3,3,3)

        self.res_scale = res_scale
        self.RDB1 = ResidualDenseBlock(inch=nin, outch=nin, res_scale=res_scale, kernel_size=kernel_size, norm=norm, act=act, conv_mode=conv_mode)
        self.conv1 = ConvST3d(nin, nout, (3,3,3), stride=(2,2,2), padding=(1,1,1), act=act, norm=norm, conv_mode=conv_mode)
        self.RDB2 = ResidualDenseBlock(inch=nout, outch=nout, res_scale=res_scale, kernel_size=kernel_size, norm=norm, act=act, conv_mode=conv_mode)


    def forward(self, x):
        out = self.RDB1(x)
        out = self.conv1(out)
        #_,_,t,h,w = out.size()
        #o_down = torch.nn.functional.interpolate(out, size=[t,h,w], mode='trilinear', align_corners=False)
        out = self.RDB2(out)
        return out
        #return out.mul(self.res_scale) + o_down

class DecoderBlock1(nn.Module):
    def __init__(self, nin, nout, scale_mode='trilinear', scale_factor=(2,2,2), res_scale=0.2, act='lrelu', norm='bn', conv_mode='3D'):
        super(DecoderBlock1, self).__init__()
        self.RDB1 = ResidualDenseBlock(inch=nin, outch=nin, res_scale=res_scale, kernel_size=(3,3,3), norm=norm, act=act, conv_mode=conv_mode)
        self.conv1 = ConvST3d(nin, nout, (3,3,3), padding=(1,1,1), scale_mode=scale_mode, scale_factor=scale_factor, act=act, norm=norm, conv_mode=conv_mode)

    def forward(self, x):
        y = self.RDB1(x)
        y = self.conv1(y)
        return y

class RGB2FeatProjector1(nn.Module):
    def __init__(self, outch, nlayer, stride, act='lrelu', acto='tanh', norm=None, conv_mode='3D'):
        super(RGB2FeatProjector1, self).__init__()
        self.conv1 = ConvST3d(3, outch, kernel_size=(1,1,1), act=acto, norm=None, conv_mode='3D')
    
    def forward(self, x):
        y = self.conv1(x)
        return y

class Feat2RGBProjector1(nn.Module):
    def __init__(self, inch, nlayer, stride, act='lrelu', acto='tanh', norm=None, conv_mode='3D'):
        super(Feat2RGBProjector1, self).__init__()
        seq = []
        for i in range(nlayer):
            outch = inch#//(2**(i+1))
            seq += [('l%d'%i, ConvST3d(inch, outch, kernel_size=(3,3,3), stride=stride ,padding=(1,1,1), act=act, norm=norm, conv_mode=conv_mode))]
            inch = outch
        
        seq += [('l%d'%nlayer, ConvST3d(inch, 3, kernel_size=(1,1,1), act=acto, norm=None, conv_mode='3D'))]

        self.seq = nn.Sequential(OrderedDict(seq))

    def forward(self, x):
        return self.seq(x)


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
