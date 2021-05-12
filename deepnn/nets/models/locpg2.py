# PROGRESSIVE ENCODER + PROGRESSIVE DECODER
import torch
import torch.nn as nn

from collections import OrderedDict

from deepnn.layer.conv3d import ConvST3d

class LOCPG2(nn.Module):
    """
    Args:
        mode: [3D | ST] 3D: Increase spatial and temporal size together. ST: Increase spatial then temporal (seperately)
    """
    def __init__(self, mode, scale_factor, ngf=32, nbf=32, actenc='lrelu', actdec='lrelu', acto='tanh', norm='spec', conv_mode='3D'):
        super(LOCPG2, self).__init__()
        assert mode in ['3D', 'ST']
        assert scale_factor in [4 , 8]
        
        self.convin = ConvST3d(3, ngf, (3,3,3), padding=(1,1,1), act=None, norm=None, conv_mode=conv_mode)

        blocks = [
            RRDB(inch=ngf, nbf=nbf, norm=norm, act=actenc, conv_mode=conv_mode),
            RRDB(inch=ngf, nbf=nbf, norm=norm, act=actenc, conv_mode=conv_mode),
            RRDB(inch=ngf, nbf=nbf, norm=norm, act=actenc, conv_mode=conv_mode),
        ]
        if scale_factor == 4:
            blocks = blocks[1:]

        if mode == '3D':
            self.enc_dec_ratio = 1
            ups = [
                nn.Sequential(
                    ConvST3d(nbf, nbf, kernel_size=(3,3,3), padding=(1,1,1), stride=(2,1,1), act=actdec, norm=norm, conv_mode=conv_mode),
                    ConvST3d(nbf, nbf, kernel_size=(3,3,3), padding=(1,1,1), stride=(2,1,1), act=actdec, norm=norm, conv_mode=conv_mode),
                    UpBlock(nbf, nbf, kernel=(3,3,3), padding=(1,1,1), scale_factor=[1,2,2], act=actdec, norm=norm, conv_mode=conv_mode)
                ),
                nn.Sequential(
                    ConvST3d(nbf, nbf, kernel_size=(3,3,3), padding=(1,1,1), stride=(2,1,1), act=actdec, norm=norm, conv_mode=conv_mode),
                    UpBlock(nbf, nbf, kernel=(3,3,3), padding=(1,1,1), scale_factor=[1,2,2], act=actdec, norm=norm, conv_mode=conv_mode)
                ),
                UpBlock(nbf, nbf, kernel=(3,3,3), padding=(1,1,1), scale_factor=[1,2,2], act=actdec, norm=norm, conv_mode=conv_mode),
            ]

            proj = [
                RGBBlock(ngf, 1, stride=(1,1,1), act=actenc, acto=acto, norm=norm, conv_mode=conv_mode),
                RGBBlock(ngf, 1, stride=(1,1,1), act=actenc, acto=acto, norm=norm, conv_mode=conv_mode),
                RGBBlock(ngf, 1, stride=(1,1,1), act=actenc, acto=acto, norm=norm, conv_mode=conv_mode)
            ]

            if scale_factor == 4:
                ups, proj = ups[1:], proj[1:]

        elif mode == 'ST':
            self.enc_dec_ratio = 2
            ups = [
                nn.Sequential(
                    ConvST3d(nbf, nbf, kernel_size=(3,3,3), padding=(1,1,1), stride=(2,1,1), act=actdec, norm=norm, conv_mode=conv_mode),
                    ConvST3d(nbf, nbf, kernel_size=(3,3,3), padding=(1,1,1), stride=(2,1,1), act=actdec, norm=norm, conv_mode=conv_mode)
                ),
                nn.Sequential(
                    ConvST3d(nbf, nbf, kernel_size=(3,3,3), padding=(1,1,1), stride=(2,1,1), act=actdec, norm=norm, conv_mode=conv_mode),
                    ConvST3d(nbf, nbf, kernel_size=(3,3,3), padding=(1,1,1), stride=(2,1,1), act=actdec, norm=norm, conv_mode=conv_mode),
                    UpBlock(nbf, nbf, kernel=(3,3,3), padding=(1,1,1), scale_factor=[1,2,2], act=actdec, norm=norm, conv_mode=conv_mode)
                ),
                nn.Sequential(
                    ConvST3d(nbf, nbf, kernel_size=(3,3,3), padding=(1,1,1), stride=(2,1,1), act=actdec, norm=norm, conv_mode=conv_mode)
                ),
                nn.Sequential(
                    ConvST3d(nbf, nbf, kernel_size=(3,3,3), padding=(1,1,1), stride=(2,1,1), act=actdec, norm=norm, conv_mode=conv_mode),
                    UpBlock(nbf, nbf, kernel=(3,3,3), padding=(1,1,1), scale_factor=[1,2,2], act=actdec, norm=norm, conv_mode=conv_mode)
                ),
                ConvST3d(nbf, nbf, kernel_size=(3,3,3), padding=(1,1,1), stride=(1,1,1), act=actdec, norm=norm, conv_mode=conv_mode),
                UpBlock(nbf, nbf, kernel=(3,3,3), padding=(1,1,1), scale_factor=[1,2,2], act=actdec, norm=norm, conv_mode=conv_mode)
            ]

            proj = [
                RGBBlock(ngf, 1, stride=(1,1,1), act=actenc, acto=acto, norm=norm, conv_mode=conv_mode),
                RGBBlock(ngf, 1, stride=(1,1,1), act=actenc, acto=acto, norm=norm, conv_mode=conv_mode),
                RGBBlock(ngf, 1, stride=(1,1,1), act=actenc, acto=acto, norm=norm, conv_mode=conv_mode),
                RGBBlock(ngf, 1, stride=(1,1,1), act=actenc, acto=acto, norm=norm, conv_mode=conv_mode),
                RGBBlock(ngf, 1, stride=(1,1,1), act=actenc, acto=acto, norm=norm, conv_mode=conv_mode),
                RGBBlock(ngf, 1, stride=(1,1,1), act=actenc, acto=acto, norm=norm, conv_mode=conv_mode)
            ]

            if scale_factor == 4:
                ups, proj = ups[2:], proj[2:]


        self.res_hall = nn.ModuleList(blocks)
        self.up_layers = nn.ModuleList(ups)
        self.rgb_proj = nn.ModuleList(proj)

        if scale_factor == 8:
            self.saliency_dec = nn.Sequential(
                ConvST3d(nbf, nbf//2, (3,3,3), padding=(1,1,1), scale_mode='trilinear', scale_factor=(1,2,2), act=actdec, norm='bn', conv_mode=conv_mode),
                ConvST3d(nbf//2, nbf//4, (3,3,3), padding=(1,1,1), scale_mode='trilinear', scale_factor=(1,2,2), act=actdec, norm='bn', conv_mode=conv_mode),
                ConvST3d(nbf//4, 1, (3,3,3), padding=(1,1,1), scale_mode='trilinear', scale_factor=(1,2,2), act='sigmoid', norm=None, conv_mode=conv_mode)
            )
        elif scale_factor == 4:
            self.saliency_dec = nn.Sequential(
                ConvST3d(nbf, nbf//4, (3,3,3), padding=(1,1,1), scale_mode='trilinear', scale_factor=(1,2,2), act=actdec, norm='bn', conv_mode=conv_mode),
                ConvST3d(nbf//4, 1, (3,3,3), padding=(1,1,1), scale_mode='trilinear', scale_factor=(1,2,2), act='sigmoid', norm=None, conv_mode=conv_mode)
            )

    # Args:
    #   x: input low res image
    #   scale: 1: 2x | 2: 4x | 3: 8x
    #   fadein: fadein factor for smooth transition between consequtive scales
    #           out = current_rgb * (1-fadein) + previous_rgb * fadein

    def forward(self, x, scale=None, fadein=0.0):
        scale = scale if scale is not None else len(self.up_layers)

        # Initial conv block to increase number of channel
        c1 = self.convin(x)
        
        # Residual Hall
        y = self.encoder(c1, scale=scale, fadein=fadein)

        # Saliency
        sal = self.saliency_dec(y)

        # LR Residual
        y = c1 + y

        # Upsampling
        y = self.decoder(y, scale=scale, fadein=fadein)

        # HR Residual (from VDSR)
        xup = nn.functional.interpolate(x, size=[y.size(-3), y.size(-2), y.size(-1)], mode='trilinear', align_corners=False)
        y = y + xup

        
        return y, sal

    def encoder(self, x, scale, fadein):
        y = x
        # Residual hall
        for i in range(scale):
            idx = i//self.enc_dec_ratio
            feat_prev = y
            y = self.res_hall[idx](y)

        # Fadein for residual blocks
        if fadein > 0.0:
            assert scale > 1
            y = feat_prev * fadein + y * (1.0 - fadein)

        return y

    def decoder(self, x, scale, fadein):
        y = x
        # Upsampling layers
        for i in range(scale):
            feat_prev = y
            if i > 0:
                t, h, w = x.size(-3), y.size(-2), y.size(-1)
                xup = nn.functional.interpolate(x, size=[t,h,w], mode='trilinear', align_corners=False)
                yup = nn.functional.interpolate(y, size=[t,h,w], mode='trilinear', align_corners=False)
                y = yup+xup #torch.cat([y, xup], dim=2)
            
            y = self.up_layers[i](y)

        # RGB Projection
        y = self.rgb_proj[scale-1](y)

        # Fadein for upsampling block
        if fadein > 0.0:
            assert scale > 1
            y_prev = self.rgb_proj[scale-2](feat_prev)
            y_prev_up = nn.functional.interpolate(y_prev, size=[y.size(-3), y.size(-2), y.size(-1)], mode='trilinear', align_corners=False)
            y = y_prev_up * fadein + y * (1.0 - fadein)
        
        return y

class RGBBlock(nn.Module):
    def __init__(self, inch, nlayer, stride, act='lrelu', acto='tanh', norm=None, conv_mode='3D'):
        super(RGBBlock, self).__init__()
        seq = []
        for i in range(nlayer):
            outch = inch#//(2**(i+1))
            seq += [('l%d'%i, ConvST3d(inch, outch, kernel_size=(3,3,3), stride=stride ,padding=(1,1,1), act=act, norm=norm, conv_mode=conv_mode))]
            inch = outch
        
        seq += [('l%d'%nlayer, ConvST3d(inch, 3, kernel_size=(1,1,1), act=acto, norm=None, conv_mode='3D'))]

        self.seq = nn.Sequential(OrderedDict(seq))

    def forward(self, x):
        return self.seq(x)


class UpBlock(nn.Module):
    def __init__(self, nin, nout, kernel, scale_factor, padding=(0,0,0), stride=(1,1,1), act='lrelu', norm='spec', conv_mode='3D'):
        super(UpBlock, self).__init__()
        self.c1 = ConvST3d(nin, nout, kernel, padding=padding, act=act, norm=norm, conv_mode=conv_mode)
        self.c2 = ConvST3d(nin, nout, kernel, padding=padding, scale_factor=scale_factor, stride=stride, scale_mode='trilinear', act=act, norm=norm, conv_mode=conv_mode)
        self.c3 = ConvST3d(nin, nout, kernel, padding=padding, act=act, norm=norm, conv_mode=conv_mode)

    def forward(self, x):
        y = self.c1(x)
        y = self.c2(y)
        y = self.c3(y)
        return y

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
