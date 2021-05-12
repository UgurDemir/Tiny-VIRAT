# Some part of the code are taken from https://github.com/xinntao/ESRGAN/blob/master/block.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict

from deepnn.layer.conv3d import ConvST3d

"""
ESRGAN (winner of PRIM 2018) Residual in Residual Dense Block
Args:
    cnorm: normalization layer after convolution
    unorm: normalization layer after upconv layer
    actc: activation function for encoding step
    actd: activation function for decoding step
    scse: Skip connection from start to end [True|False]

    attn: [selfcorr | selfcorr2 | attncorr]
"""
class G6(nn.Module):
    def __init__(self, attn, ngf=64, nbf=32, nblock=5, nup=2, vdsr=False, conv_mode='3D', act='lrelu', acto='tanh', norm='spec'):
        super(G6, self).__init__()
        self.vdsr = vdsr

        self.convin = ConvST3d(3, ngf, (3,3,3), padding=(1,1,1), act=None, norm=None, conv_mode=conv_mode)
        self.res_hall = nn.Sequential(OrderedDict([('rrdb%d'%(i+1), RRDB(inch=ngf, nbf=nbf, norm=norm, act=act, conv_mode=conv_mode)) for i in range(nblock)]))
        self.convbout = ConvST3d(ngf, ngf, (3,3,3), padding=(1,1,1), act=None, norm=norm, conv_mode=conv_mode)

        up = [('upconv%d'%(i+1), ConvST3d(ngf, ngf, (3,3,3), padding=(1,1,1), scale_factor=[1,2,2], scale_mode='trilinear', act=act, norm=None, conv_mode=conv_mode)) for i in range(nup)]
        self.up_hall = nn.Sequential(OrderedDict(up))
        self.conv1 = ConvST3d(ngf, ngf, (3,3,3), padding=(1,1,1), act=act, norm=None, conv_mode=conv_mode)
        self.conv2 = ConvST3d(ngf, 3, (3,3,3), padding=(1,1,1), act=acto, norm=None, conv_mode=conv_mode)

        if attn is None:
            self.corr = None
        elif attn == 'selfcorr':
            self.corr = SelfCorrelation(ngf, c1=ngf//2)
        elif attn == 'selfcorr2':
            self.corr = SelfCorrelation2(ngf)
        elif attn == 'selfcorr3':
            self.corr = SelfCorrelation3(ngf)
        elif attn == 'selfcorr4':
            self.corr = SelfCorrelation4(ngf)
        elif attn == 'attncorr':
            self.corr = AttentionCorrelation(ngf, c1=ngf//2)
        else:
            raise Exception('Unknown attention type ' + str(attn))

    def forward(self, x):
        c1 = self.convin(x)
        y = self.res_hall(c1)
        y = self.convbout(y)
        y = y + c1

        if self.corr is not None:
            y = self.corr(y)

        y = self.up_hall(y)
        y = self.conv1(y)
        y = self.conv2(y)

        if self.vdsr:
            xup = nn.functional.interpolate(x, size=[y.size(-3), y.size(-2), y.size(-1)], mode='trilinear', align_corners=False)
            y = y + xup

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

class SelfCorrelation4(nn.Module):
    def __init__(self, cin):
        super(SelfCorrelation4, self).__init__()

        self.path1a = nn.Conv3d(cin, cin//4, kernel_size=(1,1,1))

        self.path2a = nn.Conv3d(cin, cin//4, kernel_size=(1,1,1))

        self.path3 = nn.Conv3d(cin, cin, kernel_size=(1,1,1))

        self.softmax = nn.Softmax(dim=-1)

    
    def forward(self, x):
        _, c, t, h, w = x.size()

        # Path 1
        p1 = self.path1a(x)
        _, c1, t1, h1, w1 = p1.size()
        Q = p1.view(-1, c1, t1*h1*w1)
        
        # Path 2
        p2 = self.path2a(x)
        _, c2, t2, h2, w2 = p2.size()
        K = p2.view(-1, c2, t2*h2*w2)

        # Attention
        attn = torch.bmm(K.permute(0, 2, 1), Q)
        attn = self.softmax(attn)

        # Path3
        p3 = self.path3(x)
        _, c3, t3, h3, w3 = p3.size()
        V = p3.view(-1, c3, t3*h3*w3)

        # Output
        o_raw = torch.bmm(V, attn)
        o = o_raw.view(-1, c, t, h, w)
        
        return o

class SelfCorrelation3(nn.Module):
    def __init__(self, cin):
        super(SelfCorrelation3, self).__init__()

        self.path1a = nn.Conv3d(cin, cin, kernel_size=(3,3,3), padding=(1,1,1), stride=(2,2,2))

        self.path2a = nn.Conv3d(cin, cin, kernel_size=(3,3,3), padding=(1,1,1), stride=(2,2,2))

        self.path3 = nn.Conv3d(cin, cin, kernel_size=(1,1,1))

        self.softmax = nn.Softmax(dim=-1)

    
    def forward(self, x):
        _, c, t, h, w = x.size()

        # Path 1
        p1 = self.path1a(x)
        _, c1, t1, h1, w1 = p1.size()
        Q = p1.view(-1, c1, t1*h1*w1)
        
        # Path 2
        p2 = self.path2a(x)
        _, c2, t2, h2, w2 = p2.size()
        K = p2.view(-1, c2, t2*h2*w2)

        # Attention
        attn = torch.bmm(Q, K.permute(0, 2, 1))
        attn = self.softmax(attn)

        # Path3
        p3 = self.path3(x)
        _, c3, t3, h3, w3 = p3.size()
        V = p3.view(-1, c3, t3*h3*w3)

        # Output
        o_raw = torch.bmm(attn, V)
        o = o_raw.view(-1, c, t, h, w)
        
        return o

class SelfCorrelation2(nn.Module):
    def __init__(self, cin):
        super(SelfCorrelation2, self).__init__()

        self.path1a = nn.Conv3d(cin, cin, kernel_size=(3,3,3), padding=(1,1,1), stride=(2,2,2))
        self.path1b = nn.Conv3d(cin, cin, kernel_size=(1,1,1))

        self.path2a = nn.Conv3d(cin, cin, kernel_size=(3,3,3), padding=(1,1,1), stride=(2,2,2))
        self.path2b = nn.Conv3d(cin, cin, kernel_size=(1,1,1))

        self.path3 = nn.Conv3d(cin, cin, kernel_size=(1,1,1))

        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=1)

    
    def forward(self, x):
        _, c, t, h, w = x.size()

        # Path 1
        p1a = self.path1a(x)
        p1 = self.path1b(p1a)
        _, c1, t1, h1, w1 = p1.size()
        Q = p1.view(-1, c1, t1*h1*w1)
        
        # Path 2
        p2a = self.path2a(x)
        p2 = self.path2b(p2a)
        _, c2, t2, h2, w2 = p2.size()
        K = p2.view(-1, c2, t2*h2*w2)

        # Attention
        attn = torch.bmm(Q, K.permute(0, 2, 1))
        attn = self.softmax(attn)

        # Path3
        p3 = self.path3(x)
        _, c3, t3, h3, w3 = p3.size()
        V = p3.view(-1, c3, t3*h3*w3)

        # Output
        o_raw = torch.bmm(attn, V)
        o = o_raw.view(-1, c, t, h, w)

        #print('\np1', p1.size())
        #print('Q', Q.size())
        #print('p2', p2.size())
        #print('K', K.size())
        #print('attn', attn.size())
        #print('p3', p3.size())
        #print('V', V.size())
        #print('o_raw', o_raw.size())
        #print('o', o.size())
        
        return x + self.gamma * o

class SelfCorrelation(nn.Module):
    def __init__(self, cin, c1):
        super(SelfCorrelation, self).__init__()

        self.path1a = nn.Conv3d(cin, c1, kernel_size=(1,1,1), stride=(1,1,1))
        self.path1b = nn.Conv3d(c1, 14*14*8, kernel_size=(1,1,1), stride=(2,2,2))
        self.correlation = nn.Conv1d(14*14*8, 14*14*8, kernel_size=1)

        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=1)

    
    def forward(self, x):
        # Path 1
        p1a = self.path1a(x)
        p1b = self.path1b(p1a)
        
        # Correlation
        _, c, t, h, w = p1b.size()
        p1 = p1b.view(-1, c, t*h*w)
        p1_t = p1.permute(0, 2, 1)
        corr_raw = self.correlation(p1_t)
        corr_in = corr_raw.unsqueeze(1)
        corr_up = F.interpolate(corr_in, size=[t*h*w*8], mode='bilinear', align_corners=False).squeeze(1)
        corr = self.softmax(corr_up)

        # Apply attention
        _, cx, tx, hx, wx = x.size()
        x_raw = x.view(-1, cx, tx*hx*wx)
        o_raw = torch.bmm(x_raw, corr)
        o = o_raw.view(-1, cx, tx, hx, wx)
        
        return x + self.gamma * o

class AttentionCorrelation(nn.Module):
    def __init__(self, cin, c1):
        super(AttentionCorrelation, self).__init__()

        self.path1a = nn.Conv3d(cin, c1, kernel_size=(1,1,1), stride=(1,1,1))
        self.path1b = nn.Conv3d(c1, c1, kernel_size=(1,1,1), stride=(2,2,2))
        self.correlation = nn.Conv1d(14*14*8, 14*14*8, kernel_size=1)

        self.path2a = nn.Conv3d(cin, c1, kernel_size=(1,1,1), stride=(1,1,1))
        self.path2b = nn.Conv3d(c1, c1, kernel_size=(1,1,1), stride=(2,2,2))

        self.path3a = nn.Conv3d(cin, c1, kernel_size=(1,1,1), stride=(1,1,1))
        self.path3b = nn.Conv3d(c1, cin, kernel_size=(1,1,1), stride=(2,2,2))

        self.upconv = nn.Conv3d(cin, cin, kernel_size=(3,3,3), padding=(1,1,1))

        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=1)

    
    def forward(self, x):
        # Path 1
        p1a = self.path1a(x)
        p1b = self.path1b(p1a)
        
        # Correlation
        _, c, t, h, w = p1b.size()
        p1 = p1b.view(-1, c, t*h*w)
        p1_t = p1.permute(0, 2, 1)
        corr = self.correlation(p1_t)

        # Path 2
        p2a = self.path2a(x)
        p2b = self.path2b(p2a)
        _, c2, t2, h2, w2 = p2b.size()
        p2 = p2b.view(-1, c2, t2*h2*w2)

        # Self Attention
        attn = torch.bmm(corr, p2)
        attn = self.softmax(attn)

        # Path3
        p3a = self.path3a(x)
        p3b = self.path3b(p3a)
        _, c3, t3, h3, w3 = p3b.size()
        p3 = p3b.view(-1, c3, t3*h3*w3)

        # Output
        out_raw = torch.bmm(p3, attn)
        out_down = out_raw.view(-1, c3, t3, h3, w3)
        out_int = F.interpolate(out_down, size=[t3*2, h3*2, w3*2], mode='trilinear', align_corners=False)
        out = self.upconv(out_int)
        
        return x + self.gamma * out