import torch
from torch.autograd import Variable
from math import exp
import torch.nn.functional as F

"""
Source code is taken from https://github.com/Po-Hsun-Su/pytorch-ssim
and https://github.com/waleedgondal/pytorch-ssim

Original source is modified to make it compatible with the project
"""

"""
    Args:
        img1, img2 : Batch of images with dimension (batch_size, ch, h, w) or (batch_size, h, w)
        win_size : Gaussian filter size. It must be odd number
        K1, K2 : Parameters from the article http://www.cns.nyu.edu/~lcv/ssim/
        sigma : For gaussian filter
        L : Data range for images. Default 2.0 for tanh outputs [-1, 1] max-min=2
"""

import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def _ssim(img1, img2, window, window_size, channel, size_average = True):
    mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)
    mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean().item()
    else:
        return ssim_map.mean(1).mean(1).mean(1).item()

class SSIM(torch.nn.Module):
    def __init__(self, window_size = 11, size_average = True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, x1, x2):
        with torch.no_grad():
            img1 = x1
            img2 = x2
            (_, channel, _, _) = img1.size()

            if channel == self.channel and self.window.data.type() == img1.data.type():
                window = self.window
            else:
                window = create_window(self.window_size, channel)

                if img1.is_cuda:
                    window = window.cuda(img1.get_device())
                window = window.type_as(img1)

                self.window = window
                self.channel = channel


            return _ssim(img1, img2, window, self.window_size, channel, self.size_average)

def ssim(img1, img2, window_size = 11, size_average = True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.to(img1.device)
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

"""
class SSIM():
    def __init__(self, window_size=11, sigma=1.5, size_average=True, gpu_id=None):
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 3
        self.window = create_window(window_size, self.channel, sigma)
        self.gpu_id = None
        self.is_gpu_set = False
        self.sigma = sigma

    def cuda(self, device=None):
        self.is_gpu_set = True
        self.gpu_id = device
        self.window = self.window.cuda(self.gpu_id)

    def __call__(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel:
            window = self.window
            print 'flag1', type(window)
        else:
            window = create_window(self.window_size, channel, self.sigma)
            self.window = window
            self.channel = channel
            if self.is_gpu_set:
                self.window = self.window.cuda(self.gpu_id)
                window = self.window
                #self.cuda(self.gpu_id)
            print 'flag2', type(window.data)

        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)

def _ssim(x1, x2, window, window_size, channel, K1=0.01, K2=0.03, L=2.0, size_average=True):
    img1 = Variable(x1, volatile=True)
    img2 = Variable(x2, volatile=True)
    mu1 = F.conv2d(img1, window, padding = window_size/2, groups = channel)
    mu2 = F.conv2d(img2, window, padding = window_size/2, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding = window_size/2, groups = channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding = window_size/2, groups = channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding = window_size/2, groups = channel) - mu1_mu2

    C1 = (K1*L)**2
    C2 = (K2*L)**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean().data[0]
    else:
        return ssim_map.mean(1).mean(1).mean(1).data[0]


def create_window(window_size, channel, sigma):
    _1D_window = gaussian(window_size, sigma).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = to_variable(_2D_window.expand(channel, 1, window_size, window_size))
    return window

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size/2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def to_variable(x):
    #Convert tensor to variable.
    #if torch.cuda.is_available():
    #    x = x.cuda(self.gpu_id)
    return Variable(x)

def ssim(img1, img2, win_size=11, K1=0.01, K2=0.03, sigma=1.5, L=2.0):
    (_, channel, _, _) = img1.size()
    window = create_window(win_size, channel, sigma)
    return _ssim(img1, img2, window, win_size, channel, size_average=True)
"""
