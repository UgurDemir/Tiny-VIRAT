import torch
import torch.nn as nn
from torch.autograd import Variable

from os.path import join
import random
import time
import copy
import sys
import imageio

from deepnn.measure.psnr import PSNR
from deepnn.measure.ssim import SSIM
from deepnn.log.stats_logger import save_param_dict_density_plt
from deepnn.log.history_logger import HistorySaver
from deepnn.util.avg_meter import AverageMeter
from gan.generator import GeneratorWrapper
from deepnn.nets.network_wrapper import NetworkWrapper
from reader.loader import get_test_loader
import deepnn.util as utl

device = torch.device("cuda")
#device = torch.device("cpu")
#cpu = torch.device('cpu')
#gpu = torch.device('cuda')

# Confs
visrange = (-1.0, 1.0)

dataset = {
    'name': 'hmdb51', 
    #'name': 'ucf101',
    #'root': '/home/ugur/Documents/dataset/ucf101',
    'root': '/home/ugur/Documents/dataset/hmdb51',
    'split': {
        #'train_file': 'reader/ucf101/train.txt',
        #'test_file': 'reader/ucf101/test.txt',
        'train_file': 'reader/hmdb51/1/train.txt',
        'test_file': 'reader/hmdb51/1/test.txt',
    },
    'scale_size': 128, 
    'crop_size': 112,
    'nframe': 16,
    'scale_factor': 8,
    'upscale_mode': 'bicubic',
    'downscale_mode': 'bicubic',
    'produce': ['down', 'up', 'orig'],
    'workers': 4,
    'norm': '-11', # -11 | 01
    'batch_size': 18
}

EXPID = '76'
#CHCID = '259'

netG = {
    'gpu': 'cuda:0',

    'model': {
        #'arch': 'g4', 'ngf':32, 'nbf':16, 'nblock':4, 'nup':2, 'act':'lrelu', 'acto':None, 'norm':None,
        #'arch': 'g6', 'attn':None, 'ngf':32, 'nbf':16, 'nblock':4, 'nup':2, 'acto':'tanh', 'norm':'spec', 'conv_mode':'3D', #75
        #'arch': 'g6', 'attn':None, 'ngf':32, 'nbf':16, 'nblock':4, 'nup':2, 'acto':'tanh', 'norm':'spec', 'conv_mode':'3D', #79
        #'arch': 'g6', 'attn':'selfcorr3', 'ngf':32, 'nbf':16, 'nblock':4, 'nup':2, 'acto':'tanh', 'norm':'spec', 'conv_mode':'3D', #85
        #'arch': 'g6', 'attn':None, 'ngf':32, 'nbf':16, 'nblock':4, 'nup':2, 'acto':'tanh', 'norm':'spec', 'conv_mode':'3D', #68
        #'arch': 'g6', 'attn':None, 'ngf':32, 'nbf':16, 'nblock':4, 'nup':3, 'acto':'tanh', 'norm':'spec', 'conv_mode':'3D', #86
        'arch': 'g6', 'attn':None, 'ngf':32, 'nbf':16, 'nblock':4, 'nup':3, 'acto':'tanh', 'norm':'spec', 'conv_mode':'3D', #76
    },
    'resume': '/home/ugur/Desktop/tmp/img/weights/'+EXPID+'/chk_sr_best.pth.tar'
}

visrange = (-1.0, 1.0) if dataset['norm'] == '-11' else (0.0, 1.0)

def test1():
    root = '/home/ugur/Desktop/tmp/img/testout3/'

    test_loader = get_test_loader(**dataset)
    gen = GeneratorWrapper(name='G', **netG).to(device).eval()

    #batch = next(iter(eval_loader))
    for i, batch in enumerate(test_loader):
        ctx, up, gt = batch[0].to(device), batch[1].to(device), batch[2].to(device)
        b,c,t,h,w = gt.size()

        rec = gen(ctx)

        vmin, vmax = visrange
        if vmin != 0.0 or vmax != 1.0:
            ctx = (ctx - vmin) / (vmax - vmin)
            up = (up - vmin) / (vmax - vmin)
            gt = (gt - vmin) / (vmax - vmin)
            rec = (rec - vmin) / (vmax - vmin)


        for bid in range(b):
            create_gif(rec[bid,:,:,:,:], join(root, EXPID+'/movie{}_{}.gif'.format(i, bid)))
            
            create_gif(up[bid,:,:,:,:], join(root, 'hmdb_bicubic_x8/movie{}_{}.gif'.format(i, bid)))
            create_gif(gt[bid,:,:,:,:], join(root, 'hmdb_gt/movie{}_{}.gif'.format(i, bid)))
            #create_gif(ctx[bid,:,:,:,:], join(root, 'ctx_x4/movie{}_{}.gif'.format(i, bid)))

def ittir(loader, n):
    it = iter(loader)
    for _ in range(n): 
        out = next(it)
    return out

def create_gif(img, path):
    images = []
    c,t,h,w = img.size()
    for i in range(t):
        rec_np = img[:,i,:,:].permute(1,2,0).cpu().numpy()
        images.append(rec_np)
    imageio.mimsave(path, images)

if __name__ == "__main__":
    with torch.no_grad():
        test1()