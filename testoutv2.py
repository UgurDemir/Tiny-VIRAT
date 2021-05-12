import torch
import torch.nn as nn
from torch.autograd import Variable

from os.path import join
import random
import time
import copy
import sys
import imageio
import argparse

from deepnn.measure.psnr import PSNR
from deepnn.measure.ssim import SSIM
from deepnn.log.stats_logger import save_param_dict_density_plt
from deepnn.log.history_logger import HistorySaver
from deepnn.util.avg_meter import AverageMeter
from gan.generator import GeneratorWrapper
from deepnn.nets.network_wrapper import NetworkWrapper
from reader.loader import get_test_loader
import deepnn.util as utl
import conf

def create_gifs(cluster_name, exp_name, exp_root_dir, model_file_pf, out_dir, 
                cuda=False, save_rec=True, save_orig=False, save_up=False, save_down=False):
    # Configurations
    opt = conf.parse(cluster_name, exp_name)
    dataset = opt['dataset']
    dataset['produce'] = ['down', 'up', 'orig']
    dataset['batch_size'] = 28
    netG = {k:v for k, v in opt['gan1']['netG'].items() if k in ['gpu','model','resume']}
    netG.update({'resume': join(exp_root_dir, exp_name, '_results/model/chk_sr_{}.pth.tar'.format(model_file_pf))})
    visrange = (-1.0, 1.0) if dataset['norm'] == '-11' else (0.0, 1.0)
    device = torch.device('cuda') if cuda else torch.devide('cpu')

    test_loader = get_test_loader(**dataset)
    gen = GeneratorWrapper(name='G', **netG).to(device).eval()

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

        # create gif
        for bid in range(b):
            if save_rec:
                recdir = join(out_dir, dataset['name'], 'x'+str(dataset['scale_factor']), exp_name)
                utl.mkdirs(recdir)
                create_gif(rec[bid,:,:,:,:], join(recdir, 'movie{}_{}.gif'.format(i, bid)))

            if save_orig:
                origdir = join(out_dir, dataset['name'], 'orig')
                utl.mkdirs(origdir)
                create_gif(gt[bid,:,:,:,:], join(origdir, 'movie{}_{}.gif'.format(i, bid)))

            if save_down:
                downdir = join(out_dir, dataset['name'], 'x'+str(dataset['scale_factor']), 'down')
                utl.mkdirs(downdir)
                create_gif(ctx[bid,:,:,:,:], join(downdir, 'movie{}_{}.gif'.format(i, bid)))

            if save_up:
                updir = join(out_dir, dataset['name'], 'x'+str(dataset['scale_factor']), 'up')
                utl.mkdirs(updir)
                create_gif(up[bid,:,:,:,:], join(updir, 'movie{}_{}.gif'.format(i, bid)))
            


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
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', type=str, default="/home/ugur/Documents/exps/vidres", help='Experiment root folder')
    parser.add_argument('-o', type=str, required=True, help='Output directory for generated gifs')
    parser.add_argument('-e', type=str, required=True, help='Experiment name')
    parser.add_argument('--chk', type=str, default='best', help='Model checkpoint file name postfix')
    parser.add_argument('-m', choices=['crcv', 'newton', 'desktop'], required=True, help='Machine name: [crcv|newton|desktop]')
    parser.add_argument('--down', action='store_true', default=False, help='Save downsampled image')
    parser.add_argument('--up', action='store_true', default=False, help='Save upsampled image')
    parser.add_argument('--orig', action='store_true', default=False, help='Save original image')
    parser.add_argument('--rec', action='store_true', default=True, help='Save reconstructed image')
    parser.add_argument('--gpu', action='store_true', default=True, help='GPU usage flag')
    argv = parser.parse_args()

    print(argv)

    with torch.no_grad():
        create_gifs(cluster_name=argv.m, exp_name=argv.e, exp_root_dir=argv.r, model_file_pf=argv.chk, out_dir=argv.o,
                    cuda=argv.gpu, save_rec=argv.rec, save_orig=argv.orig, save_up=argv.up, save_down=argv.down)

    