import torch
import torch.nn as nn
from torch.autograd import Variable

from os.path import join
import random
import time
import copy
import sys

from deepnn.measure.psnr import PSNR
from deepnn.measure.ssim import SSIM
from deepnn.log.stats_logger import save_param_dict_density_plt
from deepnn.log.history_logger import HistorySaver
from deepnn.util.avg_meter import AverageMeter
from gan.generator import GeneratorWrapper
from gan.discriminator import DiscriminatorWrapper
from deepnn.nets.network_wrapper import NetworkWrapper
from reader.loader import get_train_loader, get_val_loader
import deepnn.util as utl
import conf

# READ experiment protocol
if len(sys.argv) != 3:
    raise Exception('Please see usage: python main.py <machine_name> <exp_id>')
machine_name, exp_name = sys.argv[1], sys.argv[2]
#exp_name = '144'

opt = conf.parse(machine_name, exp_name)

device = torch.device("cuda" if opt['cuda'] else "cpu")
cpu = torch.device('cpu')
gpu = torch.device('cuda')
visrange = (-1.0, 1.0) if opt['dataset']['norm'] == '-11' else (0.0, 1.0)

# Measurements
psnr = PSNR(peak=1.0)
ssim = SSIM()

# Logging and Visualization
se = opt['start_epoch'] if opt['start_epoch'] > 0 else -1
logger = HistorySaver(join(opt['logdir'], 'hist'), se)
logger.get_meta().update(copy.deepcopy(opt))


def main():
    train_loader = get_train_loader(**opt['dataset'])
    eval_loader = get_val_loader(**opt['dataset'])

    gen = GeneratorWrapper(name='G', **opt['gan1']['netG']).to(device)
    disc = DiscriminatorWrapper(name='D', **opt['gan1']['netD']).to(device) if 'netD' in opt['gan1'] else None
    print('G', gen)
    print('D', disc)

    # TRAINING
    best_psnr_ssim = 0.0
    for epoch in range(opt['start_epoch'], opt['epochs']):
        train_psnr, train_ssim = train(gen, disc, train_loader, epoch)
        val_psnr, val_ssim =  validate(gen, disc, eval_loader, epoch)

        if (epoch+1)%opt['save_freq'] == 0:
            save(gen, disc, join(opt['modeldir'],"chk_sr_%s.pth.tar"%epoch),
                      {'epoch': epoch, 'manuelSeed': opt['manual_seed'],
                      'train_psnr': train_psnr, 'train_ssim':train_ssim, 'val_psnr': val_psnr, 'val_ssim':val_ssim})

        if (val_psnr + val_ssim) > best_psnr_ssim:
            best_psnr_ssim = val_psnr + val_ssim
            save(gen, disc, join(opt['modeldir'],"chk_sr_best.pth.tar"),
                      {'epoch': epoch, 'manuelSeed': opt['manual_seed'], 
                      'train_psnr': train_psnr, 'train_ssim':train_ssim, 'val_psnr': val_psnr, 'val_ssim':val_ssim})


def save(netG, netD, filename, extras={}):
    state = {}
    state[netG.name] = netG.state_dict()
    if netD is not None:
        state[netD.name] = netD.state_dict()
    state.update(extras)
    torch.save(state, filename)

def train(gen, disc, loader, epoch):
    gen.train()
    if disc is not None: disc.train()
    
    psnr_meter, ssim_meter, duration_meter = AverageMeter(), AverageMeter(), AverageMeter()
    for i, batch in enumerate(loader):
        start_time = time.time()

        ctx, gt = batch[0].to(device), batch[1].to(device)

        # Forward G
        rec = gen(ctx)

        logmap = {}
        if disc is not None:
            #print('Disc')
            # Forward D
            pred_fake = disc(rec.detach())
            pred_real = disc(gt)
            # Backward D
            disc.optimizer.zero_grad()
            dloss, dinfo = disc.calc_loss(pred_fake=pred_fake, pred_real=pred_real)
            dloss.backward()
            disc.optimizer.step()
            # log
            dinfo = {'t_'+k:v for k,v in dinfo.items()}
            logmap.update(dinfo)

        if True:
            #print('G')
            # Backward G
            gen.optimizer.zero_grad()
            g_loss, ginfo = gen.calc_loss(rec, gt, disc=disc)
            g_loss.backward()
            gen.optimizer.step()
            # log
            ginfo = {'t_'+k:v for k,v in ginfo.items()}
            logmap.update(ginfo)
        
        # Denorm the output so that they lie between 0 and 1
        rec_dn = loader.dataset.denormalize(rec)
        gt_dn = loader.dataset.denormalize(gt)

        # Logging and monitoring
        b, c, t, h, w = gt.size()
        rec_batch = rec_dn.permute(0,2,1,3,4).contiguous().view(b*t, c, h, w)
        gt_batch = gt_dn.permute(0,2,1,3,4).contiguous().view(b*t, c, h, w)
        pval = psnr(rec_batch.data, gt_batch.data)
        ssim_val = ssim(rec_batch.data, gt_batch.data)
        psnr_meter.update(pval, rec_batch.size(0))
        ssim_meter.update(ssim_val, rec_batch.size(0))

        logmap.update({'t_psnr': pval, 't_ssim': ssim_val})
        logger.add(epoch, logmap)
        
        # Visualization
        if i % opt['image_freq'] == 0:
            tid = int(ctx.size(2)/2)
            ctx_vis = torch.nn.functional.interpolate(ctx[:,:,tid,:,:], size=[h, w], mode='bilinear')
            imgs = torch.cat([gt.data[:,:,tid,:,:], ctx_vis.data, rec.data[:,:,tid,:,:]], dim=3)

            utl.save_images(epoch, i, tag=join(opt['outdir'], 'train'), r=visrange, img=imgs)
            
        duration_meter.update(time.time() - start_time)
        if i % opt['print_freq'] == 0:
            print_loss(logmap, epoch, i, len(loader),psnr_meter, ssim_meter, duration_meter)
        
    # Update lr
    update_lr(epoch, gen, disc)
    
    logger.save()
    if epoch % opt['log_freq'] == 0:
        save_param_dict_density_plt(significant_param_dict(gen, disc), epoch, i, opt['logdir'])
    
    return psnr_meter.avg, ssim_meter.avg

def validate(gen, disc, loader, epoch):
    gen.eval()
    if disc is not None: disc.eval()

    psnr_meter, ssim_meter = AverageMeter(), AverageMeter()
    with torch.no_grad():
        for i, batch in enumerate(loader):
            ctx, gt = batch[0].to(device), batch[1].to(device)

            # Forward G
            rec = gen(ctx)

            # Validation
            logmap = {}
            if disc is not None:
                pred_fake = disc(rec.detach())
                pred_real = disc(gt)
                _, dinfo = disc.calc_loss(pred_fake=pred_fake, pred_real=pred_real)
                dinfo = {'v_'+k:v for k,v in dinfo.items()}
                logmap.update(dinfo)

            _, ginfo = gen.calc_loss(rec, gt, disc=disc)
            ginfo = {'v_'+k:v for k,v in ginfo.items()}
            logmap.update(ginfo)

            # Denorm the output so that they lie between 0 and 1
            rec_dn = loader.dataset.denormalize(rec)
            gt_dn = loader.dataset.denormalize(gt)

            # Logging and monitoring
            b, c, t, h, w = gt.size()
            rec_batch = rec_dn.permute(0,2,1,3,4).contiguous().view(b*t, c, h, w)
            gt_batch = gt_dn.permute(0,2,1,3,4).contiguous().view(b*t, c, h, w)
            pval = psnr(rec_batch.data, gt_batch.data)
            ssim_val = ssim(rec_batch.data, gt_batch.data)
            psnr_meter.update(pval, rec_batch.size(0))
            ssim_meter.update(ssim_val, rec_batch.size(0))

            logmap.update({'v_psnr': pval, 'v_ssim': ssim_val})

            logger.add(epoch, logmap)

            # Save validation images
            if i % opt['val_image_freq'] == 0:
                tid = int(ctx.size(2)/2)
                ctx_vis = torch.nn.functional.interpolate(ctx[:,:,tid,:,:], size=[h, w], mode='bilinear')
                imgs = torch.cat([gt.data[:,:,tid,:,:], ctx_vis.data, rec.data[:,:,tid,:,:]], dim=3)

                utl.save_images(epoch, i, tag=join(opt['outdir'], 'val'), r=visrange, img=imgs)
            
            if i % opt['val_print_freq'] == 0:
                print('eval [{}/{}] psnr: {}, ssim: {}'.format(i, len(loader), psnr_meter.avg, ssim_meter.avg))

    logger.save()
    print('Epoch %d validation psnr: %.4f ssim: %.4f'%(epoch+1, psnr_meter.avg, ssim_meter.avg), flush=True)
    return psnr_meter.avg, ssim_meter.avg

def print_loss(logmap, epoch, i, niter, psnr_meter, ssim_meter, duration_meter):
    log_str = '[{e:3d}/{nepoch}] {i:3d}/{niter}'
    kwargs = {'e':epoch+1, 'nepoch':opt['epochs'], 'i':i+1, 'niter':niter}

    if 't_d_loss' in logmap:
        log_str += ' d:{disc:.4f}'
        kwargs['disc'] = logmap['t_d_loss']

    if 't_g_loss' in logmap:
        log_str += ' g:{g_loss:.4f}[|'
        kwargs['g_loss'] = logmap['t_g_loss']

    if 't_g_rec' in logmap:
        log_str += 'r {g_rec:.3f}|'
        kwargs['g_rec'] = logmap['t_g_rec']
    if 't_g_feat' in logmap:
        log_str += 'f {g_feat:.3f}|'
        kwargs['g_feat'] = logmap['t_g_feat']
    if 't_g_gram' in logmap:
        log_str += 'g {g_gram:.3f}|'
        kwargs['g_gram'] = logmap['t_g_gram']
    if 't_g_style' in logmap:
        log_str += 'fg {g_style:.3f}|'
        kwargs['g_style'] = logmap['t_g_style']
    if 't_g_tv' in logmap:
        log_str += 'tv {g_tv:.3f}|'
        kwargs['g_tv'] = logmap['t_g_tv']
    
    if 't_g_adv' in logmap:
        log_str += 'a {g_adv:.3f}|'
        kwargs['g_adv'] = logmap['t_g_adv']

    log_str += '] p:{psnr:.2f} s:{ssim:.2f} ({dur.avg:.3f}s| {dur.val:.3f}s)'
    kwargs.update({'psnr':psnr_meter.avg, 'ssim':ssim_meter.val, 'dur':duration_meter})
    print(log_str.format(**kwargs), flush=True)

        
def significant_param_dict(gen, disc):
    param_dict = {}
    param_dict['g_kernel'] = gen.significant_parameters()
    param_dict['g_grad'] = gen.significant_grads()
    if disc is not None:
        param_dict['d_kernel'] = disc.significant_parameters()
        param_dict['d_grad'] = disc.significant_grads()

    return param_dict

def update_lr(epoch, gen, disc):
    gen.scheduler.step()
    logger.add(epoch, gen.get_lr())
    if disc is not None:
        disc.scheduler.step()
        logger.add(epoch, disc.get_lr())

if __name__ == '__main__':
    main()