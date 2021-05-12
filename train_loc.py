import torch
torch.multiprocessing.set_sharing_strategy('file_system')

import torch.nn as nn
from torch.autograd import Variable

from os.path import join
import random
import time
import copy
import sys

from deepnn.measure.psnr import PSNR
from deepnn.measure.ssim import SSIM
from deepnn.measure.accuracy import Accuracy
from deepnn.measure.f1score import F1Score
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
#exp_name = '150sa'

opt = conf.parse(machine_name, exp_name)

device = torch.device("cuda" if opt['cuda'] else "cpu")
cpu = torch.device('cpu')
gpu = torch.device('cuda')
visrange = (-1.0, 1.0) if opt['dataset']['norm'] == '-11' else (0.0, 1.0)

# Measurements
psnr = PSNR(peak=1.0)
ssim = SSIM()
accuracy = Accuracy()
f1_score = F1Score()

# Logging and Visualization
se = opt['start_epoch'] if opt['start_epoch'] > 0 else -1
logger = HistorySaver(join(opt['logdir'], 'hist'), se)
logger.get_meta().update(copy.deepcopy(opt))


def main():
    # Dataset prepration
    train_loader = get_train_loader(**opt['dataset'])
    eval_loader = get_val_loader(**opt['dataset'])

    # Network Definitions
    gen = GeneratorWrapper(name='G', **opt['gan1']['netG']).to(device)
    disc = DiscriminatorWrapper(name='D', **opt['gan1']['netD']).to(device) if 'netD' in opt['gan1'] else None
    neta = NetworkWrapper(name='netA', **opt['netA'])

    if 'reset_ac_out' in opt:
        print('Logits reset')
        neta.model.replace_logits(opt['reset_ac_out'])
        
    neta = neta.to(device)
    crit_ac = nn.__dict__[opt['netA']['loss']['loss_type']]().to(device)

    print('G', gen)
    print('D', disc)
    print('A', neta)

    # TRAINING
    best_acc = 0.0
    for epoch in range(opt['start_epoch'], opt['epochs']):
        train_acc = train(gen, disc, neta, crit_ac, train_loader, epoch)
        val_acc = validate(gen, disc, neta, crit_ac, eval_loader, epoch)

        # Save models regularly
        if (epoch+1)%opt['save_freq'] == 0:
            save(join(opt['gan_modeldir'],"chk_sr_%s.pth.tar"%epoch), gen, disc,
                epoch=epoch, manuelSeed=opt['manual_seed'])

            save(join(opt['ac_modeldir'],"chk_ac_%s.pth.tar"%epoch), neta,
                epoch=epoch, manuelSeed=opt['manual_seed'], train_acc=train_acc, val_acc=val_acc)

        # Save action classifier (ac) model if it has the best validation
        if val_acc > best_acc:
            best_acc = val_acc
            save(join(opt['ac_modeldir'],"chk_ac_best.pth.tar"), neta,
                epoch=epoch, manuelSeed=opt['manual_seed'], train_acc=train_acc, val_acc=val_acc)

            save(join(opt['gan_modeldir'],"chk_gan_best.pth.tar"), gen, disc,
                epoch=epoch, manuelSeed=opt['manual_seed'], train_acc=train_acc, val_acc=val_acc)


def save(filename, *nets, **extras):
    state = {}
    for n in nets:
        if n is not None:
            state[n.name] = n.state_dict()
    state.update(extras)
    torch.save(state, filename)

def train(gen, disc, neta, crit_ac, loader, epoch):
    neta.train()
    gen.train()
    if disc is not None: disc.train()
    
    psnr_mtr, ssim_mtr, acc_mtr, dur_mtr = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
    for i, batch in enumerate(loader):
        start_time = time.time()

        ctx, gt, label = batch[0].to(device), batch[1].to(device), batch[2].to(device)

        # Forward G
        rec, sln = gen(ctx)

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

        #gen.optimizer.zero_grad()
        # Forward A
        acin = rec.detach()
        # Resnormalization
        acin = (acin + 1.0) * 0.5 if opt.get('norm_to_01', False) else acin
        y = neta(acin)
        # Backward A
        neta.optimizer.zero_grad()
        ac_loss = crit_ac(y, label)
        ac_loss.backward()
        neta.optimizer.step()
        logmap['t_ac_loss'] = ac_loss.item()

        ################ TRAIN GENERATOR ################
        # Backward G
        gen.optimizer.zero_grad()
        g_loss, ginfo = gen.calc_loss(rec, gt, loc_pred=sln, disc=disc, neta=neta, action_labels=label)
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
        psnr_mtr.update(pval, rec_batch.size(0))
        ssim_mtr.update(ssim_val, rec_batch.size(0))
        
        acc = accuracy(y, label)[0]
        acc_mtr.update(acc, y.size(0))
        logmap.update({'t_acc': acc})
        print_kwargs = {}

        logmap.update({'t_psnr': pval, 't_ssim': ssim_val})
        logger.add(epoch, logmap)
        
        # Visualization
        if i % opt['image_freq'] == 0:
            tid = int(ctx.size(2)/2)
            ctx_dn = loader.dataset.denormalize(ctx)
            ctx_vis = torch.nn.functional.interpolate(ctx_dn[:,:,tid,:,:], size=[h, w], mode='bilinear', align_corners=False)
            sln_vis = sln.repeat(1,3,1,1,1)
            loc_mask = (sln_vis > 0.5).float()
            masked_hr = rec_dn * sln
            imgs = torch.cat([gt_dn.data[:,:,tid,:,:], ctx_vis.data, rec_dn.data[:,:,tid,:,:], masked_hr.data[:,:,tid,:,:], sln_vis.data[:,:,tid,:,:], loc_mask.data[:,:,tid,:,:]], dim=3)

            utl.save_images(epoch, i, tag=join(opt['outdir'], 'train'), r=(0.0, 1.0), img=imgs)
            
        dur_mtr.update(time.time() - start_time)
        if i % opt['print_freq'] == 0:
            print_loss(logmap, epoch, i, len(loader), psnr_mtr, ssim_mtr, acc_mtr, dur_mtr, **print_kwargs)
        
    # Update lr
    update_lr(epoch, gen, disc, neta)
    
    logger.save()
    if epoch % opt['log_freq'] == 0:
        save_param_dict_density_plt(significant_param_dict(gen, disc, neta), epoch, i, opt['logdir'])
    

    return acc_mtr.avg

def validate(gen, disc, neta, crit_ac, loader, epoch):
    neta.eval()
    gen.eval()
    if disc is not None: disc.eval()

    psnr_mtr, ssim_mtr, acc_mtr = AverageMeter(), AverageMeter(), AverageMeter()
    with torch.no_grad():
        for i, batch in enumerate(loader):
            ctx, gt, label = batch[0].to(device), batch[1].to(device), batch[2].to(device)

            # Forward G
            rec, sln = gen(ctx)

            # Validation
            logmap = {}
            if disc is not None:
                pred_fake = disc(rec.detach())
                pred_real = disc(gt)
                _, dinfo = disc.calc_loss(pred_fake=pred_fake, pred_real=pred_real)
                dinfo = {'v_'+k:v for k,v in dinfo.items()}
                logmap.update(dinfo)
            
            # Action classification
            acin = rec.detach()
            # Resnormalization
            acin = (acin + 1.0) * 0.5 if opt.get('norm_to_01', False) else acin
            
            y = neta(acin)
            ac_loss = crit_ac(y, label)
            logmap['v_ac_loss'] = ac_loss.item()

            _, ginfo = gen.calc_loss(rec, gt, loc_pred=sln, disc=disc, neta=neta, action_labels=label)
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
            psnr_mtr.update(pval, rec_batch.size(0))
            ssim_mtr.update(ssim_val, rec_batch.size(0))

            acc = accuracy(y, label)[0]
            acc_mtr.update(acc, y.size(0))
            logmap.update({'v_acc': acc})

            logmap.update({'v_psnr': pval, 'v_ssim': ssim_val})
            logger.add(epoch, logmap)

            # Save validation images
            if i % opt['val_image_freq'] == 0:
                tid = int(ctx.size(2)/2)
                ctx_dn = loader.dataset.denormalize(ctx)
                ctx_vis = torch.nn.functional.interpolate(ctx_dn[:,:,tid,:,:], size=[h, w], mode='bilinear', align_corners=False)
                sln_vis = sln.repeat(1,3,1,1,1)
                loc_mask = (sln_vis > 0.5).float()
                masked_hr = rec_dn * sln
                imgs = torch.cat([gt_dn.data[:,:,tid,:,:], ctx_vis.data, rec_dn.data[:,:,tid,:,:], masked_hr.data[:,:,tid,:,:], sln_vis.data[:,:,tid,:,:], loc_mask.data[:,:,tid,:,:]], dim=3)

                utl.save_images(epoch, i, tag=join(opt['outdir'], 'val'), r=(0.0, 1.0), img=imgs)
            
            if i % opt['val_print_freq'] == 0:
                print('eval [{}/{}] psnr: {}, ssim: {}, acc: {}'.format(i, len(loader), psnr_mtr.avg, ssim_mtr.avg, acc_mtr.avg))

    logger.save()
    print('Epoch %d validation psnr: %.4f ssim: %.4f acc: %.4f'%(epoch+1, psnr_mtr.avg, ssim_mtr.avg, acc_mtr.avg), flush=True)

    return acc_mtr.avg

def print_loss(logmap, epoch, i, niter, psnr_meter, ssim_meter, acc_meter, duration_meter, f1=None):
    log_str = '[{e:3d}/{nepoch}] {i:3d}/{niter}'
    kwargs = {'e':epoch+1, 'nepoch':opt['epochs'], 'i':i+1, 'niter':niter}

    if 't_d_loss' in logmap:
        log_str += ' d:{disc:.4f}'
        kwargs['disc'] = logmap['t_d_loss']

    if 't_g_loss' in logmap:
        log_str += ' g:{g_loss:.4f}[|'
        kwargs['g_loss'] = logmap['t_g_loss']

    if 't_g_recloc' in logmap:
        log_str += 'r {g_recloc:.3f}|'
        kwargs['g_recloc'] = logmap['t_g_recloc']
    if 'g_loc_reg' in logmap:
        log_str += 'r {g_loc_reg:.3f}|'
        kwargs['g_loc_reg'] = logmap['g_loc_reg']
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
    if 't_g_ac' in logmap:
        log_str += 'ac {g_ac:.3f}|'
        kwargs['g_ac'] = logmap['t_g_ac']
    
    if 't_g_adv' in logmap:
        log_str += 'a {g_adv:.3f}|'
        kwargs['g_adv'] = logmap['t_g_adv']
    if 't_g_seg' in logmap:
        log_str += 's {g_seg:.3f}|'
        kwargs['g_seg'] = logmap['t_g_seg']

    log_str += '] p:{psnr:.2f} s:{ssim:.2f} '
    kwargs.update({'psnr':psnr_meter.avg, 'ssim':ssim_meter.avg})

    log_str += 'acloss: {acloss:.4f} acc: {acc.avg:.2f} '
    if f1 is not None:
        log_str += 'f1: {f1:.4f} '
        kwargs['f1'] = f1
    log_str += '({dur.avg:.3f}s| {dur.val:.3f}s)'
    kwargs.update({'acloss':logmap['t_ac_loss'], 'acc':acc_meter, 'dur':duration_meter})
    print(log_str.format(**kwargs), flush=True)

        
def significant_param_dict(gen, disc, neta):
    param_dict = {}
    param_dict['g_kernel'] = gen.significant_parameters()
    param_dict['g_grad'] = gen.significant_grads()
    if disc is not None:
        param_dict['d_kernel'] = disc.significant_parameters()
        param_dict['d_grad'] = disc.significant_grads()
    param_dict['a_kernel'] = neta.significant_parameters()
    param_dict['a_grad'] = neta.significant_grads()

    return param_dict

def update_lr(epoch, gen, disc, neta):
    gen.scheduler.step()
    logger.add(epoch, gen.get_lr())
    if disc is not None:
        disc.scheduler.step()
        logger.add(epoch, disc.get_lr())
    neta.scheduler.step()
    logger.add(epoch, neta.get_lr())

if __name__ == '__main__':
    main()