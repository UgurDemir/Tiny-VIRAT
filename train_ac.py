# Inpired from https://github.com/piergiaj/pytorch-i3d/blob/master/train_i3d.py
import torch
torch.multiprocessing.set_sharing_strategy('file_system')

import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from os.path import join
import random
import time
import copy
import sys

from deepnn.measure.psnr import PSNR
from deepnn.measure.ssim import SSIM
from deepnn.measure.accuracy import Accuracy
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
#sys.argv.extend(['desktop','193a'])
if len(sys.argv) != 3:
    raise Exception('Please see usage: python main.py <machine_name> <exp_id>')
machine_name, exp_name = sys.argv[1], sys.argv[2]
#exp_name = '150a'

opt = conf.parse(machine_name, exp_name)

device = torch.device("cuda" if opt['cuda'] else "cpu")
cpu = torch.device('cpu')
gpu = torch.device('cuda')
visrange = (-1.0, 1.0) if opt['dataset']['norm'] == '-11' else (0.0, 1.0)

# Measurements
accuracy = Accuracy()


# Logging and Visualization
se = opt['start_epoch'] if opt['start_epoch'] > 0 else -1
logger = HistorySaver(join(opt['logdir'], 'hist'), se)
logger.get_meta().update(copy.deepcopy(opt))


def main():
    train_loader = get_train_loader(**opt['dataset'])
    eval_loader = get_val_loader(**opt['dataset'])

    if 'netG' in opt:
        gen = NetworkWrapper(name='G', **opt['netG']).to(device)
        gen.disable_grads()
        gen.eval()
    else:
        gen = None

    net = NetworkWrapper(name='netA', **opt['netA'])
    if 'reset_ac_out' in opt:
        print('Logits reset')
        net.model.replace_logits(opt['reset_ac_out'])
    net = net.to(device)

    criterion = nn.__dict__[opt['netA']['loss']['loss_type']]().to(device)

    print('A', net)

    # TRAINING
    best_acc = 0.0
    for epoch in range(opt['start_epoch'], opt['epochs']):
        train_acc = train(net, gen, criterion, train_loader, epoch)
        val_acc = validate(net, gen, criterion, eval_loader, epoch)

        # Save models regularly
        if (epoch+1)%opt['save_freq'] == 0:
            save(net, join(opt['modeldir'],"chk_ac_%s.pth.tar"%epoch),
                      {'epoch': epoch, 'manuelSeed': opt['manual_seed'], 
                      'train_acc':train_acc, 'val_acc': val_acc})

        if val_acc > best_acc:
            best_acc = val_acc
            save(net, join(opt['modeldir'],"chk_ac_best.pth.tar"),
                      {'epoch': epoch, 'manuelSeed': opt['manual_seed'], 
                      'train_acc':train_acc, 'val_acc': val_acc})


def save(netA, filename, extras={}):
    state = {}
    state[netA.name] = netA.state_dict()
    state.update(extras)
    torch.save(state, filename)

def train(net, gen, criterion, loader, epoch):
    net.train()
    
    acc_meter, loss_meter, duration_meter = AverageMeter(), AverageMeter(), AverageMeter()
    logmap = {}
    for i, batch in enumerate(loader):
        start_time = time.time()

        x, label = batch[0].to(device), batch[1].to(device)

        # Super-resolve the image if G is defined
        if gen is not None:
            gout = gen(x)
            rec, loc = (gout[0], gout[1]) if isinstance(gout, tuple) else (gout, None)
            acin = (rec * loc) if opt.get('apply_loc', False) else rec
        else:
            acin = x

        # Resnormalization
        acin = (acin + 1.0) * 0.5 if opt.get('norm_to_01', False) else acin

        # Forward netA
        y = net(acin)

        # Backward netA
        net.optimizer.zero_grad()
        loss = criterion(y, label)
        loss.backward()
        net.optimizer.step()

        # Logging and monitoring
        logmap['t_loss'] = loss.item()
        loss_meter.update(loss.item(), y.size(0))

        acc = accuracy(y, label)[0]
        acc_meter.update(acc, y.size(0))
        logmap.update({'t_acc': acc})
        print_kwargs = {}
        logger.add(epoch, logmap)

        duration_meter.update(time.time() - start_time)
        if i % opt['print_freq'] == 0:
            print_loss(logmap, epoch, i, len(loader),acc_meter, loss_meter, duration_meter, **print_kwargs)
        
    # Update lr
    net.scheduler.step()
    logger.add(epoch, net.get_lr())
    
    logger.save()
    if epoch % opt['log_freq'] == 0:
        save_param_dict_density_plt(significant_param_dict(net), epoch, i, opt['logdir'])
    
    return acc_meter.avg

def validate(net, gen, criterion, loader, epoch):
    net.eval()

    acc_meter, loss_meter = AverageMeter(), AverageMeter()
    with torch.no_grad():
        for i, batch in enumerate(loader):
            x, label = batch[0].to(device), batch[1].to(device)

            # Super-resolve the image if G is defined
            if gen is not None:
                gout = gen(x)
                rec, loc = (gout[0], gout[1]) if isinstance(gout, tuple) else (gout, None)
                acin = (rec * loc) if opt.get('apply_loc', False) else rec
            else:
                acin = x
            
            # Resnormalization
            acin = (acin + 1.0) * 0.5 if opt.get('norm_to_01', False) else acin

            # Forward netA
            y = net(acin)

            # Validation
            loss = criterion(y, label)

            logmap = {'v_loss': loss.item()}
            loss_meter.update(loss.item(), y.size(0))
            acc = accuracy(y, label)[0]
            acc_meter.update(acc, y.size(0))
            logmap.update({'v_acc': acc})
            
            logger.add(epoch, logmap)
            
            if i % opt['val_print_freq'] == 0:
                print('eval [{}/{}] acc: {acc:.2f} loss:{loss:.4f}'.format(i, len(loader), acc=acc_meter.avg, loss=loss_meter.avg))

    logger.save()
    print('Epoch %d validation acc: %.4f loss: %.4f'%(epoch+1, acc_meter.avg, loss_meter.avg), flush=True)
    return acc_meter.avg

def print_loss(logmap, epoch, i, niter, acc_meter, loss_meter, duration_meter, f1=None):
    kwargs = {'e':epoch+1, 'nepoch':opt['epochs'], 'i':i+1, 'niter':niter, 'loss':loss_meter.avg, 'acc':acc_meter.avg, 'dur':duration_meter}
    log_str = '[{e:3d}/{nepoch}] {i:3d}/{niter} acc: {acc:.2f} loss:{loss:.4f} '
    if f1 is not None:
        log_str += 'f1: {f1:.4f} '
        kwargs['f1'] = f1
    log_str += '({dur.avg:.3f}s| {dur.val:.3f}s)'
    
    print(log_str.format(**kwargs), flush=True)

        
def significant_param_dict(net):
    param_dict = {}
    param_dict['a_kernel'] = net.significant_parameters()
    param_dict['a_grad'] = net.significant_grads()

    return param_dict

def update_lr(epoch, gen, disc):
    gen.scheduler.step()
    logger.add(epoch, gen.get_lr())
    if disc is not None:
        disc.scheduler.step()
        logger.add(epoch, disc.get_lr())

if __name__ == '__main__':
    main()