import torch

import argparse
from os.path import join

import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from functools import reduce

from deepnn.nets.network_wrapper import NetworkWrapper
from reader.loader import get_test_loader
from deepnn.util.avg_meter import AverageMeter
from deepnn.measure.accuracy import Accuracy
from deepnn.measure.f1score import F1Score
from deepnn.measure.psnr import PSNR
from deepnn.measure.ssim import SSIM
import conf

accuracy = Accuracy()
f1_score = F1Score()
psnr = PSNR(peak=1.0)
ssim = SSIM()

class MeterCache():
    def __init__(self):
        self.cache = {}
    
    # **kwargs:
    #   <meter_name>: [<value>, <size>]
    def update(self, **kwargs):
        for k, v in kwargs.items():
            if k not in self.cache:
                self.cache[k] = AverageMeter()    
            self.cache[k].update(*v)

def get_networks(opt, exp_name, exp_root_dir, model_file_pf, device):
    # Generator Configuration
    if exp_name.endswith('sa'):
        netGConf = {k:v for k, v in opt['gan1']['netG'].items() if k in ['gpu','model','resume']}
        netGConf.update({'resume': join(exp_root_dir, exp_name, 'model/gan/chk_gan_{}.pth.tar'.format(model_file_pf))})
        netG = NetworkWrapper(name='G', **netGConf).to(device)
        netG.eval()
    else:
        netG = None

    # Classifier configuration
    netAConf = {k:v for k, v in opt['netA'].items() if k in ['gpu','model','resume']}
    if 'reset_ac_out' in opt:
        netAConf['model']['num_classes'] = opt['reset_ac_out']

    if exp_name.endswith('sa'):
        netAConf.update({'resume': join(exp_root_dir, exp_name, 'model/ac/chk_ac_{}.pth.tar'.format(model_file_pf))})
    elif exp_name.endswith('a'):
        netAConf.update({'resume': join(exp_root_dir, exp_name, 'model/chk_ac_{}.pth.tar'.format(model_file_pf))})
    else:
        raise Exception('Experiment is not compatible with test script')

    # Instantiate networks
    netA = NetworkWrapper(name='netA', **netAConf).to(device)
    netA.eval()

    return netG, netA

def get_data(exp_name, produce, batch, device):
    if exp_name.endswith('sa'):
        x, x_hr, label = batch[0].to(device), batch[1].to(device), batch[2].to(device)
    elif exp_name.endswith('a'):
        x, x_hr, label = batch[0].to(device), None, batch[1].to(device)

    return x, x_hr, label

def perform_test(cluster_name, exp_name, exp_root_dir, model_file_pf, cuda=False):
    # Dataset Configurations
    opt = conf.parse(cluster_name, exp_name)
    opt['f1_threshold'] = 0.5
    dataset = opt['dataset']
    #dataset['produce'] = ['orig', 'down', 'up', 'label']
    dataset['batch_size'] = 64
    loader = get_test_loader(**dataset)

    # Device selection
    device = torch.device('cuda') if cuda else torch.devide('cpu')

    netG, netA = get_networks(opt, exp_name, exp_root_dir, model_file_pf, device)
    
    meters = MeterCache()
    predictions, ground_truth = [], []
    for i, batch in enumerate(loader):
        x, hr_gt, label = get_data(exp_name, dataset['produce'], batch, device)

        # Super-resolve the image if G is defined
        rec = netG(x) if netG is not None else x

        # Forward netA
        y = netA(rec)

        # Save predictions for F1 score calculation
        ground_truth.append(label.cpu().data.numpy())
        predictions.append(y.cpu().data.numpy())

        # Denorm the output so that they lie between 0 and 1
        if netG is not None:
            rec_dn = loader.dataset.denormalize(rec)
            gt_dn = loader.dataset.denormalize(hr_gt)

            # PSNR and SSIM
            b, c, t, h, w = hr_gt.size()
            rec_batch = rec_dn.permute(0,2,1,3,4).contiguous().view(b*t, c, h, w)
            gt_batch = gt_dn.permute(0,2,1,3,4).contiguous().view(b*t, c, h, w)
            pval = psnr(rec_batch.data, gt_batch.data)
            ssim_val = ssim(rec_batch.data, gt_batch.data)
            metric_map = {'psnr':[pval, rec_batch.size(0)], 'ssim':[ssim_val, rec_batch.size(0)]}
            meters.update(**metric_map)

    # Print the final results
    print(exp_name+':')

    # F1 score calculation
    ground_truth = reduce(lambda x,y: np.concatenate([x,y], axis=0), ground_truth)
    predictions = reduce(lambda x,y: np.concatenate([x,y], axis=0), predictions)
    print(ground_truth.shape, predictions.shape)
    predictions = (np.array(predictions) > opt['f1_threshold']).astype(int)
    ground_truth = (np.array(ground_truth) > opt['f1_threshold']).astype(int)
    results_actions = precision_recall_fscore_support(np.array(ground_truth), np.array(predictions), average=None)
    f1_scores, precision, recall = results_actions[2], results_actions[0], results_actions[1]
    f1_mean, prec_mean, rec_mean = np.mean(f1_scores), np.mean(precision), np.mean(recall)
    print('Test F1-Score:')
    print(str(f1_scores)) 
    print('Test, [F1-Score: %4f] [Precision: %4f] [Recall: %4f]' % (f1_mean, prec_mean, rec_mean))

    for k, meter in meters.cache.items():
        print('    -{} : {:.2f}'.format(k, meter.avg), flush=True)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', type=str, default="/home/ugur/Documents/exps/vidres", help='Experiment root folder')
    parser.add_argument('-e', nargs='+', required=True, help='Experiment names')
    parser.add_argument('--chk', type=str, default='best', help='Model checkpoint file name postfix')
    parser.add_argument('-m', choices=['crcv', 'newton', 'desktop'], required=True, help='Machine name: [crcv|newton|desktop]')
    parser.add_argument('--gpu', action='store_true', default=True, help='GPU usage flag')
    argv = parser.parse_args()

    print(argv)

    with torch.no_grad():
        for exp_name in argv.e:
            perform_test(cluster_name=argv.m, exp_name=exp_name, exp_root_dir=argv.r, model_file_pf=argv.chk, cuda=argv.gpu)