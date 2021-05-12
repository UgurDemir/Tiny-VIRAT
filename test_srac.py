import torch
torch.multiprocessing.set_sharing_strategy('file_system')

import argparse
from os.path import join
from datetime import datetime
import numpy as np
from sklearn.metrics import accuracy_score
from tqdm import tqdm

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


def post_processing(results):
    pred_map = {}
    for tid in results:
        predarr = [p for p in results[tid].values()]
        #preds = np.concatenate(predarr, axis=0)
        preds = np.stack(predarr, axis=0)
        pred = preds.mean(axis=0)
        pred_map[tid] = pred

    return pred_map

def get_gt_prediction_arr(gt_map, pred_map):
    ground_truth = []
    predictions = []
    for tid in gt_map:
        ground_truth.append(gt_map[tid])
        predictions.append(pred_map[tid])

    return np.array(ground_truth), np.array(predictions)


def perform_test(cluster_name, exp_name, exp_root_dir, model_file_pf, cuda=False):
    # Dataset Configurations
    opt = conf.parse(cluster_name, exp_name)
    dataset = opt['dataset']
    dataset['produce'] = ['orig', 'down', 'label']
    dataset['batch_size'] = 64
    loader = get_test_loader(**dataset)

    # Generator Configuration
    netGConf = {k:v for k, v in opt['gan1']['netG'].items() if k in ['gpu','model','resume']}
    netGConf.update({'resume': join(exp_root_dir, exp_name, 'model/gan/chk_gan_{}.pth.tar'.format(model_file_pf))})

    # Classifier configuration
    netAConf = {k:v for k, v in opt['netA'].items() if k in ['gpu','model','resume']}
    if 'reset_ac_out' in opt:
        netAConf['model']['num_classes'] = opt['reset_ac_out']
    #netAConf.update({'resume': join(exp_root_dir, exp_name, '_results/model/chk_ac_{}.pth.tar'.format(model_file_pf))})
    netAConf.update({'resume': join(exp_root_dir, exp_name, 'model/ac/chk_ac_{}.pth.tar'.format(model_file_pf))})

    # Device selection
    device = torch.device('cuda') if cuda else torch.devide('cpu')

    # Instantiate networks
    netA = NetworkWrapper(name='netA', **netAConf).to(device)
    netA.eval()
    netG = NetworkWrapper(name='G', **netGConf).to(device)
    netG.eval()
    
    meters = MeterCache()
    results = {} # {'tube_id':{'<tag>': [predictions], ...}, ...}
    gt_map = {}
    pbar = tqdm(total=len(loader))
    for i, batch in enumerate(loader):
        x, x_down, label, tube_ids, tube_tags = batch[0].to(device), batch[1].to(device), batch[2], batch[3], batch[4]

        # Super-resolve the image if G is defined
        rec = netG(x_down)

        # Resnormalization
        acin = (rec + 1.0) * 0.5 if opt.get('norm_to_01', False) else rec

        # Forward netA
        y = netA(acin)

        for bid in range(y.size(0)):
            tid, tag = tube_ids[bid].item(), tube_tags[bid].item()
            
            if tid not in gt_map:
                gt_map[tid] = label[bid].numpy()

            if tid not in results:
                results[tid] = {}

            if tag in results[tid]:
                raise Exception('Duplicate tag is found for tid:', tid, ' | tag:', tag)
            results[tid][tag] = y[bid].cpu().numpy()

        # Denorm the output so that they lie between 0 and 1
        rec_dn = loader.dataset.denormalize(rec)
        gt_dn = loader.dataset.denormalize(x)

        # PSNR and SSIM
        b, c, t, h, w = x.size()
        rec_batch = rec_dn.permute(0,2,1,3,4).contiguous().view(b*t, c, h, w)
        gt_batch = gt_dn.permute(0,2,1,3,4).contiguous().view(b*t, c, h, w)
        pval = psnr(rec_batch.data, gt_batch.data)
        ssim_val = ssim(rec_batch.data, gt_batch.data)
        metric_map = {'psnr':[pval, rec_batch.size(0)], 'ssim':[ssim_val, rec_batch.size(0)]}
        meters.update(**metric_map)
        pbar.update(1)
    pbar.close()

    # Accuracy
    pred_map = post_processing(results)
    ground_truth, predictions = get_gt_prediction_arr(gt_map, pred_map)
    predictions = np.argmax(predictions, axis=1)
    acc = accuracy_score(ground_truth, predictions)

    # Print the final results
    print(exp_name+':')
    print('    -Accuracy : {:.4f}'.format(acc), flush=True)
    for k, meter in meters.cache.items():
        print('    -{} : {:.4f}'.format(k, meter.avg), flush=True)

    # Write results to the file
    with open(join(exp_root_dir, exp_name, 'log', 'test.txt'), 'a+') as outfile:
        outfile.write('Exp {} ({})\n-----\n'.format(exp_name, datetime.now().strftime("%m/%d/%Y, %H:%M:%S")))
        outfile.write('    -Accuracy : {:.4f}\n'.format(acc))
        for k, meter in meters.cache.items():
            outfile.write('    -{} : {:.4f}\n'.format(k, meter.avg))
        outfile.write('-'*100)
        outfile.write('\n ')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', type=str, default="_results", help='Experiment root folder')
    parser.add_argument('-e', nargs='+', required=True, help='Experiment names')
    parser.add_argument('--chk', type=str, default='best', help='Model checkpoint file name postfix')
    parser.add_argument('-m', choices=['crcv', 'newton', 'desktop'], required=True, help='Machine name: [crcv|newton|desktop]')
    parser.add_argument('--gpu', action='store_true', default=True, help='GPU usage flag')
    argv = parser.parse_args()

    print(argv)

    with torch.no_grad():
        for exp_name in argv.e:
            perform_test(cluster_name=argv.m, exp_name=exp_name, exp_root_dir=argv.r, model_file_pf=argv.chk, cuda=True)