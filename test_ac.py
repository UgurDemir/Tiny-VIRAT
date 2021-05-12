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
import conf

def post_processing(results):
    pred_map = {}
    for tid in results:
        predarr = [p for p in results[tid].values()]
        #preds = np.concatenate(predarr, axis=0)
        preds = np.stack(predarr, axis=0)
        pred = preds.mean(axis=0)
        pred_map[tid] = pred

    return pred_map

def post_processing2(results):
    pred_map = {}
    for tid in results:
        mid_tag = sorted(results[tid].keys())[len(results[tid])//2]
        pred_map[tid] = results[tid][mid_tag]

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
    #dataset['workers'] = 0
    dataset['produce'] = ['orig', 'down', 'label']
    dataset['batch_size'] = 30
    loader = get_test_loader(**dataset)

    # Classifier configuration
    netAConf = {k:v for k, v in opt['netA'].items() if k in ['gpu','model','resume']}
    netAConf['model']['num_classes'] = opt['reset_ac_out']
    #netAConf.update({'resume': join(exp_root_dir, exp_name, '_results/model/chk_ac_{}.pth.tar'.format(model_file_pf))})
    netAConf.update({'resume': join(exp_root_dir, exp_name, 'model/chk_ac_{}.pth.tar'.format(model_file_pf))})

    # Device selection
    device = torch.device('cuda') if cuda else torch.devide('cpu')

    # Instantiate networks
    netA = NetworkWrapper(name='netA', **netAConf).to(device)
    netA.eval()
    if 'netG' in opt:
        netG = NetworkWrapper(name='G', **opt['netG']).to(device)
        netG.eval()
    else:
        netG = None

    results = {} # {'tube_id':{'<tag>': [predictions], ...}, ...}
    gt_map = {}
    pbar = tqdm(total=len(loader))
    for i, batch in enumerate(loader):
        x, x_down, label, tube_ids, tube_tags = batch[0].to(device), batch[1].to(device), batch[2], batch[3], batch[4]

        # Super-resolve the image if G is defined
        if netG is not None:
            gout = netG(x_down)
            rec, loc = (gout[0], gout[1]) if isinstance(gout, tuple) else (gout, None)
            acin = (rec * loc) if opt.get('apply_loc', False) else rec
        else:
            acin = x

        # Resnormalization
        acin = (acin + 1.0) * 0.5 if opt.get('norm_to_01', False) else acin

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
        pbar.update(1)
    pbar.close()

    pred_map = post_processing(results)
    ground_truth, predictions = get_gt_prediction_arr(gt_map, pred_map)
    predictions = np.argmax(predictions, axis=1)

    acc = accuracy_score(ground_truth, predictions)

    # Print the final results
    print(exp_name+':')
    print('    -Accuracy : {:.4f}'.format(acc), flush=True)

    # Write results to the file
    with open(join(exp_root_dir, exp_name, 'log', 'test.txt'), 'a+') as outfile:
        outfile.write('Exp {} ({})\n-----\n'.format(exp_name, datetime.now().strftime("%m/%d/%Y, %H:%M:%S")))
        outfile.write('    -Accuracy : {:.4f}\n'.format(acc))
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
