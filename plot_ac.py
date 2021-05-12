import os
from os.path import join
import argparse

import numpy as np
import matplotlib.pyplot as plt

from deepnn.util.plotfig import plot_hist

def p1(log_path, out_path='./', save=False, display=True):
    logmap = np.load(log_path, allow_pickle=True)['data'].item()

    print(logmap.keys())

    plot_hist(logmap, 
        Accuracy={'src':['t_acc','v_acc'], 'label':['train','eval'],'c':['r-','b-'],'mode':'epoch'},
        Loss={'src':['t_loss','v_loss'], 'label':['train','eval'],'c':['r-','b-'],'mode':'epoch'}
    )
    if save: plt.savefig(os.path.join(out_path,'figure_1.png'))

    if display: plt.show()

# Main
parser = argparse.ArgumentParser()
parser.add_argument('-e', type=str, required=True, help='Experiment name')
parser.add_argument('-d', action='store_true', help='Display plots')
parser.add_argument('-s', action='store_true', help='Save plots')
if __name__ == "__main__":
    opt = parser.parse_args()
    print(opt)
    log_path = join('_results/{}/log/hist.npz'.format(opt.e))
    out_dir = join('_results/{}/log/'.format(opt.e))

    p1(log_path=log_path, out_path=out_dir, save=opt.s, display=opt.d)