import os
import math

from sklearn.model_selection import train_test_split

import torch
import torchvision.transforms as transforms

from reader.base_dataset import BaseDataset

def get_test_loader(name, root, batch_size, workers, **kwargs):
    if name == 'ucf101':
        test_file = 'reader/ucf101/test.txt'
        datadir = os.path.join(root, 'videos')
        eval_data = _prepare_dset(datadir=datadir, augment=False, split_file=test_file, include_all=True, **kwargs)
    elif name == 'hmdb51':
        test_file = 'reader/hmdb51/1/test.txt'
        datadir = os.path.join(root, 'hmdb51_org')
        eval_data = _prepare_dset(datadir=datadir, augment=False, split_file=test_file, include_all=True, **kwargs)
    else:
        raise Exception('Unknown dataset ' + str(name))

    eval_loader = torch.utils.data.DataLoader(eval_data, batch_size=batch_size, shuffle=False, num_workers=workers)
    return eval_loader

def get_val_loader(name, root, batch_size, workers, **kwargs):
    if name == 'ucf101':
        val_file = 'reader/ucf101/test.txt'
        datadir = os.path.join(root, 'videos')
        eval_data = _prepare_dset(datadir=datadir, augment=False, split_file=val_file, include_all=False, **kwargs)
    elif name == 'hmdb51':
        val_file = 'reader/hmdb51/1/test.txt'
        datadir = os.path.join(root, 'hmdb51_org')
        eval_data = _prepare_dset(datadir=datadir, augment=False, split_file=val_file, include_all=False, **kwargs)
    else:
        raise Exception('Unknown dataset ' + str(name))

    eval_loader = torch.utils.data.DataLoader(eval_data, batch_size=batch_size, shuffle=False, num_workers=workers)
    return eval_loader

def get_train_loader(name, root, batch_size, workers, **kwargs):
    if name == 'ucf101':
        train_file = 'reader/ucf101/train.txt'
        datadir = os.path.join(root, 'videos')
        train_data = _prepare_dset(datadir=datadir, augment=True, split_file=train_file, include_all=False, **kwargs)
    elif name == 'hmdb51':
        train_file = 'reader/hmdb51/1/train.txt'
        datadir = os.path.join(root, 'hmdb51_org')
        train_data = _prepare_dset(datadir=datadir, augment=True, split_file=train_file, include_all=False, **kwargs)
    else:
        raise Exception('Unknown dataset ' + str(name))

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=workers)
    return train_loader

"""
Args:
    split_file: File that indicates video paths, number of frame and action id

    include_all: If True, all frames from a video are included, 
                 If False, frame start points are randomly selected during training
"""
def _prepare_dset(datadir, nframe, split_file, include_all, augment, **kwargs):
    filearr = []
    with open(split_file, 'r') as tf:
        for tid, line in enumerate(tf):
            filename, vlength, classid = line.replace('\n','').replace('\r', '').split(" ")
            vlength, classid = int(vlength)-1, int(classid)

            if include_all:
                nchunks = vlength // nframe
                if nchunks == 0:
                    filearr.append({'id':tid, 'path':os.path.join(datadir, filename), 'length':vlength, 'label':classid, 'start_frame':0, 'tag':0})
                else:
                    for i in range(nchunks):
                        filearr.append({'id':tid, 'path':os.path.join(datadir, filename), 'length':vlength, 'label':classid, 'start_frame':i*nframe, 'tag':i})
            else:
                filearr.append({'id':tid, 'path':os.path.join(datadir, filename), 'length':vlength, 'label':classid, 'tag':None})

    dset = BaseDataset(filearr, augment=augment, nframe=nframe, testmode=include_all, **kwargs)
    return dset
