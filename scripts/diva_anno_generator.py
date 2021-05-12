import torch
import torch.nn as nn

import os
import os.path
from os.path import join
import numpy as np
import h5py
import pickle
import functools

import torchvision.utils as vutil

def main():
    #clip_path = '/home/dennis/praveen/diva_multiscale/tubes/train/0000/VIRAT_S_000000/1_Talking_0_59/clip_0/clip_0_center.h5'
    clip_path = '/home/dennis/praveen/diva_multiscale/tubes/train/0002/VIRAT_S_000200_03_000657_000899/1_Interacts_591_650/clip_0/clip_0_center.h5'
    #anno_path = '/home/dennis/praveen/diva_multiscale/action_labels/train/0000/VIRAT_S_000000/1_Talking_0_59/clip_0/actions_39.pkl'
    anno_path = '/home/dennis/praveen/diva_multiscale/annotations/train/0000/VIRAT_S_000000/1_Talking_0_59/clip_0/clip_annotation.pkl'
    
    #clip = get_clip(clip_path)
    #print('clip', clip.size())
    #vutil.save_image(clip,"out/test.png", nrow=10, normalize=True, range=None)

    anno = get_anno(anno_path)
    print(anno.keys())
    print('len', len(anno))
    print('start_frame', anno[b'start_frame'])
    print('end_frame', anno[b'end_frame'])
    print('clip_bbox', anno[b'clip_bbox'])
    
    print('-----------------------')
    
    #for a in anno[b'actor_bboxes']:
    #    print(a, anno[b'actor_bboxes'][a])

def get_clip(clip_path):
    hf = h5py.File(clip_path, 'r')
    clip = hf.get('data')

    cliparr = np.array(clip) / 255.0
    clip = torch.from_numpy(cliparr).permute(0, 3, 1, 2)[:,[2,1,0],:,:]
    return clip


def get_anno(anno_path):
    anno = pickle.load(open(anno_path, 'rb'), encoding='bytes')
    return anno

def create_diva_annotation(root, mode):
    viddir = join(root, 'tubes', mode)
    anndir = join(root, 'action_labels', mode)
    #anndir = join(root, 'annotations', mode)

    filearr = []
    for d1 in os.listdir(viddir):
        print(d1)
        for d2 in os.listdir(join(anndir, d1)):
            for d3 in os.listdir(join(anndir, d1,d2)):
                for d4 in os.listdir(join(anndir, d1, d2, d3)):
                    for aname in os.listdir(join(anndir, d1, d2, d3, d4)):
                        apath = join(anndir, d1, d2, d3, d4, aname)
                        vd = join(viddir, d1, d2, d3, d4)
                        if not os.path.exists(vd):
                            continue
                        
                        vnlist = os.listdir(vd)
                        assert len(vnlist) == 1
                        vname = vnlist[0]
                        vpath = join(vd, vname)
                        
                        ann = [int(aid) for aid in get_anno(apath)]
                        clip = get_clip(vpath)
                        print(clip.size(0), vpath)
                        
                        filearr.append((join('tubes', mode, d1, d2, d3, d4, vname), clip.size(0), ann))
    
    # Write to file
    write2file(filearr, mode+'.txt')


def write2file(arr, fname):
    with open(fname, "w") as f:
        for name, nframe, aid in arr:
            f.write('{} {} {}\n'.format(name, nframe, functools.reduce(lambda x,y: str(x) + ' ' + str(y), aid)))

def test2():


if __name__ == "__main__":
    test2()
    #main()
    #create_diva_annotation('/home/dennis/praveen/diva_multiscale', 'train')
    #create_diva_annotation('/home/dennis/praveen/diva_multiscale', 'validation')
