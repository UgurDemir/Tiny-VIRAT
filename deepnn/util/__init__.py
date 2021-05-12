import numpy as np
import os
import os.path
import time
from collections import namedtuple
import torchvision.utils as vutil

def save_images(epoch, it, tag='./', nrow=3, r=None, **imgs):
    taggeddir = tag
    if not os.path.exists(taggeddir):
        os.makedirs(taggeddir)
    for i in imgs.keys():
        vutil.save_image(imgs[i],"%s/%d_%d%s.png"%(taggeddir,epoch,it,i),nrow=nrow,normalize=True, range=r)

def mkdirs(*dirlist):
    for d in dirlist:
        if not os.path.exists(d):
            os.makedirs(d)

def dict2namedtuple(dictionary, name='GenericDict'):
    NT = namedtuple(name, dictionary.keys())
    gen_dict = NT(**dictionary)
    return gen_dict
