import os
from os.path import join

from PIL import Image

root = '/home/ugur/Documents/dataset/ucf101/frame-32'
outdir = '/home/ugur/Documents/dataset/ucf101/bicubicx4-128'
scale_factor = 4
mode = Image.BICUBIC

def mkdirs(*dirlist):
    for d in dirlist:
        if not os.path.exists(d):
            os.makedirs(d)

for class_name in os.listdir(root):
    for iname in os.listdir(join(root, class_name)):
        resized_dir = join(outdir, class_name, iname)
        mkdirs(resized_dir)
        for fname in os.listdir(join(root, class_name, iname)):
            img = Image.open(join(root, class_name, iname, fname))
            w, h = img.size
            rimg = img.resize((w*scale_factor, h*scale_factor), resample=mode)
            fpath = join(resized_dir, fname)
            rimg.save(fpath, "PNG")
    print(class_name)
