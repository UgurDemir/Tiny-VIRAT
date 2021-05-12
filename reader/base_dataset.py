import cv2
import os
from os.path import join
import random
import math
import torch
import numpy as np
from skvideo.io import vread
import pickle
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
import h5py

class BaseDataset(Dataset):
    """
    Args:
        root: root dir of UCF-101 dataset
        split_file: File which list all video names and action classes.
        out_shape: [C,T,H,W] | C:channel, T:temporal, H:height, W:width
        scale_factor: downsampling ratio for super-resolution
        scale_mode: inerpolation mode (please check torch.nn.functional.interpolation(...))
        frame_skip_rate: sampling ratio for data augmentation
        p_skip: probability of applying frame_skip_rate
        augment: For train set True, otherwise False
        norm: normalization type [-11|01|zn]
            None: don't apply
            -11: scales intensities into range -1 to 1
             01: scales intensities into range 0 to 1
             zn: (x - mean) / std
        dtype: output tensor type
        produce: list of requested data options:
            'orig': original image with specified size
            'down': downscaled version of orig
            'up': upscaled version of down
            'label': action labels
        prog_size: [nframe, h, w]
        file_type: std | h5
    """
    def __init__(self, cliplist, scale_size=128, crop_size=112, nframe=16, 
                frame_skip_rate=2, p_skip=0.5, p_flip=0.5, augment = True,
                scale_factor=4, upscale_mode='bicubic', downscale_mode='bicubic', norm='-11', 
                produce=['down', 'orig', 'label'], prog_size=None, testmode=False, dtype=torch.float32):
        self.dtype = dtype
        self.inch = 3
        self.scale_size = scale_size
        self.crop_size = crop_size
        self.nframe = nframe
        self.scale_factor = scale_factor
        self.norm = norm
        self.cliplist = cliplist
        self.augment = augment
        self.produce = produce
        self.prog_size = prog_size
        self.testmode = testmode
        self.upscale_mode = str2PILInterpolation(upscale_mode)
        self.downscale_mode = str2PILInterpolation(downscale_mode)

        self.frame_skip_rate = frame_skip_rate
        self.p_skip = p_skip
        self.p_flip = p_flip

    def __len__(self):
        return len(self.cliplist)

    def normalize(self, x):
        if self.norm == '-11':
            #x = ((x / 255.0) - 0.5)*2.0
            x = (x - 0.5) / 0.5
        elif self.norm == '01':
            #x = x / 255.0
            pass
        elif self.norm == 'zn':
            raise Exception('Specified normalization is not implemented')
        elif self.norm is None:
            pass
        else:
            raise Exception('Unknown normalization type ' + str(self.norm))

        return x

    def denormalize(self, x):
        if self.norm == '-11':
            y = (x + 1.0) / 2.0
        elif self.norm == '01':
            pass
        elif self.norm == 'zn':
            raise Exception('Specified normalization is not implemented')
        elif self.norm is None:
            pass
        else:
            raise Exception('Unknown normalization type ' + str(self.norm))

        return y

    def get_frame_ids(self, length):
        if self.augment:
            # Skip frame
            if length >= (self.nframe * self.frame_skip_rate):
                skip_frame = (random.uniform(0,1) <= self.p_skip)
            else:
                skip_frame = False
            fsr = self.frame_skip_rate if skip_frame else 1

            # Start frame
            if length > (self.nframe * fsr):
                start_frame = random.randint(0, length-self.nframe*fsr-1)
            else:
                start_frame = 0

            frame_ids = [i for i in range(start_frame, start_frame+self.nframe*fsr, fsr)]

        else:
            # Start frame
            if length >= self.nframe:
                start_frame = (length - self.nframe) // 2
            else:
                start_frame = 0  
            frame_ids = list(range(start_frame, start_frame+self.nframe))
        
        
        # Length Check
        if len(frame_ids) < self.nframe:
            frame_ids += [frame_ids[-1]] * (self.nframe - length)
        return frame_ids

    def read_video(self, path, vlength, frame_ids):
        # Read video file
        cap = cv2.VideoCapture(path)

        # Use same crop points for the whole squence
        sh = random.randint(0, self.scale_size-self.crop_size) if self.augment else None
        sw = random.randint(0, self.scale_size-self.crop_size) if self.augment else None
        doflip = (random.uniform(0,1) <= self.p_flip)

        outmap = {'orig': torch.zeros([self.inch, self.nframe, self.crop_size, self.crop_size], dtype=self.dtype)}
        if 'down': outmap['down'] = torch.zeros([self.inch, self.nframe, self.crop_size//self.scale_factor, self.crop_size//self.scale_factor], dtype=self.dtype)
        if 'up': outmap['up'] = torch.zeros([self.inch, self.nframe, self.crop_size, self.crop_size], dtype=self.dtype)
        if self.prog_size is not None: outmap['prog'] = torch.zeros([self.inch, self.prog_size[0], self.prog_size[1], self.prog_size[2]], dtype=self.dtype)

        prog_i = 0
        frame_pointer = frame_ids[0]
        cap.set(1, frame_pointer)
        ret, frame_bgr = cap.read()
        for i, f in enumerate(frame_ids):
            while f != frame_pointer:
                ret, frame_bgr = cap.read()
                frame_pointer += 1

            if ret:
                frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                frame = torch.from_numpy(frame)
                # HWC2CHW
                frame = frame.permute(2, 0, 1)
                
                # Preprocess
                frame_pil = transforms.functional.to_pil_image(frame)
                frame_pil = transforms.functional.resize(frame_pil, self.scale_size)
                if self.augment:
                    frame_pil = transforms.functional.crop(frame_pil, sh, sw, self.crop_size, self.crop_size)
                    if doflip:
                        frame_pil = transforms.functional.hflip(frame_pil)
                else:
                    frame_pil = transforms.functional.center_crop(frame_pil, self.crop_size)

                # Create down sampled version
                if 'down' in outmap or 'up' in outmap:
                    frame_down_pil = transforms.functional.resize(frame_pil, self.crop_size//self.scale_factor, interpolation=self.downscale_mode)
                    outmap['down'][:, i, :, :] = transforms.functional.to_tensor(frame_down_pil)

                # Create up sampled version
                if 'up' in outmap:
                    frame_up_pil = transforms.functional.resize(frame_down_pil, self.crop_size, interpolation=self.upscale_mode)
                    outmap['up'][:, i, :, :] = transforms.functional.to_tensor(frame_up_pil)

                if self.prog_size is not None and i % (len(frame_ids)//self.prog_size[0]) == 0:
                    prog_pil = transforms.functional.resize(frame_pil, [self.prog_size[1], self.prog_size[2]], interpolation=self.upscale_mode)
                    outmap['prog'][:, prog_i, :, :] = transforms.functional.to_tensor(prog_pil)
                    prog_i += 1

                outmap['orig'][:, i, :, :] = transforms.functional.to_tensor(frame_pil)
            else:
                # TODO: Add error check
                print("Skipped! ", path, vlength, frame_ids)
                break

        cap.release()
        return outmap

    def __getitem__(self, idx):
        # Read annotations
        ann = self.cliplist[idx]
        clippath, vlength, class_id = ann['path'], ann['length'], ann['label']

        # Select video segment
        if self.testmode:
            start_frame = ann['start_frame']
            frame_ids = list(range(start_frame, start_frame + self.nframe))
        else:
            frame_ids = self.get_frame_ids(vlength)

        # Read frames from selected video
        outmap = self.read_video(clippath, vlength=vlength, frame_ids=frame_ids)

        # Intensity normalization
        for p in outmap:
            outmap[p] = self.normalize(outmap[p])

        if 'label' in self.produce:
            outmap['label'] = torch.Tensor(class_id).float() if type(class_id) == list else class_id
        
        outlist = [outmap[k] for k in self.produce]

        # If it is testmode, add necessary information for stiching
        if self.testmode:
            outlist = [*outlist, ann['id'], ann['tag']]

        return outlist

def str2PILInterpolation(mode):
    if mode == 'bicubic':
        return Image.BICUBIC
    elif mode == 'bilinear':
        return Image.BILINEAR
    elif mode == 'nearest':
        return Image.NEAREST
    elif mode == 'LANCZOS':
        return Image.LANCZOS
    else:
        raise Exception('Unknown scale mode: ' + str(mode))