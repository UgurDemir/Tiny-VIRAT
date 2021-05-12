import os
from os.path import join
from sklearn.model_selection import train_test_split
from skvideo.io import vread

import cv2

root = '/home/ugur/Documents/dataset/ucf101'
datadir = os.path.join(root, 'videos')

def m1():
    splitfile = os.path.join(root, '/home/ugur/Documents/dataset/ucf101/settings/val_rgb_split1.txt')
    cliplist = []
    with open(splitfile, "r") as sf:
        for line in sf:
            filename, classid = line.replace('\n','').replace('\r', '').split(" ")
            filepath = os.path.join(datadir, filename)
            cap = cv2.VideoCapture(filepath)
            length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            cliplist.append((filename, length, int(classid)))

    train, val = train_test_split(cliplist, test_size=0.05)
    with open("train.txt", "w") as f:
        for name, nframe, aid in train:
            f.write('{} {} {}\n'.format(name, nframe, aid))

    with open("val.txt", "w") as f:
        for name, nframe, aid in val:
            f.write('{} {} {}\n'.format(name, nframe, aid))


def m2():
    train_val_split = os.path.join(root, '/home/ugur/Documents/dataset/ucf101/settings/train_rgb_split1.txt')
    test_split = os.path.join(root, '/home/ugur/Documents/dataset/ucf101/settings/val_rgb_split1.txt')

    # Train and validation
    trainval = []
    with open(train_val_split, "r") as sf:
        for line in sf:
            filename, nframe, classid = line.replace('\n','').replace('\r', '').split(" ")
            cpath = filename.split('_')[1]
            if cpath == 'HandStandPushups': cpath = 'HandstandPushups'
            fpath = os.path.join(cpath,filename+'.avi')
            trainval.append((fpath, int(nframe), int(classid)))

    # Split into train and validation
    train, val = train_test_split(trainval, test_size=0.1)

    # Test
    test = []
    with open(test_split, "r") as sf:
        for line in sf:
            filename, nframe, classid = line.replace('\n','').replace('\r', '').split(" ")
            cpath = filename.split('_')[1]
            if cpath == 'HandStandPushups': cpath = 'HandstandPushups'
            fpath = os.path.join(cpath,filename+'.avi')
            test.append((fpath, int(nframe), int(classid)))
    
    def write2file(arr, fname):
        with open(fname, "w") as f:
            for name, nframe, aid in arr:
                f.write('{} {} {}\n'.format(name, nframe, aid))

    write2file(train, 'train.txt')
    write2file(val, 'val.txt')
    write2file(test, 'test.txt')

    print(len(train), len(val), len(test))

def write2file(arr, fname):
    with open(fname, "w") as f:
        for name, nframe, aid in arr:
            f.write('{} {} {}\n'.format(name, nframe, aid))

def hmdb():
    root = '/home/ugur/Documents/dataset/hmdb51/'
    annot_dir = join(root, 'testTrainMulti_7030_splits')
    split_id = '1'
    outdir = join('reader/hmdb51/', split_id)

    train = []
    test = []
    classids = []

    class_id = 0
    for cfile_path in sorted(os.listdir(annot_dir)):
        if cfile_path[-5] != split_id: continue
        
        class_name = cfile_path[:-16]
        classids.append((class_name, class_id))
        with open(join(annot_dir, cfile_path), 'r') as cfile:
            for line in cfile:
                vidname, vid, _ = line[:-1].replace('\n','').replace('\r', '').split(" ")
                video_path = join(class_name, vidname)
                raw_vid = vread(join(root, 'hmdb51_org', video_path))
                vid_lenght = raw_vid.shape[0]
                # Training set
                if vid == '1':
                    train.append((video_path, vid_lenght, int(class_id)))
                # Test
                elif vid == '2':
                    test.append((video_path, vid_lenght, int(class_id)))

        class_id += 1
        print(class_name, class_id)
    

    write2file(train, join(outdir, 'train.txt'))
    write2file(test, join(outdir, 'test.txt'))

    with open(join(outdir,'classname.txt'), "w") as f:
        for cname, cid in classids:
            f.write('{} {}\n'.format(cname, cid))

    for l in train:
        print(l)
    
    for l in test:
        print(l)

    print('train', len(train))
    print('test', len(test))

if __name__ == "__main__":
    hmdb()
