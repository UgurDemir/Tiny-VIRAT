import torch

import h5py
import numpy as np
import cv2
import sys
import os
import os.path
from os.path import join

UCF_VID_ROOT = '/home/ugur/Documents/dataset/ucf101/videos'
UCF_OF_ROOT = '/home/ugur/Documents/dataset/ucf101/opt_flow'

HMDB_VID_ROOT = '/home/ugur/Documents/dataset/hmdb51/hmdb51_org'
HMDB_OF_ROOT = '/home/ugur/Documents/dataset/hmdb51/opt_flow'

optical_flow = cv2.DualTVL1OpticalFlow_create()
optical_flow.setTau(0.25)
optical_flow.setLambda(0.1)
optical_flow.setTheta(0.3)
optical_flow.setScalesNumber(5)
optical_flow.setWarpingsNumber(5)
optical_flow.setEpsilon(0.005)
#optical_flow.setNumIterations(300)
#optical_flow.setInnerIterations(300)
#optical_flow.setOuterIterations(300)
optical_flow.setUseInitialFlow(False)

def calculate_of(vid_path):
    cap = cv2.VideoCapture(vid_path)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    vid_flow = []
    # Read the initial frame
    ret1, frame1 = cap.read()
    # Calculate optical flow for a video
    for i in range(length-1):
        # Read next frame
        ret2, frame2 = cap.read()

        # Calculate optical flow
        flow_frame = optical_flow.calc(cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY), cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY), None)
        #print(flow_frame.shape)

        # Add calculated flow to ourput arr
        vid_flow.append(flow_frame)

        # Set new current frame
        frame1 = frame2
        
    cap.release()
    vid_flow = np.array(vid_flow)
    return vid_flow

def save_of(flow, flow_dir, flow_name):
    if not os.path.exists(flow_dir):
        os.makedirs(flow_dir)

    with h5py.File(join(flow_dir, flow_name), 'w') as hf:
        hf.create_dataset("flow", data=flow)

def load_of(flow_path):
    clip = np.array(h5py.File(flow_path, 'r').get('flow'))
    return clip

def ucf101(vid_root, of_root):
    for act_name in os.listdir(vid_root):
        for vid_name in os.listdir(join(vid_root, act_name)):
            # Calculate flow
            flow = calculate_of(join(vid_root, act_name, vid_name))

            # Save flow file
            save_of(flow, join(of_root, act_name), vid_name.split('.')[0]+'.h5')

            print(vid_name)

def process_action(vid_root, act_name, of_root):
    for vid_name in os.listdir(join(vid_root, act_name)):
        # Calculate flow
        flow = calculate_of(join(vid_root, act_name, vid_name))

        # Save flow file
        save_of(flow, join(of_root, act_name), vid_name.split('.')[0]+'.h5')

        print(vid_name, flush=True)


def main():
    print(sys.argv, flush=True)

    dset_name = sys.argv[1]
    action_id = int(sys.argv[2])

    assert len(sys.argv) == 3
    assert dset_name in ['ucf101', 'hmdb51']

    if dset_name == 'ucf101':
        act_name = sorted(os.listdir(UCF_VID_ROOT))[action_id]
        process_action(UCF_VID_ROOT, act_name, UCF_OF_ROOT)
    elif dset_name == 'hmdb51':
        act_name = sorted(os.listdir(HMDB_OF_ROOT))[action_id]
        process_action(HMDB_VID_ROOT, act_name, HMDB_OF_ROOT)
    else:
        raise Exception('Unknown dataset name ' + str(dset_name))
    
    print('Finished', flush=True)

import flowlib as flowlib 
def visualize(flow_path):
    flow = load_of(flow_path)
    print(flow.shape)
    flow_img = flowlib.flow_to_image(flow[0,:,:,:])
    print(flow_img.shape)
    
    cv2.imwrite('flow1.png',flow_img)



if __name__ == '__main__':
    UCF_VID_ROOT = '/home/ugur/Documents/dataset/ucf101/videos'
    UCF_OF_ROOT = '/home/ugur/Documents/dataset/ucf101/opt_flow'
    ucf101(UCF_VID_ROOT, UCF_OF_ROOT)
    #visualize('/home/ugur/Documents/dataset/ucf101/opt_flow/SkyDiving/v_SkyDiving_g02_c03.h5')
    #main()