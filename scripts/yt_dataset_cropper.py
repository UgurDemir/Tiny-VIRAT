import os
from os.path import join
import random
import sys

import cv2
import numpy as np

SOURCE = '/home/ugur/Desktop/youtube-dl/v2'
DESTINATION = '/home/ugur/Desktop/yt_dst'
PRE_SCALE_RATIO = 0.5
CROP_HEIGHT = 128
CROP_WIDTH = 128
CROP_FRAME = (80, 300)

def crop_video_cube_and_save(vcap, path, fourcc, fps, frame_ids, height, width, pre_scale):
    hh = height[1]- height[0]
    ww = width[1]-width[0]
    out = cv2.VideoWriter(path,fourcc, fps, (hh, ww), True)
    for i, f in enumerate(frame_ids):
        vcap.set(1, f)
        ret, frame_bgr = vcap.read()
        if ret:
            frame_bgr_resized = cv2.resize(frame_bgr, pre_scale)
            frame_bgr_cropped = frame_bgr_resized[height[0]:height[1], width[0]:width[1], :]
            out.write(frame_bgr_cropped)
        else:
            # TODO: Add error check
            print("Skipped! ", path, frame_ids)
    
    out.release()

def create(root, vid_list):
    total = 0
    for vid_name in vid_list:
        vid_path = join(root, vid_name)
        cap = cv2.VideoCapture(vid_path)

        # Video information
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        orig_width = int(cap.get(3))
        orig_height = int(cap.get(4))
        fps = cap.get(5)
        fourcc = cv2.VideoWriter_fourcc(*'MJPG') #int(cap.get(6))

        height = int(PRE_SCALE_RATIO * orig_height)
        width = int(PRE_SCALE_RATIO * orig_width)


        ncubes = (length//CROP_FRAME[1]) * 16
        total += ncubes

        vid_dir = join(DESTINATION, vid_name.replace(' ', '').replace('-','').replace('.', ''))
        mkdirs(vid_dir)
        print(vid_dir, length)
        
        for cid in range(ncubes):
            # If video is too short, skip the video
            if length < CROP_FRAME[0]: continue

            # Clip selection
            sf = random.randint(0, length - CROP_FRAME[1] - 1)
            fl = random.randint(CROP_FRAME[0], CROP_FRAME[1])
            sh = random.randint(0, height-CROP_HEIGHT)
            sw = random.randint(0, width-CROP_WIDTH)
            

            clip_path = join(vid_dir, '{:05}.avi'.format(cid))

            # Crop the selected cube
            crop_video_cube_and_save(cap, path=clip_path, 
                                    fourcc = fourcc,
                                    fps = fps,
                                    frame_ids = range(sf, sf+fl), 
                                    height = (sh, sh+CROP_HEIGHT), 
                                    width =(sw, sw+CROP_WIDTH), 
                                    pre_scale = (width, height))
            print('    -{} is cropperd, [{}, {}, {}]'.format(clip_path.split('/')[-1], fl, sh, sw))
        
        
        
        cap.release()
    print('total', total)


def mkdirs(*dirlist):
    for d in dirlist:
        if not os.path.exists(d):
            os.makedirs(d)

def list_vids():
    for vid_name in os.listdir(SOURCE):
        print('"{}",'.format(vid_name))



VIDLIST = [
    "UNTITLED SEQUENCES-193873193.mp4",
    "4K polyflowpi 1-136103202.mp4",
    "I Love the Ocean - Long Version - GH4 4K + SLR Magic Anamorphot-97692560.mp4",
    "Denver Broncos 2016 Season in 4K-198634890.mp4",
    "Alhambra 4K-214367017.mp4",
    "STAVANGER _ NORWAY (4K)-167892347.mp4",
    "Osnabrück in 4k-175198558.mp4",
    "The Future Medicine 4K Corporate Package (After Effects Project Files)-159455925.mp4",
    "VIENNA from above _ DJI Phantom 3 Professional [ 4K]-133842385.mp4",
    "Endless Music Visualization 4K Project (After Effects Project Files)-162166758.mp4",
    "AJA Ki Pro Quad - Efficient 4K workflows.-40439273.mov",
    "BeSteady Four _ BlackMagic Production Camera 4K Tests-87389090.mp4",
    "TOKYO REVERSE - TRAILER-89936769.mp4",
    "De Slufter van boven - Texel in 4K-163736142.mp4",
    "Glidecam 4K - Flight of the Shogun-115829238.mp4",
    "SNOWFLAKES 4K OPENERS (After Effects Project Files)-145096806.mp4",
    "GOLDEN VIEW 4K (After Effects Project Files)-148058982.mp4",
    "Peekskill At Night (Blackmagic 4k Production Camera low-light test)-91555886.mp4",
    "Bella Italia pt2 4K-174952003.mp4",
    "Voornes Duin - Phantom 4K-160578133.mp4",
    "BOKK (loop)-162670765.mp4",
    "Takashi + Ayumi wedding story same day edit 4k GH4 OSMO-146484162.mp4",
    "4K Cinema Camera - With Voiceover-121649159.mp4",
    "One Day in Istanbul 4K-181180995.mp4",
    "NORWAY TRIP (4K)-175824196.mp4",
    "Experimenting with particles in Krakatoa(Nebulas, Energy dust trail, Comet, Fiery formation, Space Sands)-114053015.mp4",
    "Arashiyama 4k-142480565.mp4",
    "NEWS 24 4K Broadcast package (After Effects Project Files)-150225201.mp4",
    "Preikestolen (pulpit Rock aerial)  i 4K (public)-169958461.mp4",
    "Stephon Karolus Reel 2014 UHD 4K-117887224.mp4",
    "De Groene Jonker van boven - 4K-165643973.mp4",
    "AJA CION NAB 2015 Reel-124373479.mov",
    "Philadelphia in 4K-137203743.mp4",
    "Baby Coot's in 4K - Baby Meerkoet in 4K-125621327.mp4",
    "ICELAND in 4K-189872577.mp4",
]


def main():
    if len(sys.argv) != 2: raise Exception('Please see usage')

    tg = sys.argv[1]
    if tg == '1':
        create(SOURCE, VIDLIST[:9])
    elif tg == '2':
        create(SOURCE, VIDLIST[9:18])
    elif tg == '3':
        create(SOURCE, VIDLIST[18:27])
    elif tg == '4':
        create(SOURCE, VIDLIST[27:])
    else:
        raise Exception('Unknown option')
    print('FINISHED')

if __name__ == "__main__":
    main()