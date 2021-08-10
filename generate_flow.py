import sys
sys.path.append('core')

import os
import numpy as np
import torch
import cv2
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
import argparse
from pathlib import Path

from utils import flow_viz
from utils.raft import RAFT
from utils.utils import InputPadder

os.environ['CUDA_VISIBLE_DEVICES'] = '5'
DEVICE = 'cuda' if torch.cuda.is_available else 'cpu'

images_dir = '/home/ljw/projects/vehicle-speed-estimation/data/test/'
output_dir = '/data_4T/EV_RAFT/of_kitti/'
outfig_dir = '/home/ljw/projects/vehicle-speed-estimation/result/vis_kitti/'
model_path = '/home/ljw/projects/vehicle-speed-estimation/model/raft_models/raft-kitti.pth'
videos_path = '/home/ljw/projects/vehicle-speed-estimation/speedchallenge/data/'



def generate_frames(videos_path, images_dir):
    vidcap = cv2.VideoCapture(videos_path+'test.mp4')  

    frames_rate=vidcap.get(5)
    frame_num = int(vidcap.get(7))
    print("frame rate", frames_rate)

    for i in tqdm(range(frame_num)):
        success,image = vidcap.read()
        if image.size == 0: 
            pass
        else:
            cv2.imwrite(images_dir+"frame%d.png" % i, image)



def vis(npy_dir, output_dir):
    npy_dir = Path(npy_dir)
    output_dir = Path(output_dir)

    npy_files = list(npy_dir.glob('*.npy'))

    for i, npy_file in enumerate(npy_files):
        f = str(npy_file)
        of = np.load(f)
        of = torch.from_numpy(of)
        of = of[0].permute(1,2,0).numpy()
        of = flow_viz.flow_to_image(of)
        img = Image.fromarray(of)
        output_f = output_dir / npy_file.stem
        output_f = str(output_f) + '.jpg'
        img.save(output_f)

        if i % 20 == 0: print(f'{i}/{len(npy_files)}')

def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)

def run(args):
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(model_path))

    model = model.module
    model.to(DEVICE)
    model.eval()

    output_dir = Path(args.output_dir)
    images_dir = Path(args.images_dir)
    images = list(images_dir.glob('frame*.png'))

    with torch.no_grad():
        images = sorted(images)

        #for i in tqdm(range(len(images)-1)):
        # run first 200
        for i in tqdm(range(200)):
            im_f1 = str(images[i])
            im_f2 = str(images[i+1])
            
            image1 = load_image(im_f1)
            image2 = load_image(im_f2)

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)

            flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)

            # 2.4 MB each npy
            of_f_name = output_dir / f'{i}.npy' 
            np.save(of_f_name, flow_up.cpu())
    
    if args.out_vis:
        vis(output_dir, outfig_dir)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--images_dir',
                        default=os.path.abspath(images_dir),
                        help="directory with your images")
    parser.add_argument('--output_dir',
                        default=os.path.abspath(output_dir),
                        help="optical flow images will be stored here as .npy files")
    parser.add_argument('--out_vis',
                        action='store_true',
                        help="output optical flow images")
    args = parser.parse_args()
    # transfer videos to frames
    # generate_frames(videos_path,images_dir)

    run(args)
    

