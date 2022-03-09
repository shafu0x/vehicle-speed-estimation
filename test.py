import os
import torch
from efficientnet_pytorch import EfficientNet
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = '7'
# check if cuda is available
device = 'cuda' if torch.cuda.is_available() else 'cpu'


# you can find a pretrained model at model/b3.pth
MODEL_F = '/home/ljw/projects/vehicle-speed-estimation/model/b0.pth'
# directory with the numpy optical flow images you want to use for inference
OF_NPY_DIR = '/data_4T/EV_RAFT/opticalflow/'



V = 0     # what version of efficientnet did you use
IN_C = 2  # number of input channels
NUM_C = 1 # number of classes to predict

model = EfficientNet.from_pretrained(f'efficientnet-b{V}', in_channels=IN_C, num_classes=NUM_C)
state = torch.load(MODEL_F)
model.load_state_dict(state)
model.to(device)


def inference(of_f):
    of = np.load(of_f)
    i = torch.from_numpy(of).to(device)
    pred = model(i)
    del i
    torch.cuda.empty_cache()
    return pred

# loop over all files in directory and predict
for f in Path(OF_NPY_DIR).glob('*.npy'):
    y_hat = inference(f).item()
    print(f'{f.name}: {round(y_hat, 2)}')