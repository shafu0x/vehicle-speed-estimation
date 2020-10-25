import os
from PIL import Image
import cv2


def video_to_frames(video_f, out_dir, n_to_skip=0, verbose=True):
    'Split `video_f` into frames and save to `out_dir`.'
    i = 0
    n_img_saved = 0
    cap = cv2.VideoCapture(video_f)
    while(cap.isOpened()):
        try:
            ret, frame = cap.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if i % (n_to_skip+1) == 0: 
                img = Image.fromarray(frame)
                img.save(os.path.join(out_dir, str(i) + '.png'))
                if verbose: print(f'n_img_saved: {n_img_saved}')
                n_img_saved += 1
        except Exception as e: return
        i += 1

def videos_to_frames(video_dir, out_dir, skip):
    print(SZ)
    for f in os.listdir(video_dir):                                                                      
        video_f = os.path.join(video_dir, f)
        print(video_f)
        video_out_dir = os.path.join(out_dir, os.path.splitext(f)[0])
        if not os.path.exists(video_out_dir): os.mkdir(video_out_dir)
        else: continue
        video_to_frames(video_f, video_out_dir, int(skip), False)

if __name__ == '__main__':
    v_d = '/home/sharif/Documents/commai-challenge/data/videos/train.mp4'
    o_d = '/home/sharif/Documents/commai-challenge/data/frames/train'
    video_to_frames(v_d, o_d)
