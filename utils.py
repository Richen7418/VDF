import cv2
import os
import tqdm
import numpy as np
from skimage.metrics import structural_similarity as ssim
import re
import argparse

def extract_imgs(video_path, save_path, fps=25):

    print(f'[INFO] ===== extract images from {video_path} to {save_path} =====')
    cmd = f'ffmpeg -i {video_path} -vf fps={fps} -qmin 1 -q:v 10 -start_number 0 {os.path.join(save_path, "%d.jpg")}'
    os.system(cmd)
    print(f'[INFO] ===== extracted images =====')

def extract_num(filename):
    # 提取文件名前面的数字部分
    match = re.match(r"(\d+)", filename)
    return int(match.group(1)) if match else -1

def calculate_ssim(target_path):
    
    image_path = target_path
    image_names = [f for f in os.listdir(image_path) if f.endswith('.jpg') or f.endswith('.png')]
    image_names = sorted(image_names, key=extract_num)
    score_list = []
    for i in tqdm.tqdm(range(len(image_names)-1), desc='Calculating SSIM'):
        img1 = cv2.imread(os.path.join(image_path, image_names[i]))
        img2 = cv2.imread(os.path.join(image_path, image_names[i+1]))
        score = ssim(img1, img2, channel_axis=-1, data_range=255)
        score_list.append(score)
    score_np = np.array(score_list)
    save_path = os.path.join(target_path, 'ssim_score.npy')
    np.save(save_path, score_np)
    return
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_path', type=str, default='./data/May_45s.mp4', help='video_path')
    parser.add_argument('--save_path', type=str, default='./data/ori_imgs', help='save_path')
    args = parser.parse_args()
    video_path = args.video_path
    save_path = args.save_path
    extract_imgs(video_path, save_path)
    calculate_ssim(save_path)
