import cv2
import os
import tqdm
import numpy as np
from skimage.metrics import structural_similarity as ssim
import re
import argparse
import subprocess
import pdb

def extract_audio(path, out_path, sample_rate=16000):
    print(f'[INFO] ===== extract audio from {path} to {out_path} =====')
    cmd = f'ffmpeg -i {path} -f wav -ar {sample_rate} {out_path}'
    os.system(cmd)
    print(f'[INFO] ===== extracted audio =====')

def extract_imgs(video_path, save_path, fps=25):

    print(f'[INFO] ===== extract images from {video_path} to {save_path} =====')
    cmd = f'ffmpeg -i {video_path} -vf fps={fps} -qmin 1 -q:v 1 -start_number 0 {os.path.join(save_path, "%d.jpg")}'
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

def combine_images_and_audio_to_video(adv_images_path, audio_path, output_video, key_word='mouth', fps=25):
    """
    从图片序列和音频合成视频（支持自定义文件名格式）
    """
    print(f'[INFO] ===== combine images and audio to {output_video} =====')
    
    # 获取并按数字排序所有PNG文件
    image_files = sorted(
        [f for f in os.listdir(adv_images_path) if f.endswith('.png') and key_word in f ],
        key=lambda x: int(re.search(r'(\d+)_', x).group(1))  # 提取文件名中的数字
    )
    # pdb.set_trace()
    # 临时重命名文件为FFmpeg可识别的数字序列（0.png, 1.png...）
    temp_files = []
    for idx, old_name in enumerate(image_files):
        new_name = f"{idx:06d}.png"
        old_path = os.path.join(adv_images_path, old_name)
        new_path = os.path.join(adv_images_path, new_name)
        
        # pdb.set_trace()
        os.rename(old_path, new_path)
        temp_files.append((new_path, old_path))  # 记录临时文件，后续恢复
    
    # 构建FFmpeg命令（使用%d.png）
    cmd = [
        'ffmpeg',
        '-y',
        '-framerate', str(fps),
        '-i', os.path.join(adv_images_path, '%06d.png'),
        '-i', audio_path,
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv444p',
        '-crf', '0',
        '-preset', 'veryslow',
        # '-shortest',
        output_video
    ]
    
    try:
        subprocess.run(cmd, check=True)
        print(f'[INFO] Successfully saved video to {output_video}')
    except subprocess.CalledProcessError as e:
        print(f'[ERROR] FFmpeg error: {e}')
    finally:
        # 恢复原始文件名
        for new_path, old_path in temp_files:
            if os.path.exists(new_path):
                os.rename(new_path, old_path)

    
def Preprocess(video_path, audio_path, image_path):
    if not os.path.exists(image_path):
        os.makedirs(image_path)
    extract_audio(video_path, audio_path)
    extract_imgs(video_path, image_path)
    calculate_ssim(image_path)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='Preprocess or Reconstruction', help='mode')
    parser.add_argument('--video_path', type=str, default='./data/Obama1_45s.mp4', help='video_path')
    parser.add_argument('--audio_path', type=str, default='./data/audio.wav', help='audio_path')
    parser.add_argument('--image_path', type=str, default='./data/ori_imgs', help='image_path')
    parser.add_argument('--adv_images_path', type=str, default='./data/ifftmask_fft_adv_imgs', help='adv_image_path')
    parser.add_argument('--key_word', type=str, default='mouth', help='key_word')
    parser.add_argument('--output_video', type=str, default='./data/adv_output.mp4', help='output_video')
    args = parser.parse_args()
    mode = args.mode

    if mode == 'Preprocess':
        video_path = args.video_path
        audio_path = args.audio_path
        image_path = args.image_path
        Preprocess(video_path, audio_path, image_path)
    elif mode == 'Reconstruction':
        adv_images_path = args.adv_images_path
        audio_path = args.audio_path
        output_video = args.output_video
        key_word = args.key_word
        combine_images_and_audio_to_video(adv_images_path, audio_path, output_video, key_word)
    else:
        print('Please input correct mode')

