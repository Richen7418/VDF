data_path=data
# data process
CUDA_VISIBLE_DEVICES=0 python utils.py --video_path data/May_45s.mp4 --save_path data/ori_imgs
# defense
CUDA_VISIBLE_DEVICES=0 python fft_ifftmask.py --data_path data
