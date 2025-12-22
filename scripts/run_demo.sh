data_path=data
video_name=May_45s.mp4
# data process
CUDA_VISIBLE_DEVICES=0 python utils.py --video_path ${data_path}/${video_name} --save_path ${data_path}/ori_imgs
# defense
CUDA_VISIBLE_DEVICES=0 python fft_ifftmask.py --data_path $data_path
