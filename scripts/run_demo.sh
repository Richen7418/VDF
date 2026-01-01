data_path=data
video_name=May_45s.mp4
# data process
CUDA_VISIBLE_DEVICES=0 python utils.py --mode Preprocess --video_path ${data_path}/${video_name} --audio_path ${data_path}/audio.wav --image_path ${data_path}/ori_imgs 
# defense
CUDA_VISIBLE_DEVICES=0 python fft_ifftmask.py --data_path $data_path
# Video compositing
CUDA_VISIBLE_DEVICES=0 python utils.py --mode Reconstruction --adv_images_path ${data_path}/ifftmask_fft_adv_imgs/full_face/epsilon_0.05/ --audio_path ${data_path}/audio.wav --output_video ${data_path}/adv_output.mp4  --key_word full_face
