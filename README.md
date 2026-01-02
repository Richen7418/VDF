# VDF

Official implementation of **Video Defense Framework (VDF)**.

## Environment Setup
```bash
# For users in mainland China, it is recommended to switch pip to a mirror
# (e.g., Tsinghua mirror) to achieve faster download speeds.
conda env create -f environment.yml
```
Prepare face-parsing model.
```bash
wget https://github.com/YudongGuo/AD-NeRF/blob/master/data_util/face_parsing/79999_iter.pth?raw=true -O BiSeNet_model/79999_iter.pth
```
## Protect a Given Video
```bash
bash scripts/run_demo.sh
```
## TODO
Demo videos and Huggingface Space are coming soon.
