# VDF
Video Defense Framework (VDF)官方代码
# 配置环境
```bash
# 如果是在国内位了更快的pip下载速度请换源清华源等
conda env create -f environment.yml
```
准备 face-parsing 模型，可能需要手动下载
```bash
wget https://github.com/YudongGuo/AD-NeRF/blob/master/data_util/face_parsing/79999_iter.pth?raw=true -O BiSeNet_model/79999_iter.pth
```
# 对指定视频进行防护
```bash
bash scripts/run_demo.sh
```
