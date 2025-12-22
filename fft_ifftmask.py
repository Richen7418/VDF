import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
from BiSeNet_model.model import BiSeNet
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import configargparse
import os.path as osp
import tqdm
import cv2
from pathlib import Path
import pdb
import torch.fft as fft
from torchvision import models, transforms
import re
import json
from pytorch_msssim import ssim as SSIM
import time
from datetime import timedelta

def differentiable_resize(tensor, size, mode='bilinear', align_corners=False):
    """
    可微的张量resize函数
    
    参数:
        tensor: 输入张量 (N, C, H, W)
        size: 目标大小 (height, width) 或缩放因子
        mode: 插值方法 'nearest' | 'bilinear' | 'bicubic' | 'area'
        align_corners: 是否对齐角落
    """
    # 如果size是缩放因子
    if isinstance(size, (float, int)):
        h, w = tensor.shape[2:]
        new_h = int(h * size)
        new_w = int(w * size)
        size = (new_h, new_w)
    
    # 使用插值函数
    return F.interpolate(tensor, size=size, mode=mode, align_corners=align_corners)
def multi_pt_resize(image):
    
    image_15times = differentiable_resize(image, size=1.5, mode='bilinear', align_corners=False)
    image_2times = differentiable_resize(image, size=2, mode='bilinear', align_corners=False)
    image_75_perc = differentiable_resize(image, size=0.75, mode='bilinear', align_corners=False)
    image_50_perc = differentiable_resize(image, size=0.50, mode='bilinear', align_corners=False)

    return image_15times, image_2times, image_75_perc, image_50_perc

class DifferentiableQuantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return torch.clamp((x * 255).round(), 0, 255)
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output 
    
def get_mask(images, net, source_region):
    """
    images: [B, C, H, W] 或 [C, H, W]
    返回: [B, H, W] 或 [H, W]
    """
    if images.dim() == 3:
        images = images.unsqueeze(0)
        squeeze_out = True
    else:
        squeeze_out = False

    with torch.no_grad():
        outputs = net(images) 
        preds = outputs.argmax(1).cpu().numpy()  

    if isinstance(source_region, (list, tuple)):
        masks = np.zeros_like(preds, dtype=bool)  
        for region in source_region:
            masks |= (preds == region)
    else:
        masks = (preds == source_region) 
    tensor_masks = torch.tensor(masks, dtype=torch.bool, device=images.device) 
    if squeeze_out:
        tensor_masks = tensor_masks.squeeze(0)
    return tensor_masks

def calulate_loss(images, net, masks, target_regions):
    if images.dim() == 3:
        images = images.unsqueeze(0)

    outputs = net(images) 

    B = images.size(0)
    losses = []
    for b in range(B):
        mask = masks[b]  
        if not mask.any():
            losses.append(torch.tensor(0.0, device=device))
            continue
        logits = outputs[b, :, mask]  
        probs = F.softmax(logits, dim=0)
        target_probs = 1 - torch.sum(probs[target_regions, :], dim=0)
        loss = torch.mean(torch.log(target_probs + 1e-4))
        losses.append(loss)
    return torch.stack(losses)  

def calulate_norm_loss(adv_image, image):
    pert_norm = torch.norm(adv_image-image, p=2)
    return pert_norm

def check_all_success(adv_image, adv_15times, adv_2times, adv_75_perc, adv_55_perc, net, batch_image_mask, 
                      image_15times_mask, image_2times_mask, image_75_perc_mask, image_55_perc_mask, target_regions, threshold=0.5):
    _, adv_success = check_attack_success(adv_image, net, batch_image_mask, target_regions, threshold=threshold)
    _, adv_15times_success = check_attack_success(adv_15times, net, image_15times_mask, target_regions, threshold=threshold)
    _, adv_2times_success = check_attack_success(adv_2times, net, image_2times_mask, target_regions, threshold=threshold)
    _, adv_75_perc_success = check_attack_success(adv_75_perc, net, image_75_perc_mask, target_regions, threshold=threshold)
    _, adv_55_perc_success = check_attack_success(adv_55_perc, net, image_55_perc_mask, target_regions, threshold=threshold)
    result = (adv_success and adv_15times_success and adv_2times_success and adv_75_perc_success and adv_55_perc_success)
    return result

def check_attack_success(adv_image, net, batch_image_mask, target_regions, threshold=0.9):
    """
    判断当前batch是否攻击成功（唇部区域被误分为非唇部的比例超过阈值）
    adv_image: [B, C, H, W]
    net: 分割模型
    batch_image_mask: [B, H, W]，唇部区域mask（bool）
    target_regions: list[int]，唇部类别标签
    threshold: 攻击成功比例阈值
    返回: bool，是否提前成功
    """
    
    with torch.no_grad():
        adv_output = net(adv_image)  
        adv_pred = adv_output.argmax(1)  
        ratios = []
        target_labels = torch.tensor(target_regions, device=adv_pred.device)
        for b in range(adv_pred.shape[0]):
            mask = batch_image_mask[b]  
            mouth_region_pred = adv_pred[b][mask] 
            if mouth_region_pred.numel() == 0:
                ratios.append(0.0)
                continue
            is_target = (mouth_region_pred.unsqueeze(-1) == target_labels).any(dim=-1)
            num_target = is_target.sum().item()
            num_total = mouth_region_pred.numel()
            ratio = num_target / (num_total + 1e-8)
            ratios.append(ratio)
        ratios = torch.tensor(ratios)

    success_flags = (ratios >= 1.0)
    mean_success = (ratios.mean() > threshold)
    return success_flags, mean_success

LABEL_MAP = {
    'background':0,  # 背景
    'face': 1,       # 面部
    'neck': 14,      # 颈部
    'body': 16,      # 躯干n
    'tongue': 11,    # 舌头
    'up_lip': 12,    # 上唇
    'down_lip': 13,   # 下唇
}

to_tensor = transforms.ToTensor()
to_pil = transforms.ToPILImage()
inv_normalize = transforms.Normalize(
    mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
    std=[1/0.229, 1/0.224, 1/0.225]
)
def transform_batch(batch_tensor):
    mean = torch.tensor([0.485, 0.456, 0.406], device=batch_tensor.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=batch_tensor.device).view(1, 3, 1, 1)
    return (batch_tensor - mean) / std

def inv_normalize_batch(batch_tensor):
    mean = torch.tensor([0.485, 0.456, 0.406], device=batch_tensor.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=batch_tensor.device).view(1, 3, 1, 1)
    return batch_tensor * std + mean

class SemanticLoss(nn.Module):
    def __init__(self, feature_layers=None, weights=None, device='cuda' if torch.cuda.is_available() else 'cpu'):
        super(SemanticLoss, self).__init__()
        
        # 加载预训练的VGG16模型
        self.vgg = models.vgg16(pretrained=True).features.to(device).eval()
        
        # 冻结所有参数，只作为特征提取器使用
        for param in self.vgg.parameters():
            param.requires_grad = False
        
        # 默认提取这些层的特征（低层到高层的语义信息）
        self.feature_layers = feature_layers or ['3', '8', '15', '22']
        self.weights = weights or [1.0, 0.75, 0.5, 0.25]

        self.features = {}
        self._register_hooks()
    
    def _register_hooks(self):
        """注册前向钩子来捕获指定层的输出"""
        def get_features(name):
            def hook(model, input, output):
                self.features[name] = output
            return hook
        
        for i, layer in enumerate(self.vgg):
            if str(i) in self.feature_layers:
                layer.register_forward_hook(get_features(str(i)))
    
    def forward(self, x, y):
        """
        计算两张图像之间的语义损失
        x: 原始图像或对抗图像 [batch, 3, H, W]
        y: 另一张图像 [batch, 3, H, W]
        """
        x_normalized = x
        y_normalized = y
        self.features = {}

        _ = self.vgg(x_normalized)
        features_x = {k: v for k, v in self.features.items()}
        
        _ = self.vgg(y_normalized)
        features_y = {k: v for k, v in self.features.items()}
        

        total_loss = 0.0
        for i, layer_name in enumerate(self.feature_layers):
            if layer_name in features_x and layer_name in features_y:

                loss = F.mse_loss(features_x[layer_name], features_y[layer_name], reduction='none')

                loss = loss.view(loss.size(0), -1).mean(dim=1)  # [B]
                total_loss += self.weights[i] * loss
        return total_loss 
    

def add_frequency_perturbation(image_tensor, perturbation):
    """
    支持batch的频域扰动函数。
    image_tensor: [B, C, H, W] 或 [C, H, W]
    perturbation: [B, 2, H, W] 或 [2, H, W]
    返回: [B, C, H, W] 或 [C, H, W]
    """
    if image_tensor.dim() == 3:
        image_tensor = image_tensor.unsqueeze(0)
        perturbation = perturbation.unsqueeze(0)
        squeeze_out = True
    else:
        squeeze_out = False
    B, C, H, W = image_tensor.shape
    image_fft = fft.fft2(image_tensor, dim=(-2, -1))  # [B, C, H, W]
    amplitude = torch.abs(image_fft)
    phase = torch.angle(image_fft)
    mask = torch.ones(1, 1, H, W, device=image_tensor.device)

    amp_pert = torch.tanh(perturbation[:,0]).unsqueeze(1).expand(-1, C, -1, -1)
    phase_pert = torch.tanh(perturbation[:,1]).unsqueeze(1).expand(-1, C, -1, -1)

    amplitude_perturbed = amplitude * (1 + amp_pert * mask)
    phase_perturbed = phase + (1 + phase_pert * mask)

    real = amplitude_perturbed * torch.cos(phase_perturbed)
    imag = amplitude_perturbed * torch.sin(phase_perturbed)
    image_fft_perturbed = torch.complex(real, imag)
    image_perturbed = fft.ifft2(image_fft_perturbed, dim=(-2, -1))
    out = image_perturbed.real
    if squeeze_out:
        out = out.squeeze(0)
    return out

def extract_num(filename):
    # 提取文件名前面的数字部分
    match = re.match(r"(\d+)", filename)
    return int(match.group(1)) if match else -1

def main(
    image_path,      # 输入图像路径（需包含清晰源区域）
    save_path,       # 保存文件夹路径
    attack_class,    # 攻击方式
    epsilon=0.02,    # 最大扰动幅度
    max_iter=50,     # 迭代次数
    lr=1e-2,
    ifftmask_lr=1e-2,
    temporal_lambda=1,  # 语义损失权重
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),  
    ssim_path='/home/yxstar/synctalk/clamp_data/clamp_Lieu/ssim_score.npy',
    attack_strength=10,
    threshold=0.85,
):
    """
    参数：
        image_path: 输入图像路径（需包含清晰源区域）
        epsilon: 扰动最大幅度（避免视觉失真）
        max_iter: 迭代优化次数
        lr: Adam优化器学习率
    返回：
        adv_image: 对抗样本（PIL格式）
        orig_pred: 原始图像解析结果（numpy数组，[H,W]）
        adv_pred: 对抗样本解析结果（numpy数组，[H,W]）
        tongue_mask: 源区域掩码（numpy数组，[H,W]，True表示源区域像素）
    """
    n_classes = 19 
    net = BiSeNet(n_classes=n_classes)
    net.to(device)
    # 加载预训练权重（替换为你的模型路径）
    net.load_state_dict(torch.load("./BiSeNet_model/79999_iter.pth"))
    net.eval()  

    if attack_class not in ['neck', 'mouth', 'face', 'full_face']:
        raise ValueError("攻击方式仅支持'neck','mouth', 'face', 'bg'")
    elif attack_class == 'neck':
        source_region = LABEL_MAP['neck']      
        target_regions = [LABEL_MAP['face'], LABEL_MAP['background']] 
    elif attack_class == 'face':
        source_region = LABEL_MAP['face']
        target_regions = [LABEL_MAP['neck'], LABEL_MAP['body']]
    elif attack_class == 'mouth':
        source_region = [LABEL_MAP['tongue'], LABEL_MAP['up_lip'], LABEL_MAP['down_lip']]
        target_regions = [LABEL_MAP['neck'], LABEL_MAP['body'], LABEL_MAP['background']]
    elif attack_class == 'full_face':
        source_region = list(range(1, 14))
        target_regions = [LABEL_MAP['background']]

    ssim = np.load(ssim_path)
    image_paths = os.listdir(image_path)
    count = 0
    save_path = osp.join(save_path, attack_class)
    save_path = osp.join(save_path, f'epsilon={str(epsilon)}')
    Path(save_path).mkdir(parents=True, exist_ok=True)

    config_dict = {
        "attack_class": attack_class,
        "epsilon": epsilon,
        "max_iter": max_iter,
        "lr": lr,
        "ifftmask_lr": ifftmask_lr,
        "temporal_lambda": temporal_lambda,
        "device": str(device),
        'threshold': threshold,
    }
    with open(osp.join(save_path, "config.txt"), "w") as f:
        json.dump(config_dict, f, indent=4, ensure_ascii=False)

    image_sematic = SemanticLoss(feature_layers=['3', '8', '15', '22'], weights=[1.0, 0.75, 0.5, 0.25], device=device)
    B = 1
    image_names = [f for f in os.listdir(image_path) if f.endswith('.jpg') or f.endswith('.png')]
    image_names = sorted(image_names, key=extract_num)
    num_images = len(image_names)
    torch.manual_seed(42)  
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42) 
    start_time = time.time()
    for i in tqdm.tqdm(range(0, num_images, B), desc='attack images'):

        last_idx = min(i + B, num_images)
        batch_names = image_names[i:last_idx]
        current_B = len(batch_names)
        batch_images = []
        for name in batch_names:
            img = Image.open(osp.join(image_path, name)).convert('RGB')
            batch_images.append(img)
   
        batch_images_tensor = torch.stack([to_tensor(img) for img in batch_images], dim=0).to(device) 
        norm_batch_images = transform_batch(batch_images_tensor).to(device)
        
        image_15times, image_2times, image_75_perc, image_55_perc = multi_pt_resize(norm_batch_images)

        batch_image_mask = get_mask(norm_batch_images, net, source_region)
        image_15times_mask = get_mask(image_15times, net, source_region)
        image_2times_mask = get_mask(image_2times, net, source_region)
        image_75_perc_mask = get_mask(image_75_perc, net, source_region)
        image_55_perc_mask = get_mask(image_55_perc, net, source_region)

        H, W = norm_batch_images.shape[-2], norm_batch_images.shape[-1]
        if i == 0:
            adv_noise = torch.randn(current_B, 2, H, W, device=device, requires_grad=True)
            ifft_mask = torch.randn(H, W, device=device, requires_grad=True)
        else:
            ssim_score = ssim[i-1:i+current_B-1]
            new_noise = torch.tensor(ssim_score, device=device, dtype=torch.float32) * adv_noise * 0.8
            adv_noise = new_noise.clone().detach()  
            adv_noise.requires_grad_(True)
            new_ifft_mask = ifft_mask
            ifft_mask = new_ifft_mask.clone().detach()
            ifft_mask.requires_grad_(True)
            
           
        optimizer = optim.Adam([
            {'params': adv_noise, 'lr': lr},
            {'params': ifft_mask, 'lr': ifftmask_lr}
            ])  

        exit_flag = False

        exit_count = 0
        ifft_mask = ifft_mask.unsqueeze(0).unsqueeze(0)
        if i == 0:
            prev_last_adv_image = None
        for iter in range(max_iter):

            optimizer.zero_grad()  
            refer_image = norm_batch_images.clone()  
            adv_image = add_frequency_perturbation(refer_image, adv_noise)
            adv_image = inv_normalize_batch(adv_image)
            ifft_noise = (adv_image-batch_images_tensor) * 10
            ifft_noise = torch.tanh(ifft_noise)
            adv_image = batch_images_tensor + ifft_noise * epsilon
            adv_image = transform_batch(adv_image)
            norm_ifft_mask = torch.sigmoid(ifft_mask) 
            adv_image = norm_ifft_mask * adv_image + (1-norm_ifft_mask) * refer_image
            ori_range_image = inv_normalize_batch(adv_image) 
            quantized_image = DifferentiableQuantize.apply(ori_range_image) / 255.0 

            adv_image = transform_batch(quantized_image) 


            # 计算语义损失
            sematic_loss = image_sematic(adv_image, refer_image)
            adv_image_15times, adv_image_2times, adv_image_75_perc, adv_image_50_perc = multi_pt_resize(adv_image)



            if (i == 0 and iter % 5 == 0) or i > 0:
                exit_flag = check_all_success(adv_image, adv_image_15times, adv_image_2times, adv_image_75_perc, adv_image_50_perc, net,
                                              batch_image_mask, image_15times_mask, image_2times_mask, image_75_perc_mask,   
                                                image_55_perc_mask, target_regions, threshold=threshold)
                if exit_flag:
                    exit_count += 1
                    if exit_count >= attack_strength:
                        break

            batch_image_loss = calulate_loss(adv_image, net, batch_image_mask, target_regions)
            image_15times_loss = calulate_loss(adv_image_15times, net, image_15times_mask, target_regions)
            image_2times_loss = calulate_loss(adv_image_2times, net, image_2times_mask, target_regions)
            image_75_perc_loss = calulate_loss(adv_image_75_perc, net, image_75_perc_mask, target_regions)
            image_50_perc_loss = calulate_loss(adv_image_50_perc, net, image_55_perc_mask, target_regions)
            
            temporal_loss = 0.0
            if prev_last_adv_image is not None:
                inter_frame_temporal_loss = torch.mean((adv_image[0] - prev_last_adv_image.detach())**2)
            else:
                inter_frame_temporal_loss = 0.0

            for t in range(1, adv_image.shape[0]):
                temporal_loss += torch.mean((adv_image[t] - adv_image[t-1])**2) + inter_frame_temporal_loss

            # ssim_loss = 1 - SSIM(adv_image, refer_image, data_range=1.0)
            total_loss = (
                1*batch_image_loss + 
                0.5*sematic_loss + temporal_lambda*temporal_loss +
                0.8*image_15times_loss + 0.6*image_2times_loss + 0.6*image_75_perc_loss + 0.6*image_50_perc_loss
            ) 

            total_loss.backward()  
            optimizer.step()

        with torch.no_grad():
            adv_image = add_frequency_perturbation(refer_image, adv_noise)
            adv_image = inv_normalize_batch(adv_image) 
            ifft_noise = (adv_image-batch_images_tensor) * 10
            ifft_noise = torch.tanh(ifft_noise)
            adv_image = batch_images_tensor + ifft_noise * epsilon
            adv_image = transform_batch(adv_image)  
            norm_ifft_mask = torch.sigmoid(ifft_mask)
            adv_image = norm_ifft_mask * adv_image + (1-norm_ifft_mask) * refer_image
            ori_range_image = inv_normalize_batch(adv_image)
            quantized_image = DifferentiableQuantize.apply(ori_range_image) / 255.0  
            adv_image = transform_batch(quantized_image)  

            prev_last_adv_image = adv_image.clone().detach()

            ifft_mask = ifft_mask.squeeze(0).squeeze(0)

        for idx, (adv_no, name, adv_img_tensor) in enumerate(zip(adv_noise, batch_names, adv_image)):
            vis_all_size_parsing_maps(adv_img_tensor, net, save_path, name)

            adv_pixel_final = inv_normalize(adv_img_tensor.cpu())
            adv_pixel_final = torch.clamp(adv_pixel_final, 0.0, 1.0)
            adv_image_pil = to_pil(adv_pixel_final)
            image_NAME = name.split('.')[0]

            adv_image_name = f'{image_NAME}_{attack_class}_fft.png'
            save_file_path = osp.join(save_path, adv_image_name)
            adv_image_pil.save(save_file_path)

    print(f"共{len(image_paths)/2}张图像，无源区域图像{count}张")
    time_save_path = osp.join(save_path, "0_time.txt")
    end_time = time.time()
    total_time = end_time - start_time
    elapsed_time = timedelta(seconds=total_time)
    formatted_time = str(elapsed_time)
    with open(time_save_path, "w") as f:
        f.write(f"Total time: {total_time}s\n")
        f.write(f"Total time: {formatted_time}")
    return 

def vis_all_size_parsing_maps(adv_image, net, save_path, name):
    vis_adv_image = adv_image.unsqueeze(0)
    adv_image_15times, adv_image_2times, adv_image_75_perc, adv_image_50_perc = multi_pt_resize(vis_adv_image)
    for i, adv_img in enumerate([vis_adv_image, adv_image_15times, adv_image_2times, adv_image_75_perc, adv_image_50_perc]):
        adv_output = net(adv_img)
        adv_pred = adv_output.argmax(1).cpu().numpy() 
        adv_save_path = osp.join(save_path, f'{name}_adv_parsing_map_{i}.jpg')
        img_size = adv_img.shape[-2:]
        vis_parsing_maps(adv_pred[0], stride=1, save_im=True, save_path=adv_save_path, img_size=img_size)

def vis_parsing_maps(parsing_anno, stride, save_im=False, save_path='vis_results/parsing_map_on_im.jpg',
                     img_size=(512, 512)):
    color_map = {
        0: [255, 255, 255],
        1: [255, 0, 0], 2: [255, 0, 0], 3: [255, 0, 0],
        4: [255, 0, 0], 5: [255, 0, 0], 6: [255, 0, 0],
        7: [255, 0, 0], 8: [255, 0, 0], 9: [255, 0, 0],
        10: [255, 0, 0], 11: [255, 0, 0], 12: [255, 0, 0],
        13: [255, 0, 0],
        14: [0, 255, 0], 15:[0, 255, 0], 
        16: [0, 0, 255],
        17: [255, 0, 0], 18: [255, 0, 0]
    }
    
    vis_parsing_anno = parsing_anno.copy().astype(np.uint8)
    vis_parsing_anno = cv2.resize(
        vis_parsing_anno, None, fx=stride, fy=stride, interpolation=cv2.INTER_NEAREST)
    
    vis_parsing_anno_color = np.zeros(
        (vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3), dtype=np.uint8)
    
    for class_id, color in color_map.items():
        index = np.where(vis_parsing_anno == class_id)
        vis_parsing_anno_color[index[0], index[1], :] = color
    
    vis_im = cv2.resize(vis_parsing_anno_color, img_size, interpolation=cv2.INTER_NEAREST)
    
    if save_im:
        cv2.imwrite(save_path, vis_im)
    
    return vis_im

if __name__ == "__main__":
    parser = configargparse.ArgumentParser()
    parser.add_argument('--data_path', type=str,default='./data', help='data path')
    parser.add_argument('--attack_class', type=str, default='full_face', help='neck, mouth, face, full_face')
    parser.add_argument('--epsilon', type=float, default=0.1, help='max perturbation')
    parser.add_argument('--max_iter', type=int, default=3000, help='max iteration')
    parser.add_argument('--lr', type=float, default=1e-2, help='learning rate')
    parser.add_argument('--ifftmask_lr', type=float, default=1e-2, help='ifft mask learning rate')
    parser.add_argument('--threshold', type=float, default=0.8, help='attack success threshold')
    parser.add_argument('--temporal_lambda', type=float, default=1.0, help='weight of temporal loss')
    parser.add_argument('--device', type=str, default='cuda:0', help='device for attack')
    parser.add_argument('--attack_strength', type=int, default=1, help='attack strength')

    args = parser.parse_args()
    device = torch.device(args.device)

    image_path = f'{args.data_path}/ori_imgs' # original images path
    save_path = f'{args.data_path}/ifftmask_fft_adv_imgs' # save path for adversarial images
    ssim_path = f'{args.data_path}/ori_imgs/ssim_score.npy' # ssim score path

    main(
        image_path=image_path, save_path=save_path, attack_class=args.attack_class, 
        epsilon=args.epsilon, max_iter=args.max_iter, lr=args.lr,
        temporal_lambda=args.temporal_lambda, ifftmask_lr=args.ifftmask_lr, threshold=args.threshold,
        device=device, ssim_path=ssim_path, attack_strength=args.attack_strength
    )
