import os
import sys
import argparse
import time
import json
import math
import shutil
import gc
import warnings
import types
from datetime import timedelta
from math import pi, cos

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from PIL import Image

# 引入项目中的模块
from config import GlobalConfig
from model import LidarCenterNet
from data import CARLA_Data

# 引入评估库
try:
    import lpips
    from torch_fidelity import calculate_metrics
except ImportError:
    print("Warning: lpips or torch-fidelity not found.")

# ================= PerC-AL 攻击配置 =================
ATTACK_PARAMS = {
    # PerC-AL 是迭代攻击，不是单步
    "max_iterations": 400,  # 迭代次数 (TransFuser 推理较慢，建议 50-100，原版 PerC-AL 针对 SimLingo 设为 1000)
    "alpha_l_init": 1,   # 任务损失更新步长 
    "alpha_c_init": 0.8,  # 颜色损失更新步长
    "device": "cuda:1",     
    
    # === MFA Loss 权重 ===
    "w_feature": 3,       # MSE 权重
    "w_cosine": 1.5,        # 余弦相似度权重
    "w_attn": 4.5,          # 注意力加权权重
    "attn_temp": 4.5,        # 注意力温度
}

# =========================================================================
# === 可微分颜色空间函数 (来自 openloop_gray_percal.py) ===
# =========================================================================

def rgb2xyz(rgb_image, device):
    rgb_image = rgb_image.float()
    mt = torch.tensor([[0.4124, 0.3576, 0.1805], 
                   [0.2126, 0.7152, 0.0722],
                   [0.0193, 0.1192, 0.9504]], dtype=torch.float32).to(device)
    mask1 = (rgb_image > 0.0405).float()
    mask1_no = 1 - mask1
    temp_img = mask1 * (((rgb_image + 0.055 ) / 1.055 ) ** 2.4)
    temp_img = temp_img + mask1_no * (rgb_image / 12.92)    
    temp_img = 100 * temp_img
    res = torch.matmul(mt, temp_img.permute(1, 0, 2,3).contiguous().view(3, -1)).view(3, rgb_image.size(0),rgb_image.size(2), rgb_image.size(3)).permute(1, 0, 2,3)
    return res

def xyz_lab(xyz_image, device):
    xyz_image = xyz_image.float()
    mask_value_0 = (xyz_image == 0).float().to(device)
    mask_value_0_no = 1 - mask_value_0
    xyz_image = xyz_image + 0.0001 * mask_value_0
    mask1 = (xyz_image > 0.008856).float()     
    mask1_no = 1 - mask1
    res = mask1 * (xyz_image) ** (1 / 3)
    res = res + mask1_no * ((7.787 * xyz_image) + (16 / 116))
    res = res * mask_value_0_no
    return res    

def rgb2lab_diff(rgb_image, device):
    rgb_image = rgb_image.to(device).float()
    xyz_image = rgb2xyz(rgb_image, device)
    xn = 95.0489; yn = 100.0; zn = 108.8840
    x = xyz_image[:,0, :, :]; y = xyz_image[:,1, :, :]; z = xyz_image[:,2, :, :]
    L = 116 * xyz_lab(y / yn, device) - 16
    a = 500 * (xyz_lab(x / xn, device) - xyz_lab(y / yn, device))
    b = 200 * (xyz_lab(y / yn, device) - xyz_lab(z / zn, device))
    res = torch.stack([L, a, b], dim=1)
    return res

def degrees(n): return n * (180. / np.pi)
def radians(n): return n * (np.pi / 180.)

def hpf_diff(x, y):
    x = x.float(); y = y.float()
    mask1 = ((x == 0) * (y == 0)).float()
    mask1_no = 1 - mask1
    tmphp = degrees(torch.atan2(x * mask1_no, y * mask1_no))
    tmphp1 = tmphp * (tmphp >= 0).float()
    tmphp2 = (360 + tmphp) * (tmphp < 0).float()
    return tmphp1 + tmphp2

def dhpf_diff(c1, c2, h1p, h2p):
    mask1  = ((c1 * c2) == 0).float()
    mask1_no  = 1 - mask1
    res1 = (h2p - h1p) * mask1_no * (torch.abs(h2p - h1p) <= 180).float()
    res2 = ((h2p - h1p) - 360) * ((h2p - h1p) > 180).float() * mask1_no
    res3 = ((h2p - h1p) + 360) * ((h2p - h1p) < -180).float() * mask1_no
    return res1 + res2 + res3

def ahpf_diff(c1, c2, h1p, h2p):
    mask1 = ((c1 * c2) == 0).float()
    mask1_no = 1 - mask1
    mask2 = (torch.abs(h2p - h1p) <= 180).float()
    mask2_no = 1 - mask2
    mask3 = (torch.abs(h2p + h1p) < 360).float()
    mask3_no = 1 - mask3
    res1 = (h1p + h2p) * mask1_no * mask2
    res2 = (h1p + h2p + 360.) * mask1_no * mask2_no * mask3 
    res3 = (h1p + h2p - 360.) * mask1_no * mask2_no * mask3_no
    res = (res1 + res2 + res3) + (res1 + res2 + res3) * mask1
    return res * 0.5

def ciede2000_diff(lab1, lab2, device):
    lab1 = lab1.to(device).float(); lab2 = lab2.to(device).float()
    L1 = lab1[:,0,:,:]; A1 = lab1[:,1,:,:]; B1 = lab1[:,2,:,:]
    L2 = lab2[:,0,:,:]; A2 = lab2[:,1,:,:]; B2 = lab2[:,2,:,:]   
    kL = 1; kC = 1; kH = 1
    
    mask_value_0_input1 = ((A1 == 0) * (B1 == 0)).float()
    mask_value_0_input2 = ((A2 == 0) * (B2 == 0)).float()
    mask_value_0_input1_no = 1 - mask_value_0_input1
    mask_value_0_input2_no = 1 - mask_value_0_input2
    B1 = B1 + 0.0001 * mask_value_0_input1
    B2 = B2 + 0.0001 * mask_value_0_input2 
    
    C1 = torch.sqrt((A1 ** 2.) + (B1 ** 2.))
    C2 = torch.sqrt((A2 ** 2.) + (B2 ** 2.))   
    aC1C2 = (C1 + C2) / 2.
    G = 0.5 * (1. - torch.sqrt((aC1C2 ** 7.) / ((aC1C2 ** 7.) + (25 ** 7.))))
    a1P = (1. + G) * A1; a2P = (1. + G) * A2
    c1P = torch.sqrt((a1P ** 2.) + (B1 ** 2.)); c2P = torch.sqrt((a2P ** 2.) + (B2 ** 2.))
    h1P = hpf_diff(B1, a1P); h2P = hpf_diff(B2, a2P)
    h1P = h1P * mask_value_0_input1_no; h2P = h2P * mask_value_0_input2_no 
    
    dLP = L2 - L1; dCP = c2P - c1P
    dhP = dhpf_diff(C1, C2, h1P, h2P)
    dHP = 2. * torch.sqrt(c1P * c2P) * torch.sin(radians(dhP) / 2.)
    mask_0_no = 1 - torch.max(mask_value_0_input1, mask_value_0_input2)
    dHP = dHP * mask_0_no

    aL = (L1 + L2) / 2.; aCP = (c1P + c2P) / 2.
    aHP = ahpf_diff(C1, C2, h1P, h2P)
    T = 1. - 0.17 * torch.cos(radians(aHP - 39)) + 0.24 * torch.cos(radians(2. * aHP)) + 0.32 * torch.cos(radians(3. * aHP + 6.)) - 0.2 * torch.cos(radians(4. * aHP - 63.))
    dRO = 30. * torch.exp(-1. * (((aHP - 275.) / 25.) ** 2.))
    rC = torch.sqrt((aCP ** 7.) / ((aCP ** 7.) + (25. ** 7.)))    
    sL = 1. + ((0.015 * ((aL - 50.) ** 2.)) / torch.sqrt(20. + ((aL - 50.) ** 2.)))
    sC = 1. + 0.045 * aCP; sH = 1. + 0.015 * aCP * T
    rT = -2. * rC * torch.sin(radians(2. * dRO))
    res_square = ((dLP / (sL * kL)) ** 2.) + ((dCP / (sC * kC)) ** 2.) * mask_0_no + ((dHP / (sH * kH)) ** 2.) * mask_0_no + rT * (dCP / (sC * kC)) * (dHP / (sH * kH)) * mask_0_no
    mask_0 = (res_square <= 0).float()
    mask_0_no = 1 - mask_0
    res_square = res_square + 0.0001 * mask_0    
    res = torch.sqrt(res_square); res = res * mask_0_no
    return res

def quantization(x):
   """quantize the continuous image tensors into 255 levels (8 bit encoding)"""
   x_quan=torch.round(x*255)/255 
   return x

# ================= Monkey Patching 函数 =================
def list_capturing_forward(self, x):
    B, T, C = x.size()
    k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
    q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
    v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

    att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
    att = F.softmax(att, dim=-1)
    
    self.stored_att = att # Shape: [B, n_head, T, T]

    att = self.attn_drop(att)
    y = att @ v
    y = y.transpose(1, 2).contiguous().view(B, T, C)
    y = self.resid_drop(self.proj(y))
    return y

# ================= MFA Loss Helpers =================
def get_transfuser_attention_loss(curr_img_feat, curr_lidar_feat, target_img_feat, target_lidar_feat, clean_attn_map, temperature):
    if clean_attn_map is None: return 0.0

    b, c, h_img, w_img = curr_img_feat.shape
    feat_img_flat = curr_img_feat.view(b, c, -1).permute(0, 2, 1)
    feat_lid_flat = curr_lidar_feat.view(b, c, -1).permute(0, 2, 1)
    feat_full = torch.cat([feat_img_flat, feat_lid_flat], dim=1)
    
    target_img_flat = target_img_feat.view(b, c, -1).permute(0, 2, 1)
    target_lid_flat = target_lidar_feat.view(b, c, -1).permute(0, 2, 1)
    target_full = torch.cat([target_img_flat, target_lid_flat], dim=1)
    
    importance = clean_attn_map.mean(dim=1).mean(dim=1) 
    importance = F.softmax(importance * temperature, dim=-1).unsqueeze(-1) 
    
    diff_sq = (feat_full - target_full) ** 2
    loss = (diff_sq * importance).sum(dim=1).mean()
    return loss

# ================= 辅助函数 (SSIM/Logger) =================
def find_valid_datasets(base_path):
    valid_roots = []
    print(f"Scanning for datasets in: {base_path} ...")
    for root, dirs, files in os.walk(base_path):
        has_routes = False
        for d in dirs:
            route_path = os.path.join(root, d)
            if os.path.isdir(route_path) and os.path.exists(os.path.join(route_path, 'measurements')):
                has_routes = True
                break
        if has_routes:
            valid_roots.append(root)
    return sorted(list(set(valid_roots)))

def gaussian(window_size, sigma):
    gauss = torch.Tensor([math.exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = torch.autograd.Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)
    mu1_sq = mu1.pow(2); mu2_sq = mu2.pow(2); mu1_mu2 = mu1*mu2
    sigma1_sq = F.conv2d(img1*img1, window, padding=window_size//2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding=window_size//2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding=window_size//2, groups=channel) - mu1_mu2
    C1 = 0.01**2; C2 = 0.03**2
    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
    if size_average: return ssim_map.mean()
    else: return ssim_map.mean(1).mean(1).mean(1)

class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIM, self).__init__()
        self.window_size = window_size; self.channel = 3; self.window = create_window(window_size, self.channel)
        self.size_average = size_average
    def forward(self, img1, img2):
        img1 = (img1 + 1) / 2.0; img2 = (img2 + 1) / 2.0
        if img1.is_cuda: self.window = self.window.cuda(img1.get_device())
        self.window = self.window.type_as(img1)
        return _ssim(img1, img2, self.window, self.window_size, self.channel, self.size_average)

class Logger(object):
    def __init__(self, filename="default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a")
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush() 
    def flush(self): pass

# ================= 攻击者类 (PerC-AL Implementation) =================
class TransFuserAttacker:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self._hook_features = {}
        self._register_hook()
        
        # Monkey Patching
        if hasattr(self.model, '_model') and hasattr(self.model._model, 'transformer1'):
            tf_blocks = self.model._model.transformer1.blocks
            for i, block in enumerate(tf_blocks):
                block.attn.forward = types.MethodType(list_capturing_forward, block.attn)
            print(f">> Monkey-Patch applied to {len(tf_blocks)} Attention layers for MFA Attack.")
        else:
             print("Warning: Could not find Transformer1 blocks for Monkey Patching.")

        try:
            self.lpips_vgg = lpips.LPIPS(net='vgg').to(device).eval()
        except:
            self.lpips_vgg = None
        self.ssim_metric = SSIM().to(device)

        for param in self.model.parameters():
            param.requires_grad = False
            
    def _register_hook(self):
        def hook_fn(module, input, output):
            self._hook_features['transformer1'] = output
        
        if hasattr(self.model, '_model') and hasattr(self.model._model, 'transformer1'):
            target_layer = self.model._model.transformer1
            target_layer.register_forward_hook(hook_fn)
        else:
            raise AttributeError("Error: Could not find 'transformer1'.")

    def run_attack(self, rgb_raw, lidar, target_point, target_point_image, ego_vel, bev_points=None, cam_points=None, num_points=None):
        """
        PerC-AL 攻击 (迭代优化)
        输入 rgb_raw 为 [0, 255] 的 Tensor
        """
        # PerC-AL 需要在 [0,1] 范围内操作
        inputs = rgb_raw.clone().float() / 255.0
        batch_size = inputs.shape[0]

        # Dummy inputs for feature extraction
        dummy_bev = torch.zeros(batch_size, 160, 160).to(self.device).long()
        dummy_label = torch.zeros(batch_size, 1, 7).to(self.device).float()
        dummy_depth = torch.zeros(batch_size, 160, 704).to(self.device).float()
        dummy_semantic = torch.zeros(batch_size, 160, 704).to(self.device).long()
        dummy_wp = torch.zeros(batch_size, 4, 2).to(self.device).float()

        # 1. 计算原始(Clean)特征、预测和 LAB
        self._hook_features = {} 
        with torch.no_grad():
            pred_clean, _ = self.model.forward_ego(
                rgb_raw, lidar, target_point, target_point_image, ego_vel,
                bev_points=bev_points, cam_points=cam_points, num_points=num_points
            )
            if 'transformer1' not in self._hook_features:
                 self.model(rgb_raw, lidar, dummy_wp, target_point, target_point_image, ego_vel, 
                       dummy_bev, dummy_label, dummy_depth, dummy_semantic, num_points=num_points)
            
            clean_out = self._hook_features['transformer1']
            target_feat_img = clean_out[0].clone().detach()
            target_feat_lidar = clean_out[1].clone().detach()
            
            clean_attn_map = None
            try:
                if hasattr(self.model._model.transformer1.blocks[-1].attn, 'stored_att'):
                    clean_attn_map = self.model._model.transformer1.blocks[-1].attn.stored_att.clone().detach()
            except: pass
            
            # 计算 Clean LAB 供颜色损失使用
            inputs_LAB = rgb2lab_diff(inputs, self.device)

        # 2. 初始化扰动 delta
        delta = (torch.rand_like(inputs) * 2 - 1) * 1e-5
        delta.requires_grad_(True)

        # 3. 攻击参数
        alpha_l_init = ATTACK_PARAMS['alpha_l_init']
        alpha_c_init = ATTACK_PARAMS['alpha_c_init']
        max_iterations = ATTACK_PARAMS['max_iterations']
        
        alpha_l_min = alpha_l_init / 100
        alpha_c_min = alpha_c_init / 10

        mse_loss_fn = nn.MSELoss(reduction='mean')

        # 4. 优化循环
        for i in range(max_iterations):
            # Cosine Annealing
            alpha_c = alpha_c_min + 0.5 * (alpha_c_init - alpha_c_min) * (1 + cos(i / max_iterations * pi))
            alpha_l = alpha_l_min + 0.5 * (alpha_l_init - alpha_l_min) * (1 + cos(i / max_iterations * pi))

            # --- Phase 1: Task Loss Update ---
            if delta.grad is not None: delta.grad.zero_()
            
            current_inputs = inputs + delta
            # TransFuser expects [0, 255]
            model_input_rgb = current_inputs * 255.0 
            
            self._hook_features = {} 
            self.model(model_input_rgb, lidar, dummy_wp, target_point, target_point_image, ego_vel, 
                   dummy_bev, dummy_label, dummy_depth, dummy_semantic, num_points=num_points)
            
            adv_out = self._hook_features['transformer1']
            curr_img_feat = adv_out[0]
            curr_lidar_feat = adv_out[1]

            # MFA Loss Components
            loss_mse = mse_loss_fn(curr_img_feat, target_feat_img) + mse_loss_fn(curr_lidar_feat, target_feat_lidar)
            
            flat_adv = curr_img_feat.view(batch_size, -1)
            flat_tgt = target_feat_img.view(batch_size, -1)
            loss_cos = F.cosine_similarity(flat_adv, flat_tgt, dim=-1).mean()
            
            loss_attn = 0.0
            if clean_attn_map is not None:
                loss_attn = get_transfuser_attention_loss(
                    curr_img_feat, curr_lidar_feat, 
                    target_feat_img, target_feat_lidar, 
                    clean_attn_map, temperature=ATTACK_PARAMS['attn_temp']
                )

            # PerC-AL Maximize Task Loss => Minimize (-Loss)
            total_task_obj = (ATTACK_PARAMS['w_feature'] * loss_mse + 
                              ATTACK_PARAMS['w_attn'] * loss_attn - 
                              ATTACK_PARAMS['w_cosine'] * loss_cos)
            
            task_loss = -1.0 * total_task_obj
            task_loss.backward()
            
            # Normalize Gradient
            grad_a = delta.grad.clone()
            delta.grad.zero_()
            norm_grad = torch.norm(grad_a.reshape(batch_size, -1), dim=1) + 1e-8
            normalized_grad = (grad_a.permute(1,2,3,0) / norm_grad).permute(3,0,1,2)
            
            # Update Delta (Task)
            delta.data = delta.data - alpha_l * normalized_grad

            # --- Phase 2: Color Loss Update ---
            current_adv_img = (inputs + delta).clamp(0, 1)
            current_adv_lab = rgb2lab_diff(current_adv_img, self.device)
            
            d_map = ciede2000_diff(inputs_LAB, current_adv_lab, self.device)
            if d_map.dim() == 3: d_map = d_map.unsqueeze(1)
            
            color_dis = torch.norm(d_map.reshape(batch_size, -1), dim=1)
            color_loss = color_dis.sum()
            color_loss.backward()
            
            # Normalize Gradient
            grad_color = delta.grad.clone()
            delta.grad.zero_()
            norm_grad_color = torch.norm(grad_color.reshape(batch_size, -1), dim=1) + 1e-8
            normalized_grad_color = (grad_color.permute(1,2,3,0) / norm_grad_color).permute(3,0,1,2)
            
            # Update Delta (Color)
            delta.data = delta.data - alpha_c * normalized_grad_color

            # --- Phase 3: Constraint & Quantization ---
            delta.data = (inputs + delta.data).clamp(0, 1) - inputs
            
            # ================= [新增] 实时计算并显示 LPIPS / SSIM =================
            if i % 10 == 0 or i == max_iterations - 1:
                with torch.no_grad():
                    cur_adv_img = (inputs + delta).clamp(0, 1)
                    cur_adv_norm = cur_adv_img * 2 - 1
                    clean_norm = inputs * 2 - 1
                    
                    cur_lpips = 0.0
                    if self.lpips_vgg is not None:
                        cur_lpips = self.lpips_vgg(cur_adv_norm, clean_norm).mean().item()
                    cur_ssim = self.ssim_metric(cur_adv_norm, clean_norm).item()
                    
                    print(f"\rIter [{i+1}/{max_iterations}] "
                          f"Loss: {task_loss.item():.4f} | "
                          f"LPIPS: {cur_lpips:.4f} | "
                          f"SSIM: {cur_ssim:.4f}", end="", flush=True)
            # ====================================================================

        print() # 换行
        
        # 5. Final Output Generation
        X_adv_round = quantization(inputs + delta.data)
        final_adv_img_01 = X_adv_round.clamp(0, 1)
        rgb_adv = final_adv_img_01 * 255.0 # Back to [0, 255]

        # 6. Evaluation
        with torch.no_grad():
            pred_wp_adv, _ = self.model.forward_ego(
                rgb_adv, lidar, target_point, target_point_image, ego_vel,
                bev_points=bev_points, cam_points=cam_points, num_points=num_points
            )
            
            self._hook_features = {}
            if 'transformer1' not in self._hook_features:
                 self.model(rgb_adv, lidar, dummy_wp, target_point, target_point_image, ego_vel, 
                       dummy_bev, dummy_label, dummy_depth, dummy_semantic, num_points=num_points)
            final_out = self._hook_features.get('transformer1', clean_out)
            
            sim_drift = (mse_loss_fn(final_out[0], target_feat_img) + mse_loss_fn(final_out[1], target_feat_lidar)).item()
            
            img_clean_norm = (inputs * 2) - 1 # [-1, 1] for LPIPS
            img_adv_norm = (final_adv_img_01 * 2) - 1
            
            lpips_val = 0.0
            if self.lpips_vgg is not None:
                lpips_val = self.lpips_vgg(img_adv_norm, img_clean_norm).mean().item()
            ssim_val = self.ssim_metric(img_adv_norm, img_clean_norm).item()

        return rgb_adv, pred_clean, pred_wp_adv, sim_drift, lpips_val, ssim_val

# ================= 可视化函数 (1x4 Layout) =================
def save_visualization_1x4(rgb_clean, rgb_adv, pred_clean, pred_adv, gt_wp, target_point, save_path, frame_id, metrics_str):
    img_clean_np = rgb_clean.permute(1, 2, 0).detach().cpu().numpy().astype(np.float32) / 255.0
    img_adv_np = rgb_adv.permute(1, 2, 0).detach().cpu().numpy().astype(np.float32) / 255.0
    
    img_clean_np = np.clip(img_clean_np, 0, 1)
    img_adv_np = np.clip(img_adv_np, 0, 1)
    
    # 噪声增强显示 (PerC-AL 扰动更隐蔽，这里放大显示)
    noise = np.abs(img_adv_np - img_clean_np) * 50.0
    noise = np.clip(noise, 0, 1)

    fig = plt.figure(figsize=(15, 6))
    gs = fig.add_gridspec(1, 4)
    
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(img_clean_np); ax1.set_title("Clean"); ax1.axis('off')
    
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(img_adv_np); ax2.set_title("Adv (PerC-AL)"); ax2.axis('off')
    
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(noise); ax3.set_title("Noise (x50)"); ax3.axis('off')
    
    ax4 = fig.add_subplot(gs[0, 3])
    if gt_wp is not None:
        ax4.plot(gt_wp[:, 1], gt_wp[:, 0], 'g-', alpha=0.5, linewidth=3, label='GT')
    
    ax4.plot(pred_clean[:, 1], pred_clean[:, 0], 'b-o', markersize=4, label='Clean')
    ax4.plot(pred_adv[:, 1], pred_adv[:, 0], 'r--^', markersize=4, label='Adv')
    
    if target_point is not None:
        ax4.plot(target_point[0], -target_point[1], 'k*', markersize=15, label='Target', zorder=10)
    
    ax4.plot(0, 0, 'k^', markersize=10, label='Ego')
    ax4.set_xlim(-10, 10); ax4.set_ylim(-2, 35)
    ax4.legend(loc='upper right')
    ax4.grid(True, linestyle='--', alpha=0.5)
    ax4.set_title(f"Trajectory\n{metrics_str}", fontsize=10)
    ax4.set_aspect('equal')

    plt.tight_layout()
    plt.savefig(save_path, dpi=100)
    plt.close(fig)

# ================= 主函数 =================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--root_dir', type=str, required=True)
    parser.add_argument('--backbone', type=str, default='transFuser')
    parser.add_argument('--image_architecture', type=str, default='regnety_032')
    parser.add_argument('--lidar_architecture', type=str, default='regnety_032')
    parser.add_argument('--output_dir', type=str, default='percal_transfuser_results')
    args = parser.parse_args()

    # 1. 初始化
    os.makedirs(args.output_dir, exist_ok=True)
    log_file = os.path.join(args.output_dir, "attack_log.txt")
    sys.stdout = Logger(log_file)
    
    fid_clean_dir = os.path.join(args.output_dir, "fid_clean")
    fid_adv_dir = os.path.join(args.output_dir, "fid_adv")
    if os.path.exists(fid_clean_dir): shutil.rmtree(fid_clean_dir)
    if os.path.exists(fid_adv_dir): shutil.rmtree(fid_adv_dir)
    os.makedirs(fid_clean_dir, exist_ok=True)
    os.makedirs(fid_adv_dir, exist_ok=True)

    print(f"=== TransFuser PerC-AL Attack ===")
    print(json.dumps(ATTACK_PARAMS, indent=2))
    
    device_str = ATTACK_PARAMS.get("device", "cuda:0")
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    if device.type == 'cuda': torch.cuda.set_device(device)
    print(f"Running on: {device}")

    # 2. 加载模型
    config = GlobalConfig(setting='eval')
    config.use_velocity = True 
    config.backbone = args.backbone
    config.use_target_point_image = True 
    
    print(f"Loading model from {args.model_path}...")
    model = LidarCenterNet(config, device, args.backbone, args.image_architecture, args.lidar_architecture, True)
    state_dict = torch.load(args.model_path, map_location=device)
    new_state_dict = {k[7:] if k.startswith('module.') else k: v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict, strict=False)
    model.to(device)
    model.eval()
    
    attacker = TransFuserAttacker(model, device)

    # 3. 统计变量
    valid_datasets = find_valid_datasets(args.root_dir)
    global_stats = {
        'shift': [], 'lat_err': [], 'lon_err': [], 
        'sim_drift': [], 'lpips': [], 'ssim': [], 
        'clean_len': [], 'success': [], 'avg_time': [],
        'global_tan': [], 'global_angle': []
    }
    
    overall_start_time = time.time()
    total_samples_processed = 0

    # 4. 数据集循环
    for ds_idx, dataset_path in enumerate(valid_datasets):
        ds_name = os.path.basename(dataset_path) 
        print(f"\n[{ds_idx+1}/{len(valid_datasets)}] Processing: {ds_name}")
        ds_out_dir = os.path.join(args.output_dir, ds_name)
        os.makedirs(ds_out_dir, exist_ok=True)
        
        # 数据集级统计
        ds_totals = {
            'shift': 0.0, 'lat_err': 0.0, 'lon_err': 0.0, 'drift': 0.0, 
            'lpips': 0.0, 'ssim': 0.0, 'clean_len': 0.0, 
            'success_count': 0, 'total_time': 0.0
        }
        count = 0
        
        val_dataset = CARLA_Data(root=[dataset_path], config=config, shared_dict=None)
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4)
        
        for batch_idx, data in enumerate(tqdm(val_loader, desc=ds_name)):
            t0 = time.time() 
            rgb = data['rgb'].to(device, dtype=torch.float32)
            lidar = data['lidar'].to(device, dtype=torch.float32)
            target_point = data['target_point'].to(device, dtype=torch.float32)
            tp_img = data['target_point_image'].to(device, dtype=torch.float32)
            ego_vel = data['speed'].to(device, dtype=torch.float32).reshape(-1, 1)
            gt_wp = data['ego_waypoint'].to(device, dtype=torch.float32)
            
            bev_points = data.get('bev_points', None)
            cam_points = data.get('cam_points', None)
            if bev_points is not None: bev_points = bev_points.long().to(device)
            if cam_points is not None: cam_points = cam_points.long().to(device)

            # 运行 PerC-AL 攻击
            rgb_adv, pred_clean, pred_adv, drift, lpips_val, ssim_val = attacker.run_attack(
                rgb, lidar, target_point, tp_img, ego_vel, 
                bev_points=bev_points, cam_points=cam_points
            )
            step_time = time.time() - t0 

            # 统计
            clean_wp = pred_clean[0].cpu().numpy()
            adv_wp = pred_adv[0].cpu().numpy()
            
            shift = np.linalg.norm(clean_wp - adv_wp)
            lat_err = abs(adv_wp[-1][1] - clean_wp[-1][1])
            lon_err = abs(adv_wp[-1][0] - clean_wp[-1][0])
            clean_len = np.linalg.norm(clean_wp[-1])
            is_succ = 1.0 if shift >= 1.0 else 0.0

            ds_totals['shift'] += shift
            ds_totals['lat_err'] += lat_err
            ds_totals['lon_err'] += lon_err
            ds_totals['drift'] += drift
            ds_totals['lpips'] += lpips_val
            ds_totals['ssim'] += ssim_val
            ds_totals['clean_len'] += clean_len
            ds_totals['success_count'] += is_succ
            ds_totals['total_time'] += step_time
            count += 1
            
            # 保存 FID 图片
            def save_img(tensor_255, path):
                arr = tensor_255[0].permute(1, 2, 0).detach().cpu().numpy().astype(np.uint8)
                Image.fromarray(arr).save(path)
            
            fname = f"{ds_name}_{batch_idx:04d}.jpg"
            save_img(rgb, os.path.join(fid_clean_dir, fname))
            save_img(rgb_adv, os.path.join(fid_adv_dir, fname))

            # 可视化 (1x4)
            if batch_idx % 20 == 0:
                metrics_str = f"Shift: {shift:.2f}m | LatErr: {lat_err:.2f}m | SSIM: {ssim_val:.3f}"
                save_path = os.path.join(ds_out_dir, f"frame_{batch_idx:04d}.png")
                save_visualization_1x4(rgb[0], rgb_adv[0], clean_wp, adv_wp, gt_wp[0].cpu().numpy(), 
                                       target_point[0].cpu().numpy(), save_path, batch_idx, metrics_str)

        # 5. 数据集结算
        if count > 0:
            avg_shift = ds_totals['shift'] / count
            avg_drift = ds_totals['drift'] / count
            avg_time_per_img = ds_totals['total_time'] / count
            succ_rate = (ds_totals['success_count'] / count) * 100
            
            ds_tan = ds_totals['lat_err'] / (ds_totals['clean_len'] + 1e-6)
            ds_angle = math.degrees(math.atan(ds_tan))
            
            global_stats['shift'].append(avg_shift)
            global_stats['lat_err'].append(ds_totals['lat_err'] / count)
            global_stats['lon_err'].append(ds_totals['lon_err'] / count)
            global_stats['sim_drift'].append(avg_drift)
            global_stats['lpips'].append(ds_totals['lpips'] / count)
            global_stats['ssim'].append(ds_totals['ssim'] / count)
            global_stats['clean_len'].append(ds_totals['clean_len'] / count)
            global_stats['success'].append(succ_rate)
            global_stats['avg_time'].append(avg_time_per_img)
            global_stats['global_tan'].append(ds_tan)
            
            total_samples_processed += count

            print("-" * 50)
            print(f"Dataset Summary: {ds_name}")
            print(f"  > Avg Time/Image: {avg_time_per_img:.3f} s") 
            print(f"  > Shift (Avg):    {avg_shift:.4f} m")
            print(f"  > Angle (Drift):  {ds_angle:.2f}°")
            print(f"  > Success Rate:   {succ_rate:.2f} %")
            print("-" * 50)

    # 6. 最终宏观报告
    total_elapsed_time = time.time() - overall_start_time
    if len(global_stats['shift']) > 0:
        def get_avg(k): return sum(global_stats[k]) / len(global_stats[k])
        
        avg_global_tan = get_avg('global_tan')
        avg_global_angle = math.degrees(math.atan(avg_global_tan))

        print("\n" + "="*60)
        print("MACRO AVERAGE METRICS (PerC-AL):")
        print(f"  > Total Samples:        {total_samples_processed}")
        print(f"  > Total Time:           {timedelta(seconds=int(total_elapsed_time))}")
        print(f"  > Avg Time Per Image:   {get_avg('avg_time'):.3f} s")
        print("-" * 40)
        print(f"  > Avg Global Tan:       {avg_global_tan:.4f}")
        print(f"  > Avg Global Angle:     {avg_global_angle:.2f}°")
        print(f"  > Avg Route Shift:      {get_avg('shift'):.4f} m")
        print(f"  > Avg Lateral Error:    {get_avg('lat_err'):.4f} m")
        print(f"  > Avg Longitudinal Err: {get_avg('lon_err'):.4f} m")
        print(f"  > Avg Success Rate:     {get_avg('success'):.2f} %")
        print(f"  > Avg Feature Drift:    {get_avg('sim_drift'):.4f}")
        print(f"  > Avg LPIPS:            {get_avg('lpips'):.4f}")
        print(f"  > Avg SSIM:             {get_avg('ssim'):.4f}")
        
        print("-" * 40)
        print("Calculating Global FID...")
        del model, attacker; gc.collect(); torch.cuda.empty_cache()
        
        try:
            metrics = calculate_metrics(input1=fid_clean_dir, input2=fid_adv_dir, cuda=True, isc=False, fid=True, verbose=False)
            print(f"  > Global FID:           {metrics['frechet_inception_distance']:.4f}")
        except: 
            print("  > FID Calculation Failed (torch-fidelity not installed)")
        print("="*60)

if __name__ == "__main__":
    main()