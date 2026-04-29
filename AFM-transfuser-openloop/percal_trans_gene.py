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

# 显存与性能优化
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from PIL import Image
from tqdm import tqdm
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

# TransFuser Modules
try:
    from config import GlobalConfig
    from model import LidarCenterNet
    from data1 import CARLA_Data
except ImportError:
    print("Please run this script from the TransFuser root directory.")
    sys.exit(1)

# Metrics
try:
    import lpips
    from torch_fidelity import calculate_metrics
except ImportError:
    print("Warning: lpips or torch-fidelity not found.")

# ================= 1. PerC-AL 攻击配置 (适配 MFA 结构) =================
ATTACK_PARAMS = {
    # === PerC-AL 参数 ===
    "max_iterations": 400,      # 迭代次数 (根据需要调整，原版较高，为了生成速度可适当降低)
    "alpha_l_init": 1.0,        # 任务损失更新步长 
    "alpha_c_init": 0.8,        # 颜色损失更新步长
    
    # === 损失权重 ===
    "w_feature": 3.0,           
    "w_cos": 1.5,               
    "w_attn_focus": 4.5,        
    "attn_temp": 4.5,           
    
    "device": "cuda:1",
}

# ================= 工具函数 (Logger, Scanner, SSIM) =================
class Logger(object):
    def __init__(self, filename="default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a")
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush() 
    def flush(self): pass

def find_valid_datasets(base_path):
    valid_routes = []
    print(f"Scanning for datasets in: {base_path} ...")
    for root, dirs, files in os.walk(base_path):
        if 'measurements' in dirs:
            valid_routes.append(root)
    if not valid_routes and os.path.exists(os.path.join(base_path, 'measurements')):
        valid_routes.append(base_path)
    return sorted(list(set(valid_routes)))

# SSIM Logic
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

# ================= 2. PerC-AL 核心数学函数 (颜色空间转换) =================
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
   x_quan=torch.round(x*255)/255 
   return x

# ================= Monkey Patching =================
def list_capturing_forward(self, x):
    B, T, C = x.size()
    k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
    q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
    v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

    att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
    att = F.softmax(att, dim=-1)
    self.stored_att = att 

    att = self.attn_drop(att)
    y = att @ v
    y = y.transpose(1, 2).contiguous().view(B, T, C)
    y = self.resid_drop(self.proj(y))
    return y

# ================= 3. 核心攻击类 (PerC-AL 适配版) =================
class TransFuserPercalAttacker:
    def __init__(self, transfuser_model, device):
        self.device = device
        self.tf_model = transfuser_model
        
        # Monkey Patching for Attention
        if hasattr(self.tf_model, '_model'):
            self.transformer1 = self.tf_model._model.transformer1
        else:
            raise AttributeError("TransFuser structure mismatch.")
        
        tf_blocks = self.transformer1.blocks
        for i, block in enumerate(tf_blocks):
            block.attn.forward = types.MethodType(list_capturing_forward, block.attn)
            
        # Hook Feature container
        self._hook_features = {}
        self._register_hook()

        try:
            self.lpips_vgg = lpips.LPIPS(net='vgg').to(device).eval()
        except: self.lpips_vgg = None
        self.ssim_loss = SSIM().to(device)

    def _register_hook(self):
        def hook_fn(module, input, output):
            self._hook_features['transformer1'] = output
        self.transformer1.register_forward_hook(hook_fn)

    def fix_target_point_image(self, tp_img):
        if tp_img is None:
            return torch.zeros(1, 1, 256, 256).to(self.device, dtype=torch.float32)
        if tp_img.dim() == 4: return tp_img
        batch_size = tp_img.shape[0] if tp_img.shape[0] > 0 else 1
        return torch.zeros(batch_size, 1, 256, 256).to(self.device, dtype=torch.float32)

    def run_attack(self, rgb_raw, batch_data):
        """
        使用 PerC-AL 的迭代逻辑，但接口适配 MFA
        rgb_raw: [B, 3, H, W] in [0, 255]
        """
        # 数据准备
        bs = rgb_raw.shape[0]
        inputs = rgb_raw.clone().float() / 255.0 # [0, 1]
        
        # 补全可能缺失的数据
        dummy_bev = torch.zeros(bs, 160, 160).to(self.device).long()
        dummy_label = torch.zeros(bs, 1, 7).to(self.device).float()
        dummy_depth = torch.zeros(bs, 160, 704).to(self.device).float()
        dummy_semantic = torch.zeros(bs, 160, 704).to(self.device).long()
        batch_data['target_point_image'] = self.fix_target_point_image(batch_data['target_point_image'])
        
        # 1. 原始 Clean 推理
        self._hook_features = {}
        with torch.no_grad():
            self.tf_model(
                rgb_raw, batch_data['lidar'], batch_data['ego_waypoint'], batch_data['target_point'], 
                batch_data['target_point_image'], batch_data['speed'].reshape(-1,1),
                dummy_bev, dummy_label, dummy_depth, dummy_semantic
            )
            clean_out = self._hook_features['transformer1']
            target_feat_img = clean_out[0].clone().detach()
            target_feat_lidar = clean_out[1].clone().detach()
            
            clean_tf_attn = None
            try:
                if hasattr(self.transformer1.blocks[-1].attn, 'stored_att'):
                    clean_tf_attn = self.transformer1.blocks[-1].attn.stored_att.detach().clone()
            except: pass

            pred_wp_clean, _ = self.tf_model.forward_ego(
                rgb_raw, batch_data['lidar'], batch_data['target_point'], 
                batch_data['target_point_image'], batch_data['speed'].reshape(-1,1),
                bev_points=batch_data.get('bev_points'), cam_points=batch_data.get('cam_points')
            )
            
            # 预计算 Clean LAB
            inputs_LAB = rgb2lab_diff(inputs, self.device)

        # 2. 初始化扰动
        delta = (torch.rand_like(inputs) * 2 - 1) * 1e-5
        delta.requires_grad_(True)
        
        # 3. 优化循环参数
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
            model_input_rgb = current_inputs * 255.0 # To TransFuser Range
            
            self._hook_features = {} 
            self.tf_model(
                model_input_rgb, batch_data['lidar'], batch_data['ego_waypoint'], batch_data['target_point'], 
                batch_data['target_point_image'], batch_data['speed'].reshape(-1,1),
                dummy_bev, dummy_label, dummy_depth, dummy_semantic
            )
            adv_out = self._hook_features['transformer1']
            curr_img_feat = adv_out[0]
            curr_lidar_feat = adv_out[1]
            
            # Loss Calculation
            loss_mse = mse_loss_fn(curr_img_feat, target_feat_img) + mse_loss_fn(curr_lidar_feat, target_feat_lidar)
            loss_cos = F.cosine_similarity(curr_img_feat.view(bs, -1), target_feat_img.view(bs, -1)).mean()
            
            loss_attn = 0.0
            if clean_tf_attn is not None:
                # Attention Loss Logic from PerC-AL
                b, c, h_img, w_img = curr_img_feat.shape
                feat_full = torch.cat([curr_img_feat.view(b, c, -1).permute(0, 2, 1), 
                                       curr_lidar_feat.view(b, c, -1).permute(0, 2, 1)], dim=1)
                target_full = torch.cat([target_feat_img.view(b, c, -1).permute(0, 2, 1), 
                                         target_feat_lidar.view(b, c, -1).permute(0, 2, 1)], dim=1)
                importance = clean_tf_attn.mean(dim=1).mean(dim=1)
                importance = F.softmax(importance * ATTACK_PARAMS['attn_temp'], dim=-1).unsqueeze(-1)
                diff_sq = (feat_full - target_full) ** 2
                loss_attn = (diff_sq * importance).sum(dim=1).mean()

            # Maximize Loss => Minimize -Loss
            total_task_obj = (ATTACK_PARAMS['w_feature'] * loss_mse + 
                              ATTACK_PARAMS['w_attn_focus'] * loss_attn - 
                              ATTACK_PARAMS['w_cos'] * loss_cos)
            
            task_loss = -1.0 * total_task_obj
            task_loss.backward()
            
            # Normalize Grad
            grad_a = delta.grad.clone()
            delta.grad.zero_()
            norm_grad = torch.norm(grad_a.reshape(bs, -1), dim=1) + 1e-8
            normalized_grad = (grad_a.permute(1,2,3,0) / norm_grad).permute(3,0,1,2)
            delta.data = delta.data - alpha_l * normalized_grad
            
            # --- Phase 2: Color Loss Update ---
            current_adv_img = (inputs + delta).clamp(0, 1)
            current_adv_lab = rgb2lab_diff(current_adv_img, self.device)
            d_map = ciede2000_diff(inputs_LAB, current_adv_lab, self.device)
            if d_map.dim() == 3: d_map = d_map.unsqueeze(1)
            
            color_loss = torch.norm(d_map.reshape(bs, -1), dim=1).sum()
            color_loss.backward()
            
            grad_color = delta.grad.clone()
            delta.grad.zero_()
            norm_grad_color = torch.norm(grad_color.reshape(bs, -1), dim=1) + 1e-8
            normalized_grad_color = (grad_color.permute(1,2,3,0) / norm_grad_color).permute(3,0,1,2)
            delta.data = delta.data - alpha_c * normalized_grad_color
            
            # --- Phase 3: Constraint ---
            delta.data = (inputs + delta.data).clamp(0, 1) - inputs

        # 5. 生成最终结果
        X_adv_round = quantization(inputs + delta.data)
        final_adv_img_01 = X_adv_round.clamp(0, 1)
        final_255 = final_adv_img_01 * 255.0
        
        # 6. 最终评估
        with torch.no_grad():
            self._hook_features = {}
            self.tf_model(
                final_255, batch_data['lidar'], batch_data['ego_waypoint'], batch_data['target_point'], 
                batch_data['target_point_image'], batch_data['speed'].reshape(-1,1),
                dummy_bev, dummy_label, dummy_depth, dummy_semantic
            )
            final_feat = self._hook_features.get('transformer1', clean_out)[0]
            drift = F.mse_loss(final_feat, target_feat_img).item() * 1000
            
            pred_adv, _ = self.tf_model.forward_ego(
                final_255, batch_data['lidar'], batch_data['target_point'], 
                batch_data['target_point_image'], batch_data['speed'].reshape(-1,1),
                bev_points=batch_data.get('bev_points'), cam_points=batch_data.get('cam_points')
            )
            
            lpips_val = 0.0
            if self.lpips_vgg: 
                lpips_val = self.lpips_vgg(final_adv_img_01*2-1, inputs*2-1).mean().item()
            ssim_val = self.ssim_loss(final_adv_img_01*2-1, inputs*2-1).item()

        return final_255, pred_wp_clean, pred_adv, drift, lpips_val, ssim_val

# ================= Visualization =================
def save_visualization_unified(rgb_clean, rgb_adv, pred_clean, pred_adv, gt_wp, target_point, save_path, drift, title_prefix=""):
    img_clean = np.clip(rgb_clean.permute(1,2,0).cpu().numpy()/255.0, 0, 1)
    img_adv = np.clip(rgb_adv.permute(1,2,0).cpu().numpy()/255.0, 0, 1)
    noise = np.clip(np.abs(img_adv - img_clean) * 50.0, 0, 1) # PerC-AL 扰动较小，放大显示

    fig = plt.figure(figsize=(16, 14))
    gs = fig.add_gridspec(2, 2, hspace=0.2, wspace=0.15)
    
    ax1 = fig.add_subplot(gs[0, 0]); ax1.imshow(img_clean)
    ax1.set_title(f"{title_prefix} 1. Clean", fontsize=14, fontweight='bold'); ax1.axis('off')
    
    ax2 = fig.add_subplot(gs[0, 1]); ax2.imshow(img_adv)
    ax2.set_title("2. Adv (PerC-AL)", fontsize=14, fontweight='bold'); ax2.axis('off')
    
    ax3 = fig.add_subplot(gs[1, 0]); ax3.imshow(noise)
    ax3.set_title("3. Noise (x50)", fontsize=14, fontweight='bold'); ax3.axis('off')
    
    ax4 = fig.add_subplot(gs[1, 1])
    if gt_wp is not None: ax4.plot(gt_wp[:,1], gt_wp[:,0], 'g-', lw=5, alpha=0.3, label='GT')
    ax4.plot(pred_clean[:,1], pred_clean[:,0], 'b-o', markersize=5, lw=2, label='Clean')
    ax4.plot(pred_adv[:,1], pred_adv[:,0], 'r--^', markersize=6, lw=2, label='Adv')
    
    tp = target_point.cpu().numpy() if isinstance(target_point, torch.Tensor) else target_point
    ax4.scatter(tp[0], -tp[1], c='gold', marker='*', s=300, edgecolors='black', label='Target', zorder=10)
    
    ax4.set_xlim(-12, 12); ax4.set_ylim(-2, 40)
    ax4.set_xlabel("Lateral (m)"); ax4.set_ylabel("Forward (m)")
    ax4.grid(True, linestyle=':', alpha=0.6)
    ax4.legend(loc='upper right')
    ax4.set_title(f"4. Trajectory (Drift: {drift:.1f})", fontsize=14, fontweight='bold')
    
    plt.tight_layout(); plt.savefig(save_path, dpi=120, bbox_inches='tight'); plt.close(fig)

# ================= Main =================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--root_dir', type=str, required=True)
    parser.add_argument('--backbone', type=str, default='transFuser')
    parser.add_argument('--image_architecture', type=str, default='regnety_032')
    parser.add_argument('--lidar_architecture', type=str, default='regnety_032')
    parser.add_argument('--output_dir', type=str, default='percal_mfa_style_results')
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    sys.stdout = Logger(os.path.join(args.output_dir, "attack_log.txt"))
    
    # === Global FID Directories ===
    global_fid_c = os.path.join(args.output_dir, "fid_global_clean")
    global_fid_a = os.path.join(args.output_dir, "fid_global_adv")
    if os.path.exists(global_fid_c): shutil.rmtree(global_fid_c)
    if os.path.exists(global_fid_a): shutil.rmtree(global_fid_a)
    os.makedirs(global_fid_c, exist_ok=True)
    os.makedirs(global_fid_a, exist_ok=True)
    
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    ATTACK_PARAMS['device'] = f"cuda:{args.gpu}"
    
    print("=== TransFuser PerC-AL Attack (MFA Style) ===")
    print(json.dumps(ATTACK_PARAMS, indent=2))

    config = GlobalConfig(setting='eval')
    config.use_velocity = True; config.backbone = args.backbone; config.use_target_point_image = True
    
    print(f"Loading TransFuser on {device}...")
    model = LidarCenterNet(config, device, args.backbone, args.image_architecture, args.lidar_architecture, True)
    state = torch.load(args.model_path, map_location=device)
    state = {k[7:] if k.startswith('module.') else k: v for k, v in state.items()}
    model.load_state_dict(state, strict=False)
    model.to(device); model.eval()
    for p in model.parameters(): p.requires_grad = False

    attacker = TransFuserPercalAttacker(model, device)
    
    datasets = find_valid_datasets(args.root_dir)
    
    global_stats = {
        'shift': [], 'lat_err': [], 'lon_err': [], 'sim_drift': [], 
        'lpips': [], 'ssim': [], 'clean_len': [], 'success': [], 
        'global_tan': [], 'avg_time': []
    }
    
    start_time = time.time()
    total_processed = 0

    for ds_idx, ds_path in enumerate(datasets):
        ds_name = os.path.basename(os.path.normpath(ds_path))
        print(f"\n[{ds_idx+1}/{len(datasets)}] Processing Route: {ds_name}")
        
        # === 建立基于 Route 的保存目录 (MFA Style) ===
        route_out_dir = os.path.join(args.output_dir, ds_name)
        adv_save_dir = os.path.join(route_out_dir, "adv_rgb")
        clean_save_dir = os.path.join(route_out_dir, "clean_rgb")
        vis_save_dir = os.path.join(route_out_dir, "vis_analysis")
        
        os.makedirs(route_out_dir, exist_ok=True)
        os.makedirs(adv_save_dir, exist_ok=True)
        os.makedirs(clean_save_dir, exist_ok=True)
        os.makedirs(vis_save_dir, exist_ok=True)
        
        try:
            dataset = CARLA_Data(root=[ds_path], config=config, shared_dict=None)
            loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)
            ds_totals = {'shift': 0.0, 'lat_err': 0.0, 'lon_err': 0.0, 'drift': 0.0, 
                         'lpips': 0.0, 'ssim': 0.0, 'clean_len': 0.0, 'success': 0, 'time': 0.0}
            count = 0
            
            for batch_idx, data in enumerate(tqdm(loader, desc=f"Attack {ds_name}")):
                t0 = time.time()
                
                rgb_raw = data['rgb'].to(device, dtype=torch.float32)
                if rgb_raw.shape[-2:] != (160, 704):
                    rgb = torch.nn.functional.interpolate(rgb_raw, size=(160, 704), mode='bilinear', align_corners=False)
                else:
                    rgb = rgb_raw
                
                bs = 1 
                dummy_lidar = torch.zeros(bs, 2, 256, 256).to(device, dtype=torch.float32)
                speed = data['speed'].to(device, dtype=torch.float32) if 'speed' in data else torch.tensor([4.0]).to(device)
                target_point = data['target_point'].to(device, dtype=torch.float32) if 'target_point' in data else torch.tensor([[20.0, 0.0]]).to(device)
                ego_wp = data['ego_waypoint'].to(device, dtype=torch.float32) if 'ego_waypoint' in data else torch.zeros(bs, 10, 2).to(device)

                batch = {
                    'lidar': dummy_lidar,       
                    'speed': speed,             
                    'target_point': target_point,
                    'ego_waypoint': ego_wp,     
                    'bev_points': None,
                    'cam_points': None,
                    'target_point_image': torch.zeros(bs, 1, 256, 256).to(device)
                }

                try:
                    # Run PerC-AL Attack
                    rgb_adv, p_clean, p_adv, drift, lp, ss = attacker.run_attack(rgb, batch)
                    
                    step_time = time.time() - t0
                    
                    clean_wp = p_clean[0].cpu().numpy()
                    adv_wp = p_adv[0].cpu().numpy()
                    shift = np.linalg.norm(clean_wp - adv_wp)
                    lat = abs(adv_wp[-1][1] - clean_wp[-1][1])
                    lon = abs(adv_wp[-1][0] - clean_wp[-1][0])
                    c_len = np.linalg.norm(clean_wp[-1])
                    is_succ = 1.0 if shift >= 1.0 else 0.0
                    
                    ds_totals['shift']+=shift; ds_totals['lat_err']+=lat; ds_totals['lon_err']+=lon
                    ds_totals['drift']+=drift; ds_totals['lpips']+=lp; ds_totals['ssim']+=ss
                    ds_totals['clean_len']+=c_len; ds_totals['success']+=is_succ
                    ds_totals['time']+=step_time
                    count += 1
                    
                    # === 保存逻辑 (MFA Style) ===
                    filename = f"{ds_name}_{batch_idx:04d}.png"
                    
                    Image.fromarray(rgb[0].permute(1,2,0).cpu().numpy().astype(np.uint8)).save(
                        os.path.join(clean_save_dir, filename)
                    )
                    Image.fromarray(rgb_adv[0].permute(1,2,0).cpu().numpy().astype(np.uint8)).save(
                        os.path.join(adv_save_dir, filename)
                    )
                    
                    Image.fromarray(rgb[0].permute(1,2,0).cpu().numpy().astype(np.uint8)).save(
                        os.path.join(global_fid_c, filename)
                    )
                    Image.fromarray(rgb_adv[0].permute(1,2,0).cpu().numpy().astype(np.uint8)).save(
                        os.path.join(global_fid_a, filename)
                    )

                    if batch_idx % 20 == 0:
                        vis_filename = f"{ds_name}_{batch_idx:04d}_vis.png"
                        save_visualization_unified(
                            rgb[0], rgb_adv[0], clean_wp, adv_wp, 
                            data['ego_waypoint'][0].cpu().numpy() if 'ego_waypoint' in data else None, 
                            batch['target_point'][0].cpu().numpy(), 
                            os.path.join(vis_save_dir, vis_filename), 
                            drift, title_prefix=f"{ds_name} #{batch_idx}"
                        )
                except Exception as e:
                    print(f"Error processing {ds_name} batch {batch_idx}: {e}")
                    import traceback
                    traceback.print_exc()
                torch.cuda.empty_cache()

            if count > 0:
                avg_shift = ds_totals['shift'] / count
                ds_tan = ds_totals['lat_err'] / (ds_totals['clean_len'] + 1e-6)
                
                global_stats['shift'].append(avg_shift)
                global_stats['lat_err'].append(ds_totals['lat_err'] / count)
                global_stats['lon_err'].append(ds_totals['lon_err'] / count)
                global_stats['sim_drift'].append(ds_totals['drift'] / count)
                global_stats['lpips'].append(ds_totals['lpips'] / count)
                global_stats['ssim'].append(ds_totals['ssim'] / count)
                global_stats['success'].append((ds_totals['success'] / count) * 100)
                global_stats['avg_time'].append(ds_totals['time'] / count)
                global_stats['global_tan'].append(ds_tan)
                
                total_processed += count
                
                print("-" * 50)
                print(f"Dataset Summary: {ds_name}")
                print(f"  > Route Shift:    {avg_shift:.4f} m")
                print(f"  > Success Rate:   {(ds_totals['success']/count)*100:.2f} %")
                print("-" * 50)
                
        except Exception as e_ds:
            print(f"Skipping dataset {ds_name} due to error: {e_ds}")
            import traceback
            traceback.print_exc()
        
        gc.collect()
        torch.cuda.empty_cache()

    # Final Macro Report
    if len(global_stats['shift']) > 0:
        elapsed = time.time() - start_time
        def get_avg(k): return sum(global_stats[k]) / len(global_stats[k])
        
        print("\n" + "="*60)
        print("MACRO AVERAGE METRICS (PerC-AL - MFA Style):")
        print(f"  > Total Routes:         {len(datasets)}")
        print(f"  > Total Samples:        {total_processed}")
        print(f"  > Total Time:           {timedelta(seconds=int(elapsed))}")
        print(f"  > Avg Time Per Image:   {get_avg('avg_time'):.3f} s")
        print(f"  > Avg Route Shift:      {get_avg('shift'):.4f} m")
        print(f"  > Avg Success Rate:     {get_avg('success'):.2f} %")
        print(f"  > Avg LPIPS:            {get_avg('lpips'):.4f}")
        print(f"  > Avg SSIM:             {get_avg('ssim'):.4f}")
        
        print("-" * 40)
        try:
            del attacker, model
            gc.collect()
            torch.cuda.empty_cache()
            
            print("Calculating Global FID (using torch-fidelity)...")
            metrics = calculate_metrics(
                input1=global_fid_c, 
                input2=global_fid_a, 
                cuda=True, 
                isc=False, 
                fid=True, 
                verbose=False
            )
            print(f"  > Global FID:           {metrics['frechet_inception_distance']:.4f}")
        except Exception as e:
            print(f"  > FID Calculation Failed: {e}")
            
        print("="*60)

if __name__ == "__main__":
    main()