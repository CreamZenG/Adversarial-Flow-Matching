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
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from tqdm import tqdm
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

# 引入项目中的模块 (保持不变)
from config import GlobalConfig
from model import LidarCenterNet
from data import CARLA_Data

# 引入评估库
try:
    import lpips
    from torch_fidelity import calculate_metrics
except ImportError:
    print("Warning: lpips or torch-fidelity not found.")

# ================= NCF 攻击配置 (对齐 openloop_gray_NCF.py) =================
ATTACK_PARAMS = {
    # === White-box NCF 攻击参数 (Gradient-based) ===
    
    # Phase 1: Initialization Reset (IR)
    "num_reset": 20,         # 随机初始化尝试次数
    
    # Phase 2: Neighborhood Search (NS) - Gradient Descent
    "num_iter": 100,          # 梯度下降迭代次数
    "lr_T": 0.05,            # T 矩阵的学习率
    "lr_mu": 0.05,           # 均值偏移的学习率
    "momentum": 0.9,         # 动量
    
    # === 约束参数 ===
    "lambda_sim": 10.0,      # 攻击 Loss 权重 (Feature Drift)
    "lambda_reg": 0.1,       # T 矩阵正则化权重
    "lambda_mu": 0.05,       # Mu 偏移正则化权重
    
    # === 限制幅度 ===
    "epsilon": 16.0 / 255,   # RGB 空间的 L_inf 约束
    "mu_limit": 0.3,         # Lab 空间均值偏移限制
    
    "device": "cuda:0",
}

# ================= 可微分色彩转换 (从 openloop_gray_NCF.py 移植) =================
# NCF 攻击变量计算必须保持 Float32 以确保数值稳定和梯度流

def rgb_to_lab_differentiable(rgb):
    eps = 1e-6
    rgb = torch.clamp(rgb, eps, 1.0) 
    
    mask = (rgb > 0.04045).float()
    rgb_linear = mask * (((rgb + 0.055) / 1.055) ** 2.4) + (1 - mask) * (rgb / 12.92)
    
    r, g, b = rgb_linear[:, 0, :, :], rgb_linear[:, 1, :, :], rgb_linear[:, 2, :, :]
    
    x = 0.4124564 * r + 0.3575761 * g + 0.1804375 * b
    y = 0.2126729 * r + 0.7151522 * g + 0.0721750 * b
    z = 0.0193339 * r + 0.1191920 * g + 0.9503041 * b
    
    xn, yn, zn = 0.95047, 1.00000, 1.08883
    x, y, z = x / xn, y / yn, z / zn
    
    mask_xyz = (y > 0.008856).float()
    
    def f(t):
        t = torch.clamp(t, eps, None)
        return mask_xyz * (t ** (1/3)) + (1 - mask_xyz) * (7.787 * t + 16/116)
        
    fx, fy, fz = f(x), f(y), f(z)
    
    L = 116 * fy - 16
    a = 500 * (fx - fy)
    b_ch = 200 * (fy - fz)
    
    L_norm = L / 100.0
    a_norm = (a + 128) / 255.0
    b_norm = (b_ch + 128) / 255.0
    
    return torch.stack([L_norm, a_norm, b_norm], dim=1)

def lab_to_rgb_differentiable(lab):
    L_norm, a_norm, b_norm = lab[:, 0, :, :], lab[:, 1, :, :], lab[:, 2, :, :]
    
    L = L_norm * 100.0
    a = a_norm * 255.0 - 128
    b_ch = b_norm * 255.0 - 128
    
    fy = (L + 16) / 116
    fx = a / 500 + fy
    fz = fy - b_ch / 200
    
    xn, yn, zn = 0.95047, 1.00000, 1.08883
    
    def f_inv(t):
        mask = (t > 0.2068966).float() 
        return mask * (t ** 3) + (1 - mask) * (3 * (6/29)**2 * (t - 4/29))
        
    x = f_inv(fx) * xn
    y = f_inv(fy) * yn
    z = f_inv(fz) * zn
    
    r = 3.2404542 * x - 1.5371385 * y - 0.4985314 * z
    g = -0.9692660 * x + 1.8760108 * y + 0.0415560 * z
    b = 0.0556434 * x - 0.2040259 * y + 1.0572252 * z
    
    rgb_linear = torch.stack([r, g, b], dim=1)
    rgb_linear = torch.clamp(rgb_linear, 1e-6, 1.0)
    
    mask = (rgb_linear > 0.0031308).float()
    rgb = mask * (1.055 * (rgb_linear ** (1/2.4)) - 0.055) + (1 - mask) * (12.92 * rgb_linear)
    
    return torch.clamp(rgb, 0, 1)

# ================= 评估工具类 (保持不变) =================
def find_valid_datasets(base_path):
    valid_roots = []
    for root, dirs, files in os.walk(base_path):
        has_routes = False
        for d in dirs:
            if os.path.exists(os.path.join(root, d, 'measurements')):
                has_routes = True; break
        if has_routes: valid_roots.append(root)
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

# ================= 攻击者类 (TransFuser + Gradient NCF) =================
class TransFuserNCFAttacker:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self._hook_features = {}
        self._register_hook()
        
        try:
            self.lpips_vgg = lpips.LPIPS(net='vgg').to(device).eval()
        except:
            self.lpips_vgg = None
        self.ssim_metric = SSIM().to(device)
        
        # 冻结模型参数 (对于白盒攻击，我们更新输入，不需要模型梯度更新)
        for param in self.model.parameters():
            param.requires_grad = False
            
    def _register_hook(self):
        def hook_fn(module, input, output):
            self._hook_features['transformer1'] = output
        
        # TransFuser 的结构通常是 self.model._model.transformer1
        if hasattr(self.model, '_model') and hasattr(self.model._model, 'transformer1'):
            target_layer = self.model._model.transformer1
            target_layer.register_forward_hook(hook_fn)
        else:
            raise AttributeError("Error: Could not find 'transformer1'.")

    def run_attack(self, rgb_raw, lidar, target_point, target_point_image, ego_vel, bev_points=None, cam_points=None, num_points=None):
        """
        Gradient-based NCF Attack for TransFuser.
        Optimizes Color Transformation Matrix (T) and Shift (mu).
        """
        cfg = ATTACK_PARAMS
        device = self.device
        batch_size, _, H, W = rgb_raw.shape

        # === 1. Prepare Inputs (Float32 [0,1]) ===
        # rgb_raw is [B, 3, H, W] in [0, 255]
        img_tensor = rgb_raw.float() / 255.0
        
        # Dummy inputs for full forward pass
        dummy_bev = torch.zeros(batch_size, 160, 160).to(self.device).long()
        dummy_label = torch.zeros(batch_size, 1, 7).to(self.device).float()
        dummy_depth = torch.zeros(batch_size, 160, 704).to(self.device).float()
        dummy_semantic = torch.zeros(batch_size, 160, 704).to(self.device).long()
        dummy_wp = torch.zeros(batch_size, 4, 2).to(self.device).float()

        # === 2. Clean Pass (Get Target Features) ===
        self._hook_features = {} 
        with torch.no_grad():
            pred_clean, _ = self.model.forward_ego(
                rgb_raw, lidar, target_point, target_point_image, ego_vel,
                bev_points=bev_points, cam_points=cam_points, num_points=num_points
            )
            # Ensure hook caught features (if forward_ego skips hooks, call main forward)
            if 'transformer1' not in self._hook_features:
                 self.model(rgb_raw, lidar, dummy_wp, target_point, target_point_image, ego_vel, 
                       dummy_bev, dummy_label, dummy_depth, dummy_semantic, num_points=num_points)
            
            clean_out = self._hook_features['transformer1']
            # Feature [0] is image features, [1] is lidar features
            feat_clean_img = clean_out[0].clone().detach()
            feat_clean_lidar = clean_out[1].clone().detach()

        # === 3. Pre-computation (Lab Space) ===
        with torch.no_grad():
            lab_clean = rgb_to_lab_differentiable(img_tensor)
            mu_clean = torch.mean(lab_clean, dim=[2, 3], keepdim=True) 
            # [B, 3, H, W] -> [B, H, W, 3] for matrix multiplication
            lab_centered = (lab_clean - mu_clean).permute(0, 2, 3, 1).reshape(batch_size, -1, 3)

        # ------------------------------------------------------------------
        # Phase 1: Initialization Reset (IR) - Random Search
        # ------------------------------------------------------------------
        best_T_init = torch.eye(3, device=device).unsqueeze(0).repeat(batch_size, 1, 1)
        best_mu_init = torch.zeros((batch_size, 3, 1, 1), device=device)
        best_score_init = 1.0 # Cosine similarity (1.0 is max similarity)

        for _ in range(cfg['num_reset']):
            with torch.no_grad():
                # 1. Random T
                cand_T = torch.eye(3, device=device).unsqueeze(0).repeat(batch_size, 1, 1)
                scale = torch.rand(batch_size, 3, device=device) * 1.0 + 0.5
                cand_T[:, 0, 0] = scale[:, 0]
                cand_T[:, 1, 1] = scale[:, 1]
                cand_T[:, 2, 2] = scale[:, 2]
                noise_T = torch.randn_like(cand_T) * 0.1
                noise_T[:, 0, 1:] = 0; noise_T[:, 1:, 0] = 0
                cand_T = cand_T + noise_T
                
                # 2. Random Mu
                cand_mu = torch.zeros((batch_size, 3, 1, 1), device=device)
                cand_mu[:, 0] = (torch.rand(batch_size, 1, 1, device=device) - 0.5) * 0.4
                cand_mu[:, 1:] = (torch.rand(batch_size, 2, 1, 1, device=device) - 0.5) * 0.6
                
                # 3. Apply Transform
                lab_adv_flat = torch.bmm(lab_centered, cand_T)
                lab_adv = lab_adv_flat.reshape(batch_size, H, W, 3).permute(0, 3, 1, 2)
                lab_adv = lab_adv + mu_clean + cand_mu
                
                adv_img = lab_to_rgb_differentiable(lab_adv)
                adv_img = torch.clamp(adv_img, 0, 1)
                
                delta = adv_img - img_tensor
                delta = torch.clamp(delta, -cfg['epsilon'], cfg['epsilon'])
                adv_img = torch.clamp(img_tensor + delta, 0, 1)
                
                # 4. Forward & Score
                self._hook_features = {}
                self.model(adv_img * 255.0, lidar, dummy_wp, target_point, target_point_image, ego_vel, 
                           dummy_bev, dummy_label, dummy_depth, dummy_semantic, num_points=num_points)
                
                curr_feat_img = self._hook_features['transformer1'][0]
                # Minimize Cosine Similarity = Attack Success
                score = F.cosine_similarity(curr_feat_img.flatten(1), feat_clean_img.flatten(1)).mean().item()
                
                if score < best_score_init:
                    best_score_init = score
                    best_T_init = cand_T.clone()
                    best_mu_init = cand_mu.clone()

        # ------------------------------------------------------------------
        # Phase 2: Neighborhood Search (NS) - Gradient Descent
        # ------------------------------------------------------------------
        T_matrix = best_T_init.detach().requires_grad_(True)
        delta_mu = best_mu_init.detach().requires_grad_(True)

        optimizer = torch.optim.SGD([
            {'params': [T_matrix], 'lr': cfg['lr_T']},
            {'params': [delta_mu], 'lr': cfg['lr_mu']}
        ], momentum=cfg['momentum'])

        best_adv_img = img_tensor.clone().detach()
        min_sim_score = float('inf')

        for i in range(cfg['num_iter']):
            # A. Apply NCF Transform
            lab_adv_flat = torch.bmm(lab_centered, T_matrix)
            lab_adv = lab_adv_flat.reshape(batch_size, H, W, 3).permute(0, 3, 1, 2)
            
            delta_mu_clamped = torch.clamp(delta_mu, -cfg['mu_limit'], cfg['mu_limit'])
            lab_adv = lab_adv + mu_clean + delta_mu_clamped
            
            adv_img = lab_to_rgb_differentiable(lab_adv)
            
            # Epsilon Constraint
            delta_rgb = adv_img - img_tensor
            delta_rgb = torch.clamp(delta_rgb, -cfg['epsilon'], cfg['epsilon'])
            adv_img_constrained = torch.clamp(img_tensor + delta_rgb, 0, 1)
            
            # B. Forward Pass (Scale to 0-255 for model)
            self._hook_features = {}
            # Need to ensure gradients flow through adv_img_constrained
            self.model(adv_img_constrained * 255.0, lidar, dummy_wp, target_point, target_point_image, ego_vel, 
                       dummy_bev, dummy_label, dummy_depth, dummy_semantic, num_points=num_points)
            
            feat_adv_img = self._hook_features['transformer1'][0]
            feat_adv_lidar = self._hook_features['transformer1'][1]

            # C. Loss Calculation
            # Attack Loss: Minimize Cosine Similarity (Untargeted)
            sim_loss = F.cosine_similarity(feat_adv_img.flatten(1), feat_clean_img.flatten(1)).mean()
            # Note: We attack image features primarily as NCF modifies image
            
            # Regularization
            I = torch.eye(3, device=device).unsqueeze(0)
            reg_T = torch.norm(T_matrix - I, p='fro')
            reg_mu = torch.norm(delta_mu, p=2)
            
            loss = cfg['lambda_sim'] * sim_loss + \
                   cfg['lambda_reg'] * reg_T + \
                   cfg['lambda_mu'] * reg_mu

            # D. Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_([T_matrix, delta_mu], 1.0)
            optimizer.step()
            
            # E. Track Best
            curr_sim = sim_loss.item()
            if i == 0 or curr_sim < min_sim_score:
                min_sim_score = curr_sim
                best_adv_img = adv_img_constrained.detach()

        # === 4. Final Outputs ===
        rgb_adv_255 = best_adv_01 = best_adv_img * 255.0
        
        with torch.no_grad():
            pred_wp_adv, _ = self.model.forward_ego(
                rgb_adv_255, lidar, target_point, target_point_image, ego_vel,
                bev_points=bev_points, cam_points=cam_points, num_points=num_points
            )
            # Re-calc final feature drift
            self._hook_features = {}
            self.model(rgb_adv_255, lidar, dummy_wp, target_point, target_point_image, ego_vel, 
                       dummy_bev, dummy_label, dummy_depth, dummy_semantic, num_points=num_points)
            final_out = self._hook_features['transformer1']
            
            # MSE Drift
            drift_img = F.mse_loss(final_out[0], feat_clean_img)
            drift_lidar = F.mse_loss(final_out[1], feat_clean_lidar)
            sim_drift = (drift_img + drift_lidar).item()

            # Metrics
            img_clean_norm = (img_tensor * 2) - 1
            img_adv_norm = (best_adv_img * 2) - 1
            
            lpips_val = 0.0
            if self.lpips_vgg is not None:
                lpips_val = self.lpips_vgg(img_adv_norm, img_clean_norm).mean().item()
            ssim_val = self.ssim_metric(img_adv_norm, img_clean_norm).item()

        return rgb_adv_255, pred_clean, pred_wp_adv, sim_drift, lpips_val, ssim_val

# ================= 可视化函数 (保持不变) =================
def save_visualization_1x4(rgb_clean, rgb_adv, pred_clean, pred_adv, gt_wp, target_point, save_path, frame_id, metrics_str):
    img_clean_np = rgb_clean.permute(1, 2, 0).detach().cpu().numpy().astype(np.float32) / 255.0
    img_adv_np = rgb_adv.permute(1, 2, 0).detach().cpu().numpy().astype(np.float32) / 255.0
    
    img_clean_np = np.clip(img_clean_np, 0, 1)
    img_adv_np = np.clip(img_adv_np, 0, 1)
    
    # 噪声增强显示 (x50)
    noise = np.abs(img_adv_np - img_clean_np) * 50.0
    noise = np.clip(noise, 0, 1)

    fig = plt.figure(figsize=(15, 6))
    gs = fig.add_gridspec(1, 4)
    
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(img_clean_np); ax1.set_title("Clean"); ax1.axis('off')
    
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(img_adv_np); ax2.set_title("Adv (Gradient NCF)"); ax2.axis('off')
    
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
    parser.add_argument('--output_dir', type=str, default='ncf_grad_results')
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

    print(f"=== TransFuser NCF (Gradient-based) Attack ===")
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
    
    # 使用 Gradient NCF Attacker
    attacker = TransFuserNCFAttacker(model, device)

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
            t0 = time.time() # 开始计时
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

            # 运行 NCF 攻击
            try:
                rgb_adv, pred_clean, pred_adv, drift, lpips_val, ssim_val = attacker.run_attack(
                    rgb, lidar, target_point, tp_img, ego_vel, 
                    bev_points=bev_points, cam_points=cam_points
                )
            except Exception as e:
                print(f"Attack failed for batch {batch_idx}: {e}")
                import traceback; traceback.print_exc()
                continue
                
            step_time = time.time() - t0 # 结束计时

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
        print("MACRO AVERAGE METRICS (TransFuser Gradient NCF):")
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