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

# ================= 攻击配置 =================
ATTACK_PARAMS = {
    "epsilon": 8/255,       # 扰动强度
    "steps": 40,            # PGD 迭代次数
    "alpha": 2/255,         # PGD 步长
    "device": "cuda:1",     # 默认设备
    
    # === MFA Loss 权重 ===
    "w_mse": 3.0,           # 特征距离权重 (MSE)
    "w_cos": 1.5,           # 方向差异权重 (Cosine)
    "w_attn": 6.0,          # 注意力加权权重 (Attention)
    "attn_temp": 6.0        # 注意力温度
}

# ================= Monkey Patching (捕获 Attention Map) =================
def list_capturing_forward(self, x):
    B, T, C = x.size()

    k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
    q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
    v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

    # 1. 计算 Attention Map
    att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
    att = F.softmax(att, dim=-1)
    
    # [Hack] 保存 Attention Map 到实例变量
    self.stored_att = att # Shape: [B, n_head, T, T]

    att = self.attn_drop(att)
    y = att @ v
    y = y.transpose(1, 2).contiguous().view(B, T, C)
    y = self.resid_drop(self.proj(y))
    return y

# ================= MFA Loss Helper =================
def get_transfuser_attention_loss(curr_img_feat, curr_lidar_feat, target_img_feat, target_lidar_feat, clean_attn_map, temperature):
    """
    计算基于 Attention 权重的 MSE Loss。
    """
    if clean_attn_map is None: return 0.0

    b, c, h_img, w_img = curr_img_feat.shape
    _, _, h_lid, w_lid = curr_lidar_feat.shape
    
    # 展平并拼接特征 [B, N, C]
    feat_img_flat = curr_img_feat.view(b, c, -1).permute(0, 2, 1)
    feat_lid_flat = curr_lidar_feat.view(b, c, -1).permute(0, 2, 1)
    feat_full = torch.cat([feat_img_flat, feat_lid_flat], dim=1)
    
    target_img_flat = target_img_feat.view(b, c, -1).permute(0, 2, 1)
    target_lid_flat = target_lidar_feat.view(b, c, -1).permute(0, 2, 1)
    target_full = torch.cat([target_img_flat, target_lid_flat], dim=1)
    
    # 计算 Token 重要性 (mean over heads & queries) -> [B, N]
    importance = clean_attn_map.mean(dim=1).mean(dim=1) 
    importance = F.softmax(importance * temperature, dim=-1).unsqueeze(-1) # [B, N, 1]
    
    # 加权 MSE
    diff_sq = (feat_full - target_full) ** 2
    loss = (diff_sq * importance).sum(dim=1).mean()
    return loss

# ================= 辅助函数 =================
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

# ================= PGD 攻击者 (MFA Enabled) =================
class TransFuserPGDAttacker:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self._hook_features = {}
        self._register_hook()
        
        # [关键] Monkey Patching for Attention Map
        if hasattr(self.model, '_model') and hasattr(self.model._model, 'transformer1'):
            tf_blocks = self.model._model.transformer1.blocks
            for i, block in enumerate(tf_blocks):
                block.attn.forward = types.MethodType(list_capturing_forward, block.attn)
            print(f">> Monkey-Patch applied to {len(tf_blocks)} Attention layers.")
        else:
             print("Warning: Attention Patching Failed. 'w_attn' will be ignored.")

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
            print(">> Hook registered on _model.transformer1")
        else:
            raise AttributeError("Error: Could not find 'transformer1'.")

    def run_attack(self, rgb_raw, lidar, target_point, target_point_image, ego_vel, bev_points=None, cam_points=None, num_points=None):
        """
        PGD-MFA Attack
        """
        eps_255 = ATTACK_PARAMS['epsilon'] * 255.0
        alpha_255 = ATTACK_PARAMS['alpha'] * 255.0
        steps = ATTACK_PARAMS['steps']
        
        bs = rgb_raw.shape[0]
        # Dummy inputs
        dummy_bev = torch.zeros(bs, 160, 160).to(self.device).long()
        dummy_label = torch.zeros(bs, 1, 7).to(self.device).float()
        dummy_depth = torch.zeros(bs, 160, 704).to(self.device).float()
        dummy_semantic = torch.zeros(bs, 160, 704).to(self.device).long()
        dummy_wp = torch.zeros(bs, 4, 2).to(self.device).float()

        # 1. Clean Pass (Target Features & Attention)
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
            
            # 获取 Clean Attention Map (取最后一层)
            clean_attn_map = None
            try:
                if hasattr(self.model._model.transformer1.blocks[-1].attn, 'stored_att'):
                    clean_attn_map = self.model._model.transformer1.blocks[-1].attn.stored_att.clone().detach()
            except Exception: pass

        # 2. PGD Initialization (Random Start)
        rgb_adv = rgb_raw.clone().detach()
        noise = torch.zeros_like(rgb_adv).uniform_(-eps_255, eps_255)
        rgb_adv = torch.clamp(rgb_adv + noise, 0, 255)
        
        mse_loss = nn.MSELoss(reduction='mean')

        # 3. PGD Loop
        for i in range(steps):
            rgb_adv.requires_grad = True
            self._hook_features = {} 

            self.model(rgb_adv, lidar, dummy_wp, target_point, target_point_image, ego_vel, 
                       dummy_bev, dummy_label, dummy_depth, dummy_semantic, num_points=num_points)
            
            adv_out = self._hook_features['transformer1']
            curr_img_feat = adv_out[0]
            curr_lidar_feat = adv_out[1]
            
            # --- MFA Loss Calculation ---
            # 1. MSE
            loss_mse = mse_loss(curr_img_feat, target_feat_img) + mse_loss(curr_lidar_feat, target_feat_lidar)
            
            # 2. Cosine
            flat_adv = curr_img_feat.view(bs, -1)
            flat_tgt = target_feat_img.view(bs, -1)
            loss_cos = F.cosine_similarity(flat_adv, flat_tgt, dim=-1).mean()
            
            # 3. Attention
            loss_attn = 0.0
            if clean_attn_map is not None:
                loss_attn = get_transfuser_attention_loss(
                    curr_img_feat, curr_lidar_feat, 
                    target_feat_img, target_feat_lidar, 
                    clean_attn_map, 
                    temperature=ATTACK_PARAMS['attn_temp']
                )
            
            # Objective: Maximize MSE & Attn_MSE, Minimize Cosine
            total_loss = - ATTACK_PARAMS['w_mse'] * loss_mse \
                         + ATTACK_PARAMS['w_cos'] * loss_cos \
                         - ATTACK_PARAMS['w_attn'] * loss_attn
            
            self.model.zero_grad()
            total_loss.backward()
            
            grad = rgb_adv.grad.data
            
            if grad is not None:
                # PGD Update
                rgb_adv = rgb_adv - alpha_255 * grad.sign()
                perturbation = rgb_adv - rgb_raw
                perturbation = torch.clamp(perturbation, -eps_255, eps_255)
                rgb_adv = rgb_raw + perturbation
                rgb_adv = torch.clamp(rgb_adv, 0, 255).detach()
            else:
                break

        # 4. Final Evaluation
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
            
            sim_drift = (mse_loss(final_out[0], target_feat_img) + mse_loss(final_out[1], target_feat_lidar)).item()
            
            img_clean_norm = (rgb_raw / 255.0) * 2 - 1
            img_adv_norm = (rgb_adv / 255.0) * 2 - 1
            
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
    noise = np.abs(img_adv_np - img_clean_np) * 50.0
    noise = np.clip(noise, 0, 1)

    fig = plt.figure(figsize=(15, 6))
    gs = fig.add_gridspec(1, 4)
    
    ax1 = fig.add_subplot(gs[0, 0]); ax1.imshow(img_clean_np); ax1.set_title("Clean"); ax1.axis('off')
    ax2 = fig.add_subplot(gs[0, 1]); ax2.imshow(img_adv_np); ax2.set_title("Adv (PGD-MFA)"); ax2.axis('off')
    ax3 = fig.add_subplot(gs[0, 2]); ax3.imshow(noise); ax3.set_title("Noise (x50)"); ax3.axis('off')
    
    ax4 = fig.add_subplot(gs[0, 3])
    if gt_wp is not None:
        ax4.plot(gt_wp[:, 1], gt_wp[:, 0], 'g-', alpha=0.5, linewidth=3, label='GT')
    ax4.plot(pred_clean[:, 1], pred_clean[:, 0], 'b-o', markersize=4, label='Clean')
    ax4.plot(pred_adv[:, 1], pred_adv[:, 0], 'r--^', markersize=4, label='Adv')
    if target_point is not None:
        ax4.plot(target_point[0], -target_point[1], 'k*', markersize=15, label='Target', zorder=10)
    ax4.plot(0, 0, 'k^', markersize=10, label='Ego')
    ax4.set_xlim(-10, 10); ax4.set_ylim(-2, 35)
    ax4.legend(loc='upper right'); ax4.grid(True, linestyle='--', alpha=0.5)
    ax4.set_title(f"Trajectory\n{metrics_str}", fontsize=10)
    ax4.set_aspect('equal')

    plt.tight_layout(); plt.savefig(save_path, dpi=100); plt.close(fig)

# ================= 主函数 =================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--root_dir', type=str, required=True)
    parser.add_argument('--backbone', type=str, default='transFuser')
    parser.add_argument('--image_architecture', type=str, default='regnety_032')
    parser.add_argument('--lidar_architecture', type=str, default='regnety_032')
    parser.add_argument('--output_dir', type=str, default='pgd_mfa_results')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    sys.stdout = Logger(os.path.join(args.output_dir, "attack_log.txt"))
    
    fid_clean_dir = os.path.join(args.output_dir, "fid_clean")
    fid_adv_dir = os.path.join(args.output_dir, "fid_adv")
    if os.path.exists(fid_clean_dir): shutil.rmtree(fid_clean_dir)
    if os.path.exists(fid_adv_dir): shutil.rmtree(fid_adv_dir)
    os.makedirs(fid_clean_dir, exist_ok=True)
    os.makedirs(fid_adv_dir, exist_ok=True)

    print(f"=== TransFuser PGD-MFA Attack ===")
    print(json.dumps(ATTACK_PARAMS, indent=2))
    
    device_str = ATTACK_PARAMS.get("device", "cuda:0")
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    if device.type == 'cuda': torch.cuda.set_device(device)
    print(f"Running on: {device}")

    # Load Model
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
    
    attacker = TransFuserPGDAttacker(model, device)

    # Process Datasets
    valid_datasets = find_valid_datasets(args.root_dir)
    global_stats = {
        'shift': [], 'lat_err': [], 'lon_err': [], 
        'sim_drift': [], 'lpips': [], 'ssim': [], 
        'clean_len': [], 'success': [], 'avg_time': [],
        'global_tan': []
    }
    
    overall_start = time.time()
    total_processed = 0

    for ds_idx, dataset_path in enumerate(valid_datasets):
        ds_name = os.path.basename(dataset_path) 
        print(f"\n[{ds_idx+1}/{len(valid_datasets)}] Processing: {ds_name}")
        ds_out_dir = os.path.join(args.output_dir, ds_name)
        os.makedirs(ds_out_dir, exist_ok=True)
        
        ds_totals = {'shift': 0.0, 'lat_err': 0.0, 'lon_err': 0.0, 'drift': 0.0, 'lpips': 0.0, 'ssim': 0.0, 'clean_len': 0.0, 'success': 0, 'time': 0.0}
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

            # Attack
            rgb_adv, pred_clean, pred_adv, drift, lpips_val, ssim_val = attacker.run_attack(
                rgb, lidar, target_point, tp_img, ego_vel, 
                bev_points=bev_points, cam_points=cam_points
            )
            step_time = time.time() - t0

            # Metrics
            clean_wp = pred_clean[0].cpu().numpy()
            adv_wp = pred_adv[0].cpu().numpy()
            shift = np.linalg.norm(clean_wp - adv_wp)
            lat_err = abs(adv_wp[-1][1] - clean_wp[-1][1])
            lon_err = abs(adv_wp[-1][0] - clean_wp[-1][0])
            clean_len = np.linalg.norm(clean_wp[-1])
            is_succ = 1.0 if shift >= 1.0 else 0.0

            ds_totals['shift'] += shift; ds_totals['lat_err'] += lat_err
            ds_totals['lon_err'] += lon_err; ds_totals['drift'] += drift
            ds_totals['lpips'] += lpips_val; ds_totals['ssim'] += ssim_val
            ds_totals['clean_len'] += clean_len; ds_totals['success'] += is_succ
            ds_totals['time'] += step_time
            count += 1
            
            # Save FID image
            def save_img(tensor, path):
                arr = tensor[0].permute(1, 2, 0).detach().cpu().numpy().astype(np.uint8)
                Image.fromarray(arr).save(path)
            fname = f"{ds_name}_{batch_idx:04d}.jpg"
            save_img(rgb, os.path.join(fid_clean_dir, fname))
            save_img(rgb_adv, os.path.join(fid_adv_dir, fname))

            # Visualize
            if batch_idx % 20 == 0:
                metrics_str = f"Shift: {shift:.2f}m | LatErr: {lat_err:.2f}m | SSIM: {ssim_val:.3f}"
                save_path = os.path.join(ds_out_dir, f"frame_{batch_idx:04d}.png")
                save_visualization_1x4(rgb[0], rgb_adv[0], clean_wp, adv_wp, gt_wp[0].cpu().numpy(), 
                                       target_point[0].cpu().numpy(), save_path, batch_idx, metrics_str)

        # Dataset Summary
        if count > 0:
            avg_shift = ds_totals['shift'] / count
            succ_rate = (ds_totals['success'] / count) * 100
            ds_tan = ds_totals['lat_err'] / (ds_totals['clean_len'] + 1e-6)
            ds_angle = math.degrees(math.atan(ds_tan))
            
            global_stats['shift'].append(avg_shift)
            global_stats['lat_err'].append(ds_totals['lat_err'] / count)
            global_stats['lon_err'].append(ds_totals['lon_err'] / count)
            global_stats['sim_drift'].append(ds_totals['drift'] / count)
            global_stats['lpips'].append(ds_totals['lpips'] / count)
            global_stats['ssim'].append(ds_totals['ssim'] / count)
            global_stats['clean_len'].append(ds_totals['clean_len'] / count)
            global_stats['success'].append(succ_rate)
            global_stats['avg_time'].append(ds_totals['time'] / count)
            global_stats['global_tan'].append(ds_tan)
            
            total_processed += count
            print("-" * 50)
            print(f"Summary: {ds_name} | Shift: {avg_shift:.4f}m | Drift: {ds_totals['drift']/count:.4f}")
            print(f"Tan: {ds_tan:.4f} ({ds_angle:.2f}°) | Success: {succ_rate:.2f}%")
            print("-" * 50)

    # Macro Summary
    elapsed = time.time() - overall_start
    if len(global_stats['shift']) > 0:
        def get_avg(k): return sum(global_stats[k]) / len(global_stats[k])
        avg_global_tan = get_avg('global_tan')
        avg_global_angle = math.degrees(math.atan(avg_global_tan))

        print("\n" + "="*60)
        print("MACRO AVERAGE METRICS (PGD-MFA):")
        print(f"  > Total Samples:        {total_processed}")
        print(f"  > Total Time:           {timedelta(seconds=int(elapsed))}")
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
            print("  > FID Calculation Failed")
        print("="*60)

if __name__ == "__main__":
    main()