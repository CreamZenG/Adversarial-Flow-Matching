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

# ================= 1. NCF 攻击配置 (适配 MFA 结构) =================
ATTACK_PARAMS = {
    # === Phase 1: Initialization Reset (IR) ===
    "num_reset": 20,         # 随机初始化尝试次数
    
    # === Phase 2: Neighborhood Search (NS) - Gradient Descent ===
    "num_iter": 100,          # 梯度下降迭代次数
    "lr_T": 0.05,            # 颜色矩阵 T 的学习率
    "lr_mu": 0.05,           # 偏移量 mu 的学习率
    "momentum": 0.9,         # 动量
    
    # === 约束与权重 ===
    "lambda_sim": 10.0,      # 攻击 Loss 权重 (Feature Drift)
    "lambda_reg": 0.1,       # T 矩阵正则化权重
    "lambda_mu": 0.05,       # Mu 偏移正则化权重
    
    "epsilon": 16.0 / 255,   # RGB 空间的 L_inf 约束
    "mu_limit": 0.3,         # Lab 空间均值偏移限制
    
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

# ================= 2. 可微分色彩转换 (NCF Core) =================
# 必须保持这些函数以支持梯度回传
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
    return torch.stack([L / 100.0, (a + 128) / 255.0, (b_ch + 128) / 255.0], dim=1)

def lab_to_rgb_differentiable(lab):
    L_norm, a_norm, b_norm = lab[:, 0, :, :], lab[:, 1, :, :], lab[:, 2, :, :]
    L, a, b_ch = L_norm * 100.0, a_norm * 255.0 - 128, b_norm * 255.0 - 128
    fy = (L + 16) / 116
    fx = a / 500 + fy
    fz = fy - b_ch / 200
    xn, yn, zn = 0.95047, 1.00000, 1.08883
    def f_inv(t):
        mask = (t > 0.2068966).float() 
        return mask * (t ** 3) + (1 - mask) * (3 * (6/29)**2 * (t - 4/29))
    x, y, z = f_inv(fx) * xn, f_inv(fy) * yn, f_inv(fz) * zn
    r = 3.2404542 * x - 1.5371385 * y - 0.4985314 * z
    g = -0.9692660 * x + 1.8760108 * y + 0.0415560 * z
    b = 0.0556434 * x - 0.2040259 * y + 1.0572252 * z
    rgb_linear = torch.clamp(torch.stack([r, g, b], dim=1), 1e-6, 1.0)
    mask = (rgb_linear > 0.0031308).float()
    rgb = mask * (1.055 * (rgb_linear ** (1/2.4)) - 0.055) + (1 - mask) * (12.92 * rgb_linear)
    return torch.clamp(rgb, 0, 1)

# ================= 3. NCF 攻击者类 =================
class TransFuserNCFAttacker:
    def __init__(self, transfuser_model, device):
        self.device = device
        self.tf_model = transfuser_model
        
        # Hook Setup
        self._hook_features = {}
        self._register_hook()

        try:
            self.lpips_vgg = lpips.LPIPS(net='vgg').to(device).eval()
        except: self.lpips_vgg = None
        self.ssim_loss = SSIM().to(device)

    def _register_hook(self):
        def hook_fn(module, input, output):
            self._hook_features['transformer1'] = output
        
        if hasattr(self.tf_model, '_model') and hasattr(self.tf_model._model, 'transformer1'):
            self.tf_model._model.transformer1.register_forward_hook(hook_fn)
        else:
            raise AttributeError("Error: Could not find 'transformer1' in TransFuser.")

    def fix_target_point_image(self, tp_img):
        if tp_img is None:
            return torch.zeros(1, 1, 256, 256).to(self.device, dtype=torch.float32)
        if tp_img.dim() == 4: return tp_img
        batch_size = tp_img.shape[0] if tp_img.shape[0] > 0 else 1
        return torch.zeros(batch_size, 1, 256, 256).to(self.device, dtype=torch.float32)

    def run_attack(self, rgb_raw, batch_data):
        """
        MFA 风格接口的 NCF 攻击
        rgb_raw: [B, 3, H, W], range [0, 255]
        """
        cfg = ATTACK_PARAMS
        bs = rgb_raw.shape[0]
        H, W = rgb_raw.shape[2], rgb_raw.shape[3]
        
        # Inputs [0, 1]
        img_tensor = rgb_raw.clone().float() / 255.0
        
        # 补全数据
        dummy_bev = torch.zeros(bs, 160, 160).to(self.device).long()
        dummy_label = torch.zeros(bs, 1, 7).to(self.device).float()
        dummy_depth = torch.zeros(bs, 160, 704).to(self.device).float()
        dummy_semantic = torch.zeros(bs, 160, 704).to(self.device).long()
        batch_data['target_point_image'] = self.fix_target_point_image(batch_data['target_point_image'])

        # 1. Clean Pass
        self._hook_features = {}
        with torch.no_grad():
            self.tf_model(
                rgb_raw, batch_data['lidar'], batch_data['ego_waypoint'], batch_data['target_point'], 
                batch_data['target_point_image'], batch_data['speed'].reshape(-1,1),
                dummy_bev, dummy_label, dummy_depth, dummy_semantic
            )
            feat_clean_img = self._hook_features['transformer1'][0].clone().detach()

            pred_wp_clean, _ = self.tf_model.forward_ego(
                rgb_raw, batch_data['lidar'], batch_data['target_point'], 
                batch_data['target_point_image'], batch_data['speed'].reshape(-1,1),
                bev_points=batch_data.get('bev_points'), cam_points=batch_data.get('cam_points')
            )
            
            # Pre-compute Lab
            lab_clean = rgb_to_lab_differentiable(img_tensor)
            mu_clean = torch.mean(lab_clean, dim=[2, 3], keepdim=True)
            lab_centered = (lab_clean - mu_clean).permute(0, 2, 3, 1).reshape(bs, -1, 3)

        # 2. Phase 1: Initialization Reset (Random Search)
        best_T = torch.eye(3, device=self.device).unsqueeze(0).repeat(bs, 1, 1)
        best_mu = torch.zeros((bs, 3, 1, 1), device=self.device)
        best_score = 1.0 # Minimize Cosine Similarity

        for _ in range(cfg['num_reset']):
            with torch.no_grad():
                # Random T
                cand_T = torch.eye(3, device=self.device).unsqueeze(0).repeat(bs, 1, 1)
                scale = torch.rand(bs, 3, device=self.device) * 1.0 + 0.5
                cand_T[:, 0, 0] = scale[:, 0]; cand_T[:, 1, 1] = scale[:, 1]; cand_T[:, 2, 2] = scale[:, 2]
                noise_T = torch.randn_like(cand_T) * 0.1
                noise_T[:, 0, 1:] = 0; noise_T[:, 1:, 0] = 0
                cand_T += noise_T
                
                # Random Mu
                cand_mu = torch.zeros((bs, 3, 1, 1), device=self.device)
                cand_mu[:, 0] = (torch.rand(bs, 1, 1, device=self.device) - 0.5) * 0.4
                cand_mu[:, 1:] = (torch.rand(bs, 2, 1, 1, device=self.device) - 0.5) * 0.6
                
                # Apply
                lab_adv_flat = torch.bmm(lab_centered, cand_T)
                lab_adv = lab_adv_flat.reshape(bs, H, W, 3).permute(0, 3, 1, 2) + mu_clean + cand_mu
                adv_img = torch.clamp(lab_to_rgb_differentiable(lab_adv), 0, 1)
                
                # Constraint
                delta = torch.clamp(adv_img - img_tensor, -cfg['epsilon'], cfg['epsilon'])
                adv_img = torch.clamp(img_tensor + delta, 0, 1)
                
                # Forward
                self._hook_features = {}
                self.tf_model(
                    adv_img * 255.0, batch_data['lidar'], batch_data['ego_waypoint'], batch_data['target_point'], 
                    batch_data['target_point_image'], batch_data['speed'].reshape(-1,1),
                    dummy_bev, dummy_label, dummy_depth, dummy_semantic
                )
                curr_feat = self._hook_features['transformer1'][0]
                score = F.cosine_similarity(curr_feat.flatten(1), feat_clean_img.flatten(1)).mean().item()
                
                if score < best_score:
                    best_score = score
                    best_T = cand_T.clone()
                    best_mu = cand_mu.clone()

        # 3. Phase 2: Gradient Descent
        T_matrix = best_T.detach().requires_grad_(True)
        delta_mu = best_mu.detach().requires_grad_(True)
        optimizer = optim.SGD([{'params': [T_matrix], 'lr': cfg['lr_T']},
                               {'params': [delta_mu], 'lr': cfg['lr_mu']}], momentum=cfg['momentum'])

        best_adv_img = img_tensor.clone().detach()
        min_sim_score = float('inf')

        for i in range(cfg['num_iter']):
            lab_adv_flat = torch.bmm(lab_centered, T_matrix)
            lab_adv = lab_adv_flat.reshape(bs, H, W, 3).permute(0, 3, 1, 2)
            
            delta_mu_clamped = torch.clamp(delta_mu, -cfg['mu_limit'], cfg['mu_limit'])
            lab_adv = lab_adv + mu_clean + delta_mu_clamped
            adv_img = lab_to_rgb_differentiable(lab_adv)
            
            # Epsilon Constraint
            delta_rgb = torch.clamp(adv_img - img_tensor, -cfg['epsilon'], cfg['epsilon'])
            adv_img_constrained = torch.clamp(img_tensor + delta_rgb, 0, 1)
            
            # Forward
            self._hook_features = {}
            self.tf_model(
                adv_img_constrained * 255.0, batch_data['lidar'], batch_data['ego_waypoint'], batch_data['target_point'], 
                batch_data['target_point_image'], batch_data['speed'].reshape(-1,1),
                dummy_bev, dummy_label, dummy_depth, dummy_semantic
            )
            feat_adv = self._hook_features['transformer1'][0]
            
            # Loss
            sim_loss = F.cosine_similarity(feat_adv.flatten(1), feat_clean_img.flatten(1)).mean()
            reg_T = torch.norm(T_matrix - torch.eye(3, device=self.device).unsqueeze(0), p='fro')
            reg_mu = torch.norm(delta_mu, p=2)
            loss = cfg['lambda_sim'] * sim_loss + cfg['lambda_reg'] * reg_T + cfg['lambda_mu'] * reg_mu
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_([T_matrix, delta_mu], 1.0)
            optimizer.step()
            
            if sim_loss.item() < min_sim_score:
                min_sim_score = sim_loss.item()
                best_adv_img = adv_img_constrained.detach()

        # 4. Final Output
        final_255 = best_adv_img * 255.0
        
        with torch.no_grad():
            self._hook_features = {}
            self.tf_model(
                final_255, batch_data['lidar'], batch_data['ego_waypoint'], batch_data['target_point'], 
                batch_data['target_point_image'], batch_data['speed'].reshape(-1,1),
                dummy_bev, dummy_label, dummy_depth, dummy_semantic
            )
            final_feat = self._hook_features['transformer1'][0]
            drift = F.mse_loss(final_feat, feat_clean_img).item() * 1000
            
            pred_adv, _ = self.tf_model.forward_ego(
                final_255, batch_data['lidar'], batch_data['target_point'], 
                batch_data['target_point_image'], batch_data['speed'].reshape(-1,1),
                bev_points=batch_data.get('bev_points'), cam_points=batch_data.get('cam_points')
            )
            
            lpips_val = 0.0
            if self.lpips_vgg: 
                lpips_val = self.lpips_vgg(best_adv_img*2-1, img_tensor*2-1).mean().item()
            ssim_val = self.ssim_loss(best_adv_img*2-1, img_tensor*2-1).item()

        return final_255, pred_wp_clean, pred_adv, drift, lpips_val, ssim_val

# ================= Visualization =================
def save_visualization_unified(rgb_clean, rgb_adv, pred_clean, pred_adv, gt_wp, target_point, save_path, drift, title_prefix=""):
    img_clean = np.clip(rgb_clean.permute(1,2,0).cpu().numpy()/255.0, 0, 1)
    img_adv = np.clip(rgb_adv.permute(1,2,0).cpu().numpy()/255.0, 0, 1)
    noise = np.clip(np.abs(img_adv - img_clean) * 50.0, 0, 1) # NCF noise is often global tint, x50 to see

    fig = plt.figure(figsize=(16, 14))
    gs = fig.add_gridspec(2, 2, hspace=0.2, wspace=0.15)
    
    ax1 = fig.add_subplot(gs[0, 0]); ax1.imshow(img_clean)
    ax1.set_title(f"{title_prefix} 1. Clean", fontsize=14, fontweight='bold'); ax1.axis('off')
    
    ax2 = fig.add_subplot(gs[0, 1]); ax2.imshow(img_adv)
    ax2.set_title("2. Adv (NCF)", fontsize=14, fontweight='bold'); ax2.axis('off')
    
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
    parser.add_argument('--output_dir', type=str, default='ncf_mfa_style_results')
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
    
    print("=== TransFuser NCF Attack (MFA Style) ===")
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

    attacker = TransFuserNCFAttacker(model, device)
    
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
                    # Run NCF Attack
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
        print("MACRO AVERAGE METRICS (NCF - MFA Style):")
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