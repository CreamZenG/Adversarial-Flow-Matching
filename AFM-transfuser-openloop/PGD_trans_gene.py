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

# ================= 1. PGD 攻击配置 (适配 MFA 结构) =================
ATTACK_PARAMS = {
    "epsilon": 8/255,       # 扰动强度 (L_inf 约束)
    "alpha": 2/255,         # 单步步长
    "steps": 40,            # 迭代次数 (设为1即为 FGSM)
    
    # === Loss 权重 (最大化特征距离) ===
    "w_feature": 10.0,      # MSE 权重
    "w_cosine": 5.0,        # 余弦相似度权重
    
    "device": "cuda:0",
}

# ================= 工具函数 =================
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

# ================= 2. PGD 攻击者类 =================
class TransFuserPGDAttacker:
    def __init__(self, transfuser_model, device):
        self.device = device
        self.tf_model = transfuser_model
        
        # 自动检测 FGSM
        if ATTACK_PARAMS['steps'] == 1:
            print(f">>> Detected FGSM (steps=1). Forcing alpha = epsilon ({ATTACK_PARAMS['epsilon']:.4f})")
            ATTACK_PARAMS['alpha'] = ATTACK_PARAMS['epsilon']

        # 锁定 Vision Backbone (RegNet/ResNet) 用于提取特征
        if hasattr(self.tf_model, '_model') and hasattr(self.tf_model._model, 'image_encoder'):
            self.vision_model = self.tf_model._model.image_encoder
        elif hasattr(self.tf_model, 'image_encoder'):
            self.vision_model = self.tf_model.image_encoder
        else:
            raise AttributeError("Error: Could not find 'image_encoder' in TransFuser.")
        
        print(f"Target Vision Model Locked: {self.vision_model.__class__.__name__}")

        try:
            self.lpips_vgg = lpips.LPIPS(net='vgg').to(device).eval()
        except: self.lpips_vgg = None
        self.ssim_loss = SSIM().to(device)

    def fix_target_point_image(self, tp_img):
        if tp_img is None:
            return torch.zeros(1, 1, 256, 256).to(self.device, dtype=torch.float32)
        if tp_img.dim() == 4: return tp_img
        batch_size = tp_img.shape[0] if tp_img.shape[0] > 0 else 1
        return torch.zeros(batch_size, 1, 256, 256).to(self.device, dtype=torch.float32)

    def normalize_imagenet(self, x):
        # TransFuser backbone expects ImageNet normalization
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
        return (x - mean) / std

    def get_features(self, img_tensor_norm):
        # Extract features from the CNN backbone directly
        timm_model = self.vision_model.features
        if hasattr(timm_model, 'forward_features'):
            features = timm_model.forward_features(img_tensor_norm)
        else:
            features = timm_model(img_tensor_norm)
        
        # Handle tuple output (common in timm)
        if isinstance(features, (tuple, list)): return features[-1]
        return features

    def run_attack(self, rgb_raw, batch_data):
        """
        PGD Attack
        rgb_raw: [B, 3, H, W], range [0, 255]
        """
        cfg = ATTACK_PARAMS
        bs = rgb_raw.shape[0]
        
        # 补全数据
        dummy_bev = torch.zeros(bs, 160, 160).to(self.device).long()
        dummy_label = torch.zeros(bs, 1, 7).to(self.device).float()
        dummy_depth = torch.zeros(bs, 160, 704).to(self.device).float()
        dummy_semantic = torch.zeros(bs, 160, 704).to(self.device).long()
        batch_data['target_point_image'] = self.fix_target_point_image(batch_data['target_point_image'])

        # Pre-process
        img_01 = rgb_raw.clone().float() / 255.0
        img_norm_clean = self.normalize_imagenet(img_01)
        
        # 1. Clean Pass (Target Features)
        with torch.no_grad():
            feat_clean = self.get_features(img_norm_clean).detach()
            
            pred_wp_clean, _ = self.tf_model.forward_ego(
                rgb_raw, batch_data['lidar'], batch_data['target_point'], 
                batch_data['target_point_image'], batch_data['speed'].reshape(-1,1),
                bev_points=batch_data.get('bev_points'), cam_points=batch_data.get('cam_points')
            )

        # 2. PGD Optimization
        delta = torch.zeros_like(img_01).uniform_(-1e-6, 1e-6)
        delta.requires_grad = True
        
        for _ in range(cfg['steps']):
            # Adv Image
            x_adv = torch.clamp(img_01 + delta, 0, 1)
            x_adv_norm = self.normalize_imagenet(x_adv)
            
            # Extract Features
            feat_adv = self.get_features(x_adv_norm)
            
            # Loss: Maximize Distance (Untargeted Attack)
            loss_mse = F.mse_loss(feat_adv, feat_clean)
            
            # Cosine Loss
            feat_adv_flat = feat_adv.view(feat_adv.size(0), -1)
            feat_clean_flat = feat_clean.view(feat_clean.size(0), -1)
            loss_cos = F.cosine_similarity(feat_adv_flat, feat_clean_flat, dim=-1).mean()
            
            # Total Objective: Maximize (MSE - Cosine) -> Minimize -(MSE - Cosine)
            # PGD assumes we are MINIMIZING loss, but here we want to MAXIMIZE drift.
            # Usually PGD maximizes Loss(x, y). 
            # Here "Loss" is feature similarity. We want to MINIMIZE similarity.
            # Code from original: w_feature * loss_mse - w_cosine * loss_cos
            # If we maximize this, MSE grows, Cosine (similarity) shrinks. Correct.
            
            objective = cfg['w_feature'] * loss_mse - cfg['w_cosine'] * loss_cos
            
            if delta.grad is not None: delta.grad.zero_()
            objective.backward()
            
            grad = delta.grad.detach()
            
            # PGD Update (Maximize Objective)
            delta.data = delta.data + cfg['alpha'] * grad.sign()
            delta.data = torch.clamp(delta.data, -cfg['epsilon'], cfg['epsilon'])
            delta.data = torch.clamp(img_01 + delta.data, 0, 1) - img_01

        # 3. Final Outputs
        final_img_01 = torch.clamp(img_01 + delta, 0, 1).detach()
        final_255 = final_img_01 * 255.0
        
        with torch.no_grad():
            # Final Feature Drift
            feat_adv_final = self.get_features(self.normalize_imagenet(final_img_01))
            drift = F.mse_loss(feat_adv_final, feat_clean).item() * 1000
            
            pred_adv, _ = self.tf_model.forward_ego(
                final_255, batch_data['lidar'], batch_data['target_point'], 
                batch_data['target_point_image'], batch_data['speed'].reshape(-1,1),
                bev_points=batch_data.get('bev_points'), cam_points=batch_data.get('cam_points')
            )
            
            lpips_val = 0.0
            if self.lpips_vgg: 
                lpips_val = self.lpips_vgg(final_img_01*2-1, img_01*2-1).mean().item()
            ssim_val = self.ssim_loss(final_img_01*2-1, img_01*2-1).item()

        return final_255, pred_wp_clean, pred_adv, drift, lpips_val, ssim_val

# ================= Visualization =================
def save_visualization_unified(rgb_clean, rgb_adv, pred_clean, pred_adv, gt_wp, target_point, save_path, drift, title_prefix=""):
    img_clean = np.clip(rgb_clean.permute(1,2,0).cpu().numpy()/255.0, 0, 1)
    img_adv = np.clip(rgb_adv.permute(1,2,0).cpu().numpy()/255.0, 0, 1)
    noise = np.clip(np.abs(img_adv - img_clean) * 15.0, 0, 1) # PGD noise is small, amplify x15

    fig = plt.figure(figsize=(16, 14))
    gs = fig.add_gridspec(2, 2, hspace=0.2, wspace=0.15)
    
    ax1 = fig.add_subplot(gs[0, 0]); ax1.imshow(img_clean)
    ax1.set_title(f"{title_prefix} 1. Clean", fontsize=14, fontweight='bold'); ax1.axis('off')
    
    ax2 = fig.add_subplot(gs[0, 1]); ax2.imshow(img_adv)
    ax2.set_title("2. Adv (PGD)", fontsize=14, fontweight='bold'); ax2.axis('off')
    
    ax3 = fig.add_subplot(gs[1, 0]); ax3.imshow(noise)
    ax3.set_title("3. Noise (x15)", fontsize=14, fontweight='bold'); ax3.axis('off')
    
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
    parser.add_argument('--output_dir', type=str, default='pgd_mfa_style_results')
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
    
    print("=== TransFuser PGD Attack (MFA Style) ===")
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

    attacker = TransFuserPGDAttacker(model, device)
    
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
                    # Run PGD Attack
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
        print("MACRO AVERAGE METRICS (PGD - MFA Style):")
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