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
#os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

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

# Diffusers (仅用于 VAE) & SiT
from diffusers.models import AutoencoderKL

# === 尝试导入 SiT 模型 ===
try:
    from sit import SiT_models
except ImportError:
    print("【错误】找不到 sit.py，请确保 SiT 模型代码在路径中。")
    sys.exit(1)

# TransFuser Modules
try:
    from config import GlobalConfig
    from model import LidarCenterNet
    from data import CARLA_Data
except ImportError:
    print("Please run this script from the TransFuser root directory.")
    sys.exit(1)

# Metrics
try:
    import lpips
    from torch_fidelity import calculate_metrics
except ImportError:
    print("Warning: lpips or torch-fidelity not found.")

# ================= 1. MFA 攻击配置 (TransFuser Context) =================
ATTACK_PARAMS = {
    # === SiT / MFA 参数 ===
    "sit_model": "SiT-XL/2",
    "sit_ckpt": "sit_xl_2_meanflow_ema-002.pt", # 请确保此文件存在
    "t_limit": 0.1 ,           # 对应 openloop_gray_MFA 中的 t_limit
    "sample_steps": 1,          # 采样步数
    "iterations": 50,           # 优化迭代次数
    "lr_z": 0.08,               # Latent 学习率
    "lr_u": 0,               # Velocity Field 学习率
    "epsilon_latent": 0.05,     # 潜空间截断
    "epsilon_u": 0,          # 速度场截断

    # === 损失权重 (MFA) ===
    "w_feature": 3.0,           # 特征损失
    "w_cos": 1.5,               # 余弦相似度
    "w_attn_focus": 4.5,        # 注意力引导权重
    "w_anchor": 6,            # Anchor (Latent Stability) 权重
    "attn_temp": 4.5,           # 注意力温度
    
    # === 分区攻击模式 ===
    "use_patch_attack": True,   # 启用分区攻击 (避免拼接处变形，推荐开启)
    
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
    valid_roots = []
    print(f"Scanning for datasets in: {base_path} ...")
    for root, dirs, files in os.walk(base_path):
        has_routes = False
        for d in dirs:
            route_path = os.path.join(root, d)
            if os.path.isdir(route_path) and os.path.exists(os.path.join(route_path, 'measurements')):
                has_routes = True
                break
        if has_routes: valid_roots.append(root)
    return sorted(list(set(valid_roots)))

# ================= SSIM Metric =================
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

# ================= 2. Monkey Patching (TransFuser Attention) =================
# 保持这个 Hook 用于获取 TransFuser 的注意力图，用于 MFA 的 Attention-Focused Loss
def list_capturing_forward(self, x):
    B, T, C = x.size()
    k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
    q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
    v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

    att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
    att = F.softmax(att, dim=-1)
    
    # [Capture] 保存 Attention Map
    self.stored_att = att 

    att = self.attn_drop(att)
    y = att @ v
    y = y.transpose(1, 2).contiguous().view(B, T, C)
    y = self.resid_drop(self.proj(y))
    return y

# ================= Feature Extractor Hook =================
class MultiLayerFeatureExtractor:
    def __init__(self, cnn_encoder, transformer_layer):
        self.features = {}
        self.hooks = []
        
        cnn_target = None
        if hasattr(cnn_encoder, 's4'): cnn_target = cnn_encoder.s4
        elif hasattr(cnn_encoder, 'layer4'): cnn_target = cnn_encoder.layer4
        elif hasattr(cnn_encoder, 'features'): 
            if len(list(cnn_encoder.features.children())) > 0:
                cnn_target = list(cnn_encoder.features.children())[-1]
        
        if cnn_target:
            self.hooks.append(cnn_target.register_forward_hook(self.save_cnn))
        
        self.hooks.append(transformer_layer.register_forward_hook(self.save_trans))

    def save_cnn(self, m, i, o): self.features['cnn'] = o
    def save_trans(self, m, i, o): self.features['trans'] = o 

    def clear(self): self.features = {}
    def remove(self): 
        for h in self.hooks: h.remove()

# ================= 3. 核心攻击类 (MFA Version) =================
class TransFuserMFAAttacker:
    def __init__(self, transfuser_model, device):
        self.device = device
        self.tf_model = transfuser_model
        
        # --- 1. 设置 TransFuser ---
        if hasattr(self.tf_model, '_model'):
            self.vision_model = self.tf_model._model.image_encoder
            self.transformer1 = self.tf_model._model.transformer1
        else:
            raise AttributeError("TransFuser structure mismatch.")
        
        tf_blocks = self.transformer1.blocks
        for i, block in enumerate(tf_blocks):
            block.attn.forward = types.MethodType(list_capturing_forward, block.attn)
        self.tf_extractor = MultiLayerFeatureExtractor(self.vision_model, self.transformer1)

        # --- 2. 加载 MFA 核心组件 (SiT + VAE) ---
        print(f"Loading SiT Model: {ATTACK_PARAMS['sit_model']}")
        try:
            self.sit_model = SiT_models[ATTACK_PARAMS['sit_model']](
                input_size=32, num_classes=1000, qk_norm=False, finetune=True
            ).to(self.device).eval()
            
            # 加载 Checkpoint
            if not os.path.exists(ATTACK_PARAMS['sit_ckpt']):
                 raise FileNotFoundError(f"Checkpoint not found: {ATTACK_PARAMS['sit_ckpt']}")
                 
            ckpt = torch.load(ATTACK_PARAMS['sit_ckpt'], map_location=self.device)
            state_dict = ckpt["ema"] if "ema" in ckpt else ckpt
            self.sit_model.load_state_dict({k.replace("module.", ""): v for k, v in state_dict.items()})
        except Exception as e:
            print(f"Error loading SiT: {e}")
            sys.exit(1)

        print("Loading VAE...")
        self.vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema").to(self.device).eval()
        self.vae.requires_grad_(False)
        self.sit_model.requires_grad_(False)

        # --- 3. Metrics ---
        try:
            self.lpips_vgg = lpips.LPIPS(net='vgg').to(device).eval()
        except: self.lpips_vgg = None
        self.ssim_loss = SSIM().to(device)

    def _attack_single_patch(self, patch_01, class_labels, num_iters):
        """
        对单个 patch 进行简单 latent 扰动 (不涉及 TransFuser 特征优化)
        patch_01: [B, 3, H, W] 范围 [0, 1]
        返回: 对抗 patch [B, 3, H, W] 范围 [0, 1]
        """
        bs = patch_01.shape[0]
        
        # Resize to 256x256 for VAE
        patch_256 = F.interpolate(patch_01, size=(256, 256), mode='bilinear')
        patch_norm = (patch_256 * 2.0) - 1.0  # [-1, 1]
        
        with torch.no_grad():
            latents_orig = self.vae.encode(patch_norm).latent_dist.sample() * 0.18215
            z_mid = self.invert_latents(latents_orig, class_labels, 
                                       ATTACK_PARAMS['t_limit'], ATTACK_PARAMS['sample_steps'])
        
        # 添加受控扰动
        delta_z = torch.randn_like(z_mid) * ATTACK_PARAMS['epsilon_latent'] * 0.5
        delta_u_seq = {str(i): torch.randn_like(z_mid) * ATTACK_PARAMS['epsilon_u'] * 0.5 
                       for i in range(ATTACK_PARAMS['sample_steps'])}
        
        with torch.no_grad():
            final_z = self.differentiable_flow_sampler(
                z_mid + delta_z, class_labels, ATTACK_PARAMS['t_limit'],
                delta_u_dict=delta_u_seq, num_steps=ATTACK_PARAMS['sample_steps']
            )
            final_norm = self.vae.decode(final_z / 0.18215).sample
            final_01 = (torch.clamp(final_norm, -1, 1) / 2.0 + 0.5)
            final_patch = F.interpolate(final_01, size=patch_01.shape[2:], mode='bilinear')
        
        return final_patch

    def fix_target_point_image(self, tp_img):
        if tp_img is None:
            return torch.zeros(1, 1, 256, 256).to(self.device, dtype=torch.float32)
        if tp_img.dim() == 4:
            return tp_img
        batch_size = tp_img.shape[0] if tp_img.shape[0] > 0 else 1
        return torch.zeros(batch_size, 1, 256, 256).to(self.device, dtype=torch.float32)

    def normalize_tf(self, img_01_tensor):
        mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
        return (img_01_tensor - mean) / std

    # === MFA: SiT Inversion ===
    def invert_latents(self, z, class_labels, t_limit, num_steps):
        batch_size = z.shape[0]
        time_steps = torch.linspace(0.0, t_limit, num_steps + 1, device=self.device)
        with torch.no_grad():
            for i in range(num_steps):
                t_cur, t_next = time_steps[i], time_steps[i + 1]
                u = self.sit_model(z, torch.full((batch_size,), t_cur, device=self.device), 
                                   torch.full((batch_size,), t_next, device=self.device), y=class_labels)
                if u.shape[1] == 8: u, _ = u.chunk(2, dim=1)
                z = z + (t_next - t_cur) * u
        return z

    # === MFA: Differentiable Sampler (带 Delta U) ===
    def differentiable_flow_sampler(self, z, class_labels, t_limit, delta_u_dict=None, num_steps=2):
        batch_size = z.shape[0]
        time_steps = torch.linspace(t_limit, 0.0, num_steps + 1, device=self.device)
        for i in range(num_steps):
            t_cur, t_next = time_steps[i], time_steps[i + 1]
            u = self.sit_model(z, torch.full((batch_size,), t_next, device=self.device), 
                               torch.full((batch_size,), t_cur, device=self.device), y=class_labels)
            if u.shape[1] == 8: u, _ = u.chunk(2, dim=1)
            
            # [MFA Core] Inject Delta U
            if delta_u_dict is not None and str(i) in delta_u_dict:
                u = u + delta_u_dict[str(i)]
            
            z = z - (t_cur - t_next) * u
            if torch.isnan(z).any(): z = torch.nan_to_num(z, nan=0.0)
        return z

    # === MFA Loss Functions (Adapted for TransFuser) ===
    def get_road_focused_loss(self, feat_adv, feat_clean, road_ratio=0.45):
        loss = 0.0
        # TransFuser CNN features: [B, C, H, W]
        if 'cnn' in feat_adv:
            f_adv, f_cln = feat_adv['cnn'], feat_clean['cnn']
            B, C, H, W = f_adv.shape
            weight = torch.ones((1, 1, H, W), device=self.device)
            road_start = int(H * (1.0 - road_ratio))
            weight[:, :, road_start:, :] = 3.0 # MFA increased weight
            
            diff = (f_adv - f_cln) ** 2
            weighted_diff = diff * weight
            loss += weighted_diff.mean() * ATTACK_PARAMS['w_feature']
            loss += F.cosine_similarity(f_adv.view(B, -1), f_cln.view(B, -1)).mean() * ATTACK_PARAMS['w_cos']

        # TransFuser Transformer features: [B, Tokens, Dim]
        if 'trans' in feat_adv:
            f_adv, f_cln = feat_adv['trans'][0], feat_clean['trans'][0]
            loss += F.mse_loss(f_adv, f_cln) * ATTACK_PARAMS['w_feature']
        return loss

    def get_attention_weighted_loss(self, feat_adv, feat_clean, clean_attn_map):
        if 'trans' not in feat_adv or clean_attn_map is None:
            return 0.0
        f_adv, f_cln = feat_adv['trans'][0], feat_clean['trans'][0]
        
        # clean_attn_map from TransFuser: [B, Heads, N, N]
        # Average heads and take attention of relevant tokens
        importance = clean_attn_map.mean(dim=1).mean(dim=1) # [B, N]
        
        # Temperature scaling
        importance = F.softmax(importance * ATTACK_PARAMS['attn_temp'], dim=-1).unsqueeze(-1) # [B, N, 1]
        
        # Match shape [B, N, D]
        b, c, h, w = f_adv.shape # TransFuser hook output is usually [B, C, H, W] or reshaped
        # Note: In TransFuser monkey patch, output is [B, T, C]. Let's check hook.
        # If hook captures transformer output, it might be [B, T, C].
        # In current MultiLayerFeatureExtractor for 'trans', it captures output of transformer layer.
        
        # Correct handling for TransFuser Transformer output structure
        # Assuming f_adv is [B, Tokens, Channels] or [B, Channels, H, W] depending on where it's hooked.
        # TransFuser transformer output is usually [B, T, C].
        
        if f_adv.dim() == 3: # [B, T, C]
            diff = (f_adv - f_cln) ** 2
            w_loss = (diff * importance).mean()
            return w_loss * ATTACK_PARAMS['w_attn_focus']
            
        return 0.0

    def run_diff_attack(self, rgb_raw, batch_data):
        # 1. Prepare Image
        img_01 = rgb_raw.clone().float() / 255.0 # [B, 3, H, W]
        bs = rgb_raw.shape[0]
        img_256 = F.interpolate(img_01, size=(256, 256), mode='bilinear') # VAE input
        img_norm_256 = (img_256 * 2.0) - 1.0 # [-1, 1] for VAE
        
        # Dummy Tensors for TransFuser Forward
        dummy_bev = torch.zeros(bs, 160, 160).to(self.device).long()
        dummy_label = torch.zeros(bs, 1, 7).to(self.device).float()
        dummy_depth = torch.zeros(bs, 160, 704).to(self.device).float()
        dummy_semantic = torch.zeros(bs, 160, 704).to(self.device).long()
        batch_data['target_point_image'] = self.fix_target_point_image(batch_data['target_point_image'])

        # === Step 1: Clean Pass (TransFuser) ===
        self.tf_extractor.clear()
        with torch.no_grad():
            # Original resolution for TransFuser
            self.tf_model(
                rgb_raw, batch_data['lidar'], batch_data['ego_waypoint'], batch_data['target_point'], 
                batch_data['target_point_image'], batch_data['speed'].reshape(-1,1),
                dummy_bev, dummy_label, dummy_depth, dummy_semantic
            )
            feat_clean = {k: v.detach().clone() if isinstance(v, torch.Tensor) else [x.detach().clone() for x in v] 
                          for k, v in self.tf_extractor.features.items()}
            
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

        # === Step 2: VAE Encoding & SiT Inversion ===
        # Use ImageNet class 817 (sports car) as condition, similar to MFA
        class_labels = torch.tensor([817] * bs, device=self.device)
        
        with torch.no_grad():
            latents_orig = self.vae.encode(img_norm_256).latent_dist.sample() * 0.18215
            
            # Invert to get z_mid
            z_mid = self.invert_latents(latents_orig, class_labels, 
                                      ATTACK_PARAMS['t_limit'], ATTACK_PARAMS['sample_steps'])
            real_z_std = latents_orig.std()

        # === Step 3: MFA Optimization Loop ===
        # Optimize Delta Z (Latent noise) and Delta U (Velocity Field)
        delta_z = nn.Parameter(torch.randn_like(z_mid) * 1e-4)
        delta_u_seq = nn.ParameterDict({
            str(i): nn.Parameter(torch.zeros_like(z_mid)) for i in range(ATTACK_PARAMS['sample_steps'])
        })
        
        optimizer = optim.Adam([
            {'params': [delta_z], 'lr': ATTACK_PARAMS['lr_z']},
            {'params': delta_u_seq.parameters(), 'lr': ATTACK_PARAMS['lr_u']}
        ])

        for _ in range(ATTACK_PARAMS['iterations']):
            optimizer.zero_grad()
            
            # A. Forward Flow
            z_start = z_mid + delta_z
            z_final = self.differentiable_flow_sampler(
                z_start, class_labels, ATTACK_PARAMS['t_limit'], 
                delta_u_dict=delta_u_seq, num_steps=ATTACK_PARAMS['sample_steps']
            )
            
            if torch.isnan(z_final).any(): break

            # B. Decode to Image
            img_adv_norm = self.vae.decode(z_final / 0.18215).sample
            img_adv_norm = torch.clamp(img_adv_norm, -1, 1) # [-1, 1]
            
            # Convert to TransFuser input format [0, 255] and Resize
            img_adv_01 = (img_adv_norm / 2.0) + 0.5
            # Resize back to original TransFuser input size (usually defined in data loader, e.g. input raw size)
            # Here rgb_raw is passed, assuming it matches model expectation. 
            # We resize adv image to match rgb_raw spatial dims.
            img_adv_resized = F.interpolate(img_adv_01, size=(rgb_raw.shape[2], rgb_raw.shape[3]), mode='bilinear')
            adv_rgb_255 = img_adv_resized * 255.0

            # C. TransFuser Forward (Features)
            self.tf_extractor.clear()
            self.tf_model(
                adv_rgb_255, batch_data['lidar'], batch_data['ego_waypoint'], batch_data['target_point'], 
                batch_data['target_point_image'], batch_data['speed'].reshape(-1,1),
                dummy_bev, dummy_label, dummy_depth, dummy_semantic
            )
            feat_adv = self.tf_extractor.features
            
            # D. Calculate Losses
            loss_road = self.get_road_focused_loss(feat_adv, feat_clean)
            loss_attn = self.get_attention_weighted_loss(feat_adv, feat_clean, clean_tf_attn)
            loss_anchor = torch.abs(z_final.std() - real_z_std)
            
            total_loss = (- loss_road - loss_attn 
                         + ATTACK_PARAMS['w_anchor'] * loss_anchor)
            
            total_loss.backward()
            optimizer.step()
            
            # Constraints
            with torch.no_grad():
                delta_z.data.clamp_(-ATTACK_PARAMS['epsilon_latent'], ATTACK_PARAMS['epsilon_latent'])
                for k in delta_u_seq:
                    delta_u_seq[k].data.clamp_(-ATTACK_PARAMS['epsilon_u'], ATTACK_PARAMS['epsilon_u'])

        # === Step 4: Final Generation ===
        with torch.no_grad():
            final_z = self.differentiable_flow_sampler(
                z_mid + delta_z, class_labels, ATTACK_PARAMS['t_limit'], 
                delta_u_dict=delta_u_seq, num_steps=ATTACK_PARAMS['sample_steps']
            )
            final_norm = self.vae.decode(final_z / 0.18215).sample
            final_01 = (torch.clamp(final_norm, -1, 1) / 2.0 + 0.5)
            final_resized = F.interpolate(final_01, size=(rgb_raw.shape[2], rgb_raw.shape[3]), mode='bilinear')
            img_01_orig = rgb_raw.clone().float() / 255.0
            
            final_255 = final_resized * 255.0
            
            # === Metrics & Eval ===
            self.tf_extractor.clear()
            self.tf_model(
                final_255, batch_data['lidar'], batch_data['ego_waypoint'], batch_data['target_point'], 
                batch_data['target_point_image'], batch_data['speed'].reshape(-1,1),
                dummy_bev, dummy_label, dummy_depth, dummy_semantic
            )
            final_feat = self.tf_extractor.features.get('cnn', torch.tensor(0))
            clean_feat_c = feat_clean.get('cnn', torch.tensor(0))
            drift = F.mse_loss(final_feat, clean_feat_c).item() * 1000
            
            pred_adv, _ = self.tf_model.forward_ego(
                final_255, batch_data['lidar'], batch_data['target_point'], 
                batch_data['target_point_image'], batch_data['speed'].reshape(-1,1),
                bev_points=batch_data.get('bev_points'), cam_points=batch_data.get('cam_points')
            )
            
            lpips_val = 0.0
            if self.lpips_vgg: 
                # LPIPS expects [-1, 1]
                lpips_val = self.lpips_vgg(final_resized*2-1, img_01*2-1).mean().item()
            ssim_val = self.ssim_loss(final_resized*2-1, img_01*2-1).item()

        return final_255, pred_wp_clean, pred_adv, drift, lpips_val, ssim_val

    def run_patch_attack(self, rgb_raw, batch_data):
        """
        [修改版] 分区攻击方法：串行处理每个 Patch，总迭代次数分摊。
        时间消耗将大幅降低，与 DA 方法逻辑对齐。
        """
        bs = rgb_raw.shape[0]
        H, W = rgb_raw.shape[2], rgb_raw.shape[3]  # 160, 704
        
        img_01 = rgb_raw.clone().float() / 255.0  # [B, 3, H, W]
        
        # 三个摄像头区域的边界
        cam_width = W // 3
        boundaries = [0, cam_width, cam_width * 2, W]
        
        # Dummy Tensors
        dummy_bev = torch.zeros(bs, 160, 160).to(self.device).long()
        dummy_label = torch.zeros(bs, 1, 7).to(self.device).float()
        dummy_depth = torch.zeros(bs, 160, 704).to(self.device).float()
        dummy_semantic = torch.zeros(bs, 160, 704).to(self.device).long()
        batch_data['target_point_image'] = self.fix_target_point_image(batch_data['target_point_image'])
        
        # === Step 1: Clean Pass (获取干净特征) ===
        self.tf_extractor.clear()
        with torch.no_grad():
            self.tf_model(
                rgb_raw, batch_data['lidar'], batch_data['ego_waypoint'], batch_data['target_point'], 
                batch_data['target_point_image'], batch_data['speed'].reshape(-1,1),
                dummy_bev, dummy_label, dummy_depth, dummy_semantic
            )
            feat_clean = {k: v.detach().clone() if isinstance(v, torch.Tensor) else [x.detach().clone() for x in v] 
                          for k, v in self.tf_extractor.features.items()}
            
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
        
        # 准备存储最终结果的容器
        final_patches = [None] * 3
        class_labels = torch.tensor([817] * bs, device=self.device)
        
        # === Step 2: 串行处理每个 Patch ===
        # 计算每个 patch 分配到的迭代次数 (例如 60 // 3 = 20)
        patch_iters = ATTACK_PARAMS['iterations'] // 3
        
        for i in range(3):
            # 2.1 截取当前 Patch 并编码
            x_start, x_end = boundaries[i], boundaries[i + 1]
            patch = img_01[:, :, :, x_start:x_end].clone()
            patch_size = patch.shape[2:]
            
            # VAE Encoding & Inversion
            with torch.no_grad():
                patch_256 = F.interpolate(patch, size=(256, 256), mode='bilinear')
                patch_norm = (patch_256 * 2.0) - 1.0
                latents_orig = self.vae.encode(patch_norm).latent_dist.sample() * 0.18215
                z_mid = self.invert_latents(latents_orig, class_labels,
                                           ATTACK_PARAMS['t_limit'], ATTACK_PARAMS['sample_steps'])
                real_z_std = latents_orig.std()

            # 2.2 初始化当前 Patch 的优化参数
            delta_z = nn.Parameter(torch.randn_like(z_mid) * 1e-4)
            delta_u_seq = nn.ParameterDict({
                str(k): nn.Parameter(torch.zeros_like(z_mid)) 
                for k in range(ATTACK_PARAMS['sample_steps'])
            })
            
            optimizer = optim.Adam([
                {'params': [delta_z], 'lr': ATTACK_PARAMS['lr_z']},
                {'params': delta_u_seq.parameters(), 'lr': ATTACK_PARAMS['lr_u']}
            ])
            
            # 2.3 优化循环 (次数减少)
            for _ in range(patch_iters):
                optimizer.zero_grad()
                
                # A. 生成当前对抗 Patch
                z_start = z_mid + delta_z
                z_final = self.differentiable_flow_sampler(
                    z_start, class_labels, ATTACK_PARAMS['t_limit'],
                    delta_u_dict=delta_u_seq, num_steps=ATTACK_PARAMS['sample_steps']
                )
                
                if torch.isnan(z_final).any(): break
                
                # 解码
                img_adv_norm = self.vae.decode(z_final / 0.18215).sample
                img_adv_01 = (torch.clamp(img_adv_norm, -1, 1) / 2.0 + 0.5)
                adv_patch = F.interpolate(img_adv_01, size=patch_size, mode='bilinear')
                
                # B. 拼接到全图 (其他部分保持 Clean，模仿 DA 逻辑)
                # 注意：这里我们每次都用 img_01 (Clean) 作为底板
                img_adv_full = img_01.clone()
                img_adv_full[:, :, :, x_start:x_end] = adv_patch
                adv_rgb_255 = img_adv_full * 255.0
                
                # C. TransFuser Forward
                self.tf_extractor.clear()
                self.tf_model(
                    adv_rgb_255, batch_data['lidar'], batch_data['ego_waypoint'], batch_data['target_point'],
                    batch_data['target_point_image'], batch_data['speed'].reshape(-1,1),
                    dummy_bev, dummy_label, dummy_depth, dummy_semantic
                )
                feat_adv = self.tf_extractor.features
                
                # D. Loss 计算
                loss_road = self.get_road_focused_loss(feat_adv, feat_clean)
                loss_attn = self.get_attention_weighted_loss(feat_adv, feat_clean, clean_tf_attn)
                loss_anchor = torch.abs(z_final.std() - real_z_std)
                
                total_loss = (- loss_road - loss_attn + ATTACK_PARAMS['w_anchor'] * loss_anchor)
                
                total_loss.backward()
                optimizer.step()
                
                # 约束
                with torch.no_grad():
                    delta_z.data.clamp_(-ATTACK_PARAMS['epsilon_latent'], ATTACK_PARAMS['epsilon_latent'])
                    for k in delta_u_seq:
                        delta_u_seq[k].data.clamp_(-ATTACK_PARAMS['epsilon_u'], ATTACK_PARAMS['epsilon_u'])

            # 2.4 保存当前优化好的 Patch (最后一次生成的结果)
            with torch.no_grad():
                final_z = self.differentiable_flow_sampler(
                    z_mid + delta_z, class_labels, ATTACK_PARAMS['t_limit'],
                    delta_u_dict=delta_u_seq, num_steps=ATTACK_PARAMS['sample_steps']
                )
                final_norm = self.vae.decode(final_z / 0.18215).sample
                final_01_p = (torch.clamp(final_norm, -1, 1) / 2.0 + 0.5)
                final_patch_resized = F.interpolate(final_01_p, size=patch_size, mode='bilinear')
                final_patches[i] = final_patch_resized

            # 清理显存，准备下一个 Patch
            del optimizer, delta_z, delta_u_seq, z_mid
            torch.cuda.empty_cache()

        # === Step 3: 最终拼接与评估 ===
        with torch.no_grad():
            # 将三个分别优化好的 Patch 拼在一起
            final_01 = torch.cat(final_patches, dim=3)
            final_255 = final_01 * 255.0
            
            # Metrics
            self.tf_extractor.clear()
            self.tf_model(
                final_255, batch_data['lidar'], batch_data['ego_waypoint'], batch_data['target_point'], 
                batch_data['target_point_image'], batch_data['speed'].reshape(-1,1),
                dummy_bev, dummy_label, dummy_depth, dummy_semantic
            )
            final_feat = self.tf_extractor.features.get('cnn', torch.tensor(0))
            clean_feat_c = feat_clean.get('cnn', torch.tensor(0))
            drift = F.mse_loss(final_feat, clean_feat_c).item() * 1000
            
            pred_adv, _ = self.tf_model.forward_ego(
                final_255, batch_data['lidar'], batch_data['target_point'], 
                batch_data['target_point_image'], batch_data['speed'].reshape(-1,1),
                bev_points=batch_data.get('bev_points'), cam_points=batch_data.get('cam_points')
            )
            
            lpips_val = 0.0
            if self.lpips_vgg:
                lpips_val = self.lpips_vgg(final_01*2-1, img_01*2-1).mean().item()
            ssim_val = self.ssim_loss(final_01*2-1, img_01*2-1).item()
        
        return final_255, pred_wp_clean, pred_adv, drift, lpips_val, ssim_val



# ================= Visualization =================
def save_visualization_unified(rgb_clean, rgb_adv, pred_clean, pred_adv, gt_wp, target_point, save_path, drift):
    img_clean = np.clip(rgb_clean.permute(1,2,0).cpu().numpy()/255.0, 0, 1)
    img_adv = np.clip(rgb_adv.permute(1,2,0).cpu().numpy()/255.0, 0, 1)
    noise = np.clip(np.abs(img_adv - img_clean) * 15.0, 0, 1)

    fig = plt.figure(figsize=(16, 14))
    gs = fig.add_gridspec(2, 2, hspace=0.2, wspace=0.15)
    
    ax1 = fig.add_subplot(gs[0, 0]); ax1.imshow(img_clean)
    ax1.set_title("1. Clean", fontsize=14, fontweight='bold'); ax1.axis('off')
    
    ax2 = fig.add_subplot(gs[0, 1]); ax2.imshow(img_adv)
    ax2.set_title("2. Adv (MFA-SiT)", fontsize=14, fontweight='bold'); ax2.axis('off')
    
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
    parser.add_argument('--output_dir', type=str, default='mfa_da_results')
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    sys.stdout = Logger(os.path.join(args.output_dir, "attack_log.txt"))
    fid_c = os.path.join(args.output_dir, "fid_clean")
    fid_a = os.path.join(args.output_dir, "fid_adv")
    if os.path.exists(fid_c): shutil.rmtree(fid_c)
    if os.path.exists(fid_a): shutil.rmtree(fid_a)
    os.makedirs(fid_c, exist_ok=True); os.makedirs(fid_a, exist_ok=True)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    ATTACK_PARAMS['device'] = f"cuda:{args.gpu}"
    
    print("=== TransFuser MFA Attack (SiT Version) ===")
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

    # === 初始化 MFA Attacker ===
    attacker = TransFuserMFAAttacker(model, device)
    
    datasets = find_valid_datasets(args.root_dir)
    
    global_stats = {
        'shift': [], 'lat_err': [], 'lon_err': [], 'sim_drift': [], 
        'lpips': [], 'ssim': [], 'clean_len': [], 'success': [], 
        'global_tan': [], 'avg_time': []
    }
    
    start_time = time.time()
    total_processed = 0

    for ds_idx, ds_path in enumerate(datasets):
        ds_name = os.path.basename(ds_path)
        print(f"\n[{ds_idx+1}/{len(datasets)}] Processing: {ds_name}")
        ds_out = os.path.join(args.output_dir, ds_name)
        os.makedirs(ds_out, exist_ok=True)
        
        dataset = CARLA_Data(root=[ds_path], config=config, shared_dict=None)
        
        # 预先创建 Dummy Tensors (Batch Size 固定为 1)
        bs = 1
        dummy_label = torch.zeros(bs, 1, 7).to(device).float()
        dummy_depth = torch.zeros(bs, 160, 704).to(device).float()
        dummy_semantic = torch.zeros(bs, 160, 704).to(device).long()

        loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)
        ds_totals = {'shift': 0.0, 'lat_err': 0.0, 'lon_err': 0.0, 'drift': 0.0, 
                     'lpips': 0.0, 'ssim': 0.0, 'clean_len': 0.0, 'success': 0, 'time': 0.0}
        count = 0
        
        for batch_idx, data in enumerate(tqdm(loader, desc=ds_name)):
            t0 = time.time()
            
            lidar = data['lidar'].to(device, dtype=torch.float32)
            target_point = data['target_point'].to(device, dtype=torch.float32)
            tp_img = data['target_point_image'].to(device, dtype=torch.float32)
            speed = data['speed'].to(device, dtype=torch.float32)
            
            if tp_img.dim() != 4:
                current_bs = tp_img.shape[0] if tp_img.shape[0] > 0 else 1
                tp_img = torch.zeros(current_bs, 1, 256, 256).to(device, dtype=torch.float32)

            rgb = data['rgb'].to(device, dtype=torch.float32)
            gt_wp = data['ego_waypoint'].to(device, dtype=torch.float32)

            batch = {
                'lidar': lidar,
                'ego_waypoint': gt_wp,
                'target_point': target_point,
                'target_point_image': tp_img,
                'speed': speed,
                # 为了兼容性，这里也放进去，但在 run_diff_attack 中会使用 dummy tensors
                'bev_label': dummy_label, 
                'depth': dummy_depth, 
                'semantic': dummy_semantic,
                'bev_points': data.get('bev_points').long().to(device) if 'bev_points' in data else None,
                'cam_points': data.get('cam_points').long().to(device) if 'cam_points' in data else None
            }

            try:
                # 根据配置选择攻击方法
                if ATTACK_PARAMS.get('use_patch_attack', False):
                    # 分区攻击：分别处理三个摄像头区域，避免拼接处变形
                    rgb_adv, p_clean, p_adv, drift, lp, ss = attacker.run_patch_attack(rgb, batch)
                else:
                    # 标准攻击：整体处理
                    rgb_adv, p_clean, p_adv, drift, lp, ss = attacker.run_diff_attack(rgb, batch)
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
                
                Image.fromarray(rgb[0].permute(1,2,0).cpu().numpy().astype(np.uint8)).save(os.path.join(fid_c, f"{ds_name}_{batch_idx}.jpg"))
                Image.fromarray(rgb_adv[0].permute(1,2,0).cpu().numpy().astype(np.uint8)).save(os.path.join(fid_a, f"{ds_name}_{batch_idx}.jpg"))

                if batch_idx % 20 == 0:
                    save_visualization_unified(rgb[0], rgb_adv[0], clean_wp, adv_wp, gt_wp[0].cpu().numpy(), 
                                           batch['target_point'][0].cpu().numpy(), 
                                           os.path.join(ds_out, f"f_{batch_idx}.png"), drift)
            except Exception as e:
                print(f"Err {batch_idx}: {e}")
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

    # Final Macro Report
    if len(global_stats['shift']) > 0:
        elapsed = time.time() - start_time
        def get_avg(k): return sum(global_stats[k]) / len(global_stats[k])
        
        print("\n" + "="*60)
        print("MACRO AVERAGE METRICS (MFA-SiT TransFuser):")
        print(f"  > Total Samples:        {total_processed}")
        print(f"  > Total Time:           {timedelta(seconds=int(elapsed))}")
        print(f"  > Avg Time Per Image:   {get_avg('avg_time'):.3f} s")
        print(f"  > Avg Route Shift:      {get_avg('shift'):.4f} m")
        print(f"  > Avg Lateral Error:    {get_avg('lat_err'):.4f} m")
        print(f"  > Avg Longitudinal Err: {get_avg('lon_err'):.4f} m")
        print(f"  > Avg Success Rate:     {get_avg('success'):.2f} %")
        print(f"  > Avg Global Tan:       {get_avg('global_tan'):.4f}")
        print(f"  > Avg Feature Drift:    {get_avg('sim_drift'):.4f}")
        print(f"  > Avg LPIPS:            {get_avg('lpips'):.4f}")
        print(f"  > Avg SSIM:             {get_avg('ssim'):.4f}")
        
        print("-" * 40)
        try:
            del attacker, model; gc.collect(); torch.cuda.empty_cache()
            metrics = calculate_metrics(input1=fid_c, input2=fid_a, cuda=True, isc=False, fid=True, verbose=False)
            print(f"  > Global FID:           {metrics['frechet_inception_distance']:.4f}")
        except: 
            print("  > FID Calculation Failed")
        print("="*60)

if __name__ == "__main__":
    main()