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
import numpy as np
from PIL import Image
from tqdm import tqdm
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

# Diffusers & Transformers
from diffusers import StableDiffusionPipeline, DDIMScheduler
from diffusers.models.attention import CrossAttention as Attention 
from transformers import CLIPTextModel, CLIPTokenizer

# TransFuser Modules
try:
    from config import GlobalConfig
    from model import LidarCenterNet
    from data1 import CARLA_Data  # 使用修复后的 data1 模块
except ImportError:
    print("Please run this script from the TransFuser root directory.")
    sys.exit(1)

# Metrics
try:
    import lpips
    from torch_fidelity import calculate_metrics
except ImportError:
    print("Warning: lpips or torch-fidelity not found.")

# ================= 1. 攻击配置 (SimLingo Aligned - DA Core) =================
ATTACK_PARAMS = {
    "steps": 20,            # SD 采样总步数
    "start_step": 1,        # 攻击生成过程的末端
    "iterations": 60,       # 优化迭代次数
    "lr": 0.05,             # 学习率
    "epsilon_latent": 0.05, # 潜空间截断范围
    "sd_model": "Manojb/stable-diffusion-2-base", 
    "device": "cuda:0",
    
    # === 结构约束权重 ===
    "w_sd_struct": 450.0,   # 保持原图结构的权重
    
    # === 特征攻击权重 (MFA) ===
    "w_feature": 3.0,       # 基础特征破坏权重
    "w_cos": 1.5,           # 余弦相似度攻击权重
    "w_attn_focus": 4.5,    # TransFuser 注意力引导攻击权重
    "attn_temp": 4.5,       # 注意力温度
    
    # === 分区攻击模式 ===
    "use_patch_attack": True,   # 启用分区攻击 (避免拼接处变形，推荐开启)
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
    """
    [移植自 MFA] 扫描逻辑修改：
    只将直接包含 'measurements' 文件夹的目录视为数据集（Route）。
    """
    valid_routes = []
    print(f"Scanning for datasets in: {base_path} ...")
    
    for root, dirs, files in os.walk(base_path):
        if 'measurements' in dirs:
            valid_routes.append(root)
            
    if not valid_routes:
        if os.path.exists(os.path.join(base_path, 'measurements')):
            valid_routes.append(base_path)
            
    return sorted(list(set(valid_routes)))

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

# ================= DiffAttack Controllers =================
class DiffAttackController:
    def __init__(self):
        self.loss = 0
        self.reference_maps = {}
        self.mode = "store" 
    
    def reset(self):
        self.loss = 0
        
    def step(self):
        pass
        
    def __call__(self, attn_probs, is_cross, layer_name):
        if is_cross: return attn_probs
        key = layer_name
        
        if self.mode == "store":
            self.reference_maps[key] = attn_probs.detach().clone()
        elif self.mode == "loss":
            if key in self.reference_maps:
                ref = self.reference_maps[key]
                if attn_probs.shape == ref.shape:
                    self.loss += F.mse_loss(attn_probs, ref)
        elif self.mode == "replace":
            if key in self.reference_maps:
                return self.reference_maps[key]
        return attn_probs
    
class P2PAttentionProcessor:
    def __init__(self, name, is_cross, controller):
        self.name = name  
        self.is_cross = is_cross
        self.controller = controller

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None, scale=1.0):
        query = attn.to_q(hidden_states)
        if self.is_cross: 
            key = attn.to_k(encoder_hidden_states)
            value = attn.to_v(encoder_hidden_states)
        else: 
            key = attn.to_k(hidden_states)
            value = attn.to_v(hidden_states)
            
        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)
        
        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        attention_probs = self.controller(attention_probs, self.is_cross, self.name)
        
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states
    
def register_attention_control(unet, controller):
    for name, module in unet.named_modules():
        if isinstance(module, Attention):
            is_cross = True if "attn2" in name else False 
            module.set_processor(P2PAttentionProcessor(name, is_cross, controller))

# ================= 3. 核心攻击类 =================
class TransFuserDiffAttacker:
    def __init__(self, transfuser_model, device):
        self.device = device
        self.tf_model = transfuser_model
        
        if hasattr(self.tf_model, '_model'):
            self.vision_model = self.tf_model._model.image_encoder
            self.transformer1 = self.tf_model._model.transformer1
        else:
            raise AttributeError("TransFuser structure mismatch.")
        
        tf_blocks = self.transformer1.blocks
        for i, block in enumerate(tf_blocks):
            block.attn.forward = types.MethodType(list_capturing_forward, block.attn)

        print(f"Loading SD: {ATTACK_PARAMS['sd_model']}")
        self.sd_pipe = StableDiffusionPipeline.from_pretrained(
            ATTACK_PARAMS['sd_model'], torch_dtype=torch.float32
        ).to(self.device)
        self.sd_pipe.scheduler = DDIMScheduler.from_config(self.sd_pipe.scheduler.config)
        self.sd_pipe.scheduler.set_timesteps(ATTACK_PARAMS['steps'])
        self.sd_pipe.vae.requires_grad_(False)
        self.sd_pipe.unet.requires_grad_(False)
        
        try:
            self.sd_pipe.enable_xformers_memory_efficient_attention()
            print("Enabled xformers memory efficient attention.")
        except:
            print("Xformers not available, falling back to slicing.")
            self.sd_pipe.enable_attention_slicing(slice_size="auto")

        try:
            self.lpips_vgg = lpips.LPIPS(net='vgg').to(device).eval()
        except: self.lpips_vgg = None
        self.ssim_loss = SSIM().to(device)

        self.tf_extractor = MultiLayerFeatureExtractor(self.vision_model, self.transformer1)
        with torch.no_grad():
            uncond_input = self.sd_pipe.tokenizer(
                [""], padding="max_length", max_length=self.sd_pipe.tokenizer.model_max_length, return_tensors="pt"
            )
            self.null_embs = self.sd_pipe.text_encoder(uncond_input.input_ids.to(device))[0].detach()

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

    def ddim_inversion(self, latents, start_step_idx):
        timesteps = self.sd_pipe.scheduler.timesteps
        start_index = len(timesteps) - 1
        stop_index = len(timesteps) - 1 - start_step_idx
        curr_latents = latents.clone()
        with torch.no_grad():
            for i in range(start_index, stop_index, -1):
                t = timesteps[i]
                t_next = timesteps[i-1] if i > 0 else None
                if t_next is None: break
                noise_pred = self.sd_pipe.unet(curr_latents, t, encoder_hidden_states=self.null_embs).sample
                alpha_prod_t = self.sd_pipe.scheduler.alphas_cumprod[t]
                alpha_prod_t_next = self.sd_pipe.scheduler.alphas_cumprod[t_next]
                beta_prod_t = 1 - alpha_prod_t
                beta_prod_t_next = 1 - alpha_prod_t_next
                pred_original = (curr_latents - beta_prod_t**0.5 * noise_pred) / alpha_prod_t**0.5
                curr_latents = alpha_prod_t_next**0.5 * pred_original + beta_prod_t_next**0.5 * noise_pred
        return curr_latents

    def get_road_focused_loss(self, feat_adv, feat_clean, road_ratio=0.45):
        loss = 0.0
        if 'cnn' in feat_adv:
            f_adv, f_cln = feat_adv['cnn'], feat_clean['cnn']
            B, C, H, W = f_adv.shape
            weight = torch.ones((1, 1, H, W), device=self.device)
            road_start = int(H * (1.0 - road_ratio))
            weight[:, :, road_start:, :] = 3.0 
            
            diff = (f_adv - f_cln) ** 2
            weighted_diff = diff * weight
            loss += weighted_diff.mean() * ATTACK_PARAMS['w_feature']
            loss += F.cosine_similarity(f_adv.view(B, -1), f_cln.view(B, -1)).mean() * ATTACK_PARAMS['w_cos']

        if 'trans' in feat_adv:
            f_adv, f_cln = feat_adv['trans'][0], feat_clean['trans'][0]
            loss += F.mse_loss(f_adv, f_cln) * ATTACK_PARAMS['w_feature']
        return loss

    def get_attention_weighted_loss(self, feat_adv, feat_clean, clean_attn_map):
        if 'trans' not in feat_adv or clean_attn_map is None:
            return 0.0
        f_adv, f_cln = feat_adv['trans'][0], feat_clean['trans'][0]
        importance = clean_attn_map.mean(dim=1).mean(dim=1) 
        importance = F.softmax(importance * ATTACK_PARAMS['attn_temp'], dim=-1).unsqueeze(-1) 
        b, c, h, w = f_adv.shape
        f_adv_flat = f_adv.view(b, c, -1).permute(0, 2, 1) 
        f_cln_flat = f_cln.view(b, c, -1).permute(0, 2, 1)
        n_tokens = f_adv_flat.shape[1]
        if importance.shape[1] >= n_tokens:
            imp_cut = importance[:, :n_tokens, :]
            w_loss = ((f_adv_flat - f_cln_flat)**2 * imp_cut).mean()
            return w_loss * ATTACK_PARAMS['w_attn_focus']
        return 0.0

    def run_diff_attack(self, rgb_raw, batch_data):
        img_01 = rgb_raw.clone().float() / 255.0
        bs = rgb_raw.shape[0]
        
        # 准备 Dummy Tensors
        dummy_bev = torch.zeros(bs, 160, 160).to(self.device).long()
        dummy_label = torch.zeros(bs, 1, 7).to(self.device).float()
        dummy_depth = torch.zeros(bs, 160, 704).to(self.device).float()
        dummy_semantic = torch.zeros(bs, 160, 704).to(self.device).long()
        
        batch_data['target_point_image'] = self.fix_target_point_image(batch_data['target_point_image'])
        
        # === Step 1: Clean Pass (获取干净特征) ===
        self.tf_extractor.clear()
        with torch.no_grad():
            img_norm_clean = self.normalize_tf(img_01)
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

        controller = DiffAttackController()
        controller.mode = "off"
        register_attention_control(self.sd_pipe.unet, controller)

        # === Step 2: DDIM Inversion ===
        with torch.no_grad():
            latents_orig = self.sd_pipe.vae.encode(img_01 * 2 - 1).latent_dist.sample() * 0.18215
        
        t_idx = ATTACK_PARAMS["start_step"]
        initial_noisy_latents = self.ddim_inversion(latents_orig, t_idx)
        target_t_index = len(self.sd_pipe.scheduler.timesteps) - t_idx - 1
        t = self.sd_pipe.scheduler.timesteps[target_t_index]

        # === Step 3: SD Attention Control ===
        controller.mode = "store"; controller.reset()
        with torch.no_grad():
            _ = self.sd_pipe.unet(initial_noisy_latents, t, encoder_hidden_states=self.null_embs)
        
        # === Step 4: Optimization Loop ===
        noisy_latents = initial_noisy_latents.detach().clone().requires_grad_(True)
        optimizer = torch.optim.AdamW([noisy_latents], lr=ATTACK_PARAMS['lr'])
        controller.mode = "loss"

        for _ in range(ATTACK_PARAMS['iterations']):
            optimizer.zero_grad()
            controller.reset()
            
            noise_pred = self.sd_pipe.unet(noisy_latents, t, encoder_hidden_states=self.null_embs).sample
            loss_sd_struct = controller.loss 

            alpha_prod_t = self.sd_pipe.scheduler.alphas_cumprod[t]
            beta_prod_t = 1 - alpha_prod_t
            pred_z0 = (noisy_latents - beta_prod_t**0.5 * noise_pred) / alpha_prod_t**0.5
            
            decoded = self.sd_pipe.vae.decode(pred_z0 / 0.18215).sample
            adv_img_01 = (decoded / 2 + 0.5).clamp(0, 1)
            adv_rgb_255 = adv_img_01 * 255.0
            
            self.tf_extractor.clear()
            self.tf_model(
                adv_rgb_255, batch_data['lidar'], batch_data['ego_waypoint'], batch_data['target_point'], 
                batch_data['target_point_image'], batch_data['speed'].reshape(-1,1),
                dummy_bev, dummy_label, dummy_depth, dummy_semantic
            )
            feat_adv = self.tf_extractor.features
            
            loss_road = self.get_road_focused_loss(feat_adv, feat_clean)
            loss_attn = self.get_attention_weighted_loss(feat_adv, feat_clean, clean_tf_attn)
            
            total_loss = ATTACK_PARAMS['w_sd_struct'] * loss_sd_struct - loss_road - loss_attn
            if torch.isnan(total_loss): break
            total_loss.backward()
            optimizer.step()
            
            with torch.no_grad():
                diff = noisy_latents - initial_noisy_latents
                diff = torch.clamp(diff, -ATTACK_PARAMS['epsilon_latent'], ATTACK_PARAMS['epsilon_latent'])
                noisy_latents.copy_(initial_noisy_latents + diff)

        # === Step 5: Final Generation ===
        with torch.no_grad():
            controller.mode = "replace"
            latents = noisy_latents.detach().clone()
            
            start_idx = len(self.sd_pipe.scheduler.timesteps) - t_idx - 1
            for i in range(start_idx, len(self.sd_pipe.scheduler.timesteps)):
                t_curr = self.sd_pipe.scheduler.timesteps[i]
                controller.layer_idx = 0; 
                noise_pred = self.sd_pipe.unet(latents, t_curr, encoder_hidden_states=self.null_embs).sample
                latents = self.sd_pipe.scheduler.step(noise_pred, t_curr, latents).prev_sample
                controller.step() 

            final = self.sd_pipe.vae.decode(latents / 0.18215).sample
            final_01 = (final / 2 + 0.5).clamp(0, 1)
            final_255 = final_01 * 255.0
            
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
            if self.lpips_vgg: lpips_val = self.lpips_vgg(final_01*2-1, img_01*2-1).mean().item()
            ssim_val = self.ssim_loss(final_01*2-1, img_01*2-1).item()

        return final_255, pred_wp_clean, pred_adv, drift, lpips_val, ssim_val

    def run_patch_attack(self, rgb_raw, batch_data):
        bs = rgb_raw.shape[0]
        H, W = rgb_raw.shape[2], rgb_raw.shape[3] 
        img_01 = rgb_raw.clone().float() / 255.0
        
        cam_width = W // 3
        boundaries = [0, cam_width, cam_width * 2, W]
        
        dummy_bev = torch.zeros(bs, 160, 160).to(self.device).long()
        dummy_label = torch.zeros(bs, 1, 7).to(self.device).float()
        dummy_depth = torch.zeros(bs, 160, 704).to(self.device).float()
        dummy_semantic = torch.zeros(bs, 160, 704).to(self.device).long()
        batch_data['target_point_image'] = self.fix_target_point_image(batch_data['target_point_image'])
        
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
        
        t_idx = ATTACK_PARAMS["start_step"]
        target_t_index = len(self.sd_pipe.scheduler.timesteps) - t_idx - 1
        t = self.sd_pipe.scheduler.timesteps[target_t_index]
        final_patches = []
        
        for patch_idx in range(3):
            x_start, x_end = boundaries[patch_idx], boundaries[patch_idx + 1]
            patch = img_01[:, :, :, x_start:x_end].clone()
            patch_size = patch.shape[2:] 
            
            patch_256 = F.interpolate(patch, size=(256, 256), mode='bilinear')
            patch_norm = patch_256 * 2 - 1
            
            controller = DiffAttackController()
            controller.mode = "off"
            register_attention_control(self.sd_pipe.unet, controller)
            
            with torch.no_grad():
                latents_orig = self.sd_pipe.vae.encode(patch_norm).latent_dist.sample() * 0.18215
                initial_noisy = self.ddim_inversion(latents_orig, t_idx)
            
            controller.mode = "store"; controller.reset()
            with torch.no_grad():
                _ = self.sd_pipe.unet(initial_noisy, t, encoder_hidden_states=self.null_embs)
            
            noisy_latents = initial_noisy.detach().clone().requires_grad_(True)
            optimizer = torch.optim.AdamW([noisy_latents], lr=ATTACK_PARAMS['lr'])
            controller.mode = "loss"
            
            patch_iters = ATTACK_PARAMS['iterations'] // 3
            
            for _ in range(patch_iters):
                optimizer.zero_grad()
                controller.reset()
                
                noise_pred = self.sd_pipe.unet(noisy_latents, t, encoder_hidden_states=self.null_embs).sample
                loss_sd_struct = controller.loss
                
                alpha_prod_t = self.sd_pipe.scheduler.alphas_cumprod[t]
                beta_prod_t = 1 - alpha_prod_t
                pred_z0 = (noisy_latents - beta_prod_t**0.5 * noise_pred) / alpha_prod_t**0.5
                
                decoded = self.sd_pipe.vae.decode(pred_z0 / 0.18215).sample
                adv_img_01 = (decoded / 2 + 0.5).clamp(0, 1)
                adv_patch_resized = F.interpolate(adv_img_01, size=patch_size, mode='bilinear')
                
                img_adv_full = img_01.clone()
                img_adv_full[:, :, :, x_start:x_end] = adv_patch_resized
                adv_rgb_255 = img_adv_full * 255.0
                
                self.tf_extractor.clear()
                self.tf_model(
                    adv_rgb_255, batch_data['lidar'], batch_data['ego_waypoint'], batch_data['target_point'],
                    batch_data['target_point_image'], batch_data['speed'].reshape(-1,1),
                    dummy_bev, dummy_label, dummy_depth, dummy_semantic
                )
                feat_adv = self.tf_extractor.features
                
                loss_road = self.get_road_focused_loss(feat_adv, feat_clean)
                loss_attn = self.get_attention_weighted_loss(feat_adv, feat_clean, clean_tf_attn)
                
                total_loss = ATTACK_PARAMS['w_sd_struct'] * loss_sd_struct - loss_road - loss_attn
                if torch.isnan(total_loss): break
                total_loss.backward()
                optimizer.step()
                
                with torch.no_grad():
                    diff = noisy_latents - initial_noisy
                    diff = torch.clamp(diff, -ATTACK_PARAMS['epsilon_latent'], ATTACK_PARAMS['epsilon_latent'])
                    noisy_latents.copy_(initial_noisy + diff)
            
            with torch.no_grad():
                controller.mode = "replace"
                latents = noisy_latents.detach().clone()
                start_idx = len(self.sd_pipe.scheduler.timesteps) - t_idx - 1
                for j in range(start_idx, len(self.sd_pipe.scheduler.timesteps)):
                    t_curr = self.sd_pipe.scheduler.timesteps[j]
                    noise_pred = self.sd_pipe.unet(latents, t_curr, encoder_hidden_states=self.null_embs).sample
                    latents = self.sd_pipe.scheduler.step(noise_pred, t_curr, latents).prev_sample
                    controller.step()
                
                final = self.sd_pipe.vae.decode(latents / 0.18215).sample
                final_01 = (final / 2 + 0.5).clamp(0, 1)
                final_patch = F.interpolate(final_01, size=patch_size, mode='bilinear')
                final_patches.append(final_patch)
            
            del noisy_latents, initial_noisy, latents_orig, controller
            torch.cuda.empty_cache()
        
        with torch.no_grad():
            final_01 = torch.cat(final_patches, dim=3)
            final_255 = final_01 * 255.0
            
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
def save_visualization_unified(rgb_clean, rgb_adv, pred_clean, pred_adv, gt_wp, target_point, save_path, drift, title_prefix=""):
    img_clean = np.clip(rgb_clean.permute(1,2,0).cpu().numpy()/255.0, 0, 1)
    img_adv = np.clip(rgb_adv.permute(1,2,0).cpu().numpy()/255.0, 0, 1)
    noise = np.clip(np.abs(img_adv - img_clean) * 15.0, 0, 1)

    fig = plt.figure(figsize=(16, 14))
    gs = fig.add_gridspec(2, 2, hspace=0.2, wspace=0.15)
    
    ax1 = fig.add_subplot(gs[0, 0]); ax1.imshow(img_clean)
    ax1.set_title(f"{title_prefix} 1. Clean", fontsize=14, fontweight='bold'); ax1.axis('off')
    
    ax2 = fig.add_subplot(gs[0, 1]); ax2.imshow(img_adv)
    ax2.set_title("2. Adv (DiffAttack Aligned)", fontsize=14, fontweight='bold'); ax2.axis('off')
    
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
    parser.add_argument('--output_dir', type=str, default='da_gen_results')
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    sys.stdout = Logger(os.path.join(args.output_dir, "attack_log.txt"))
    
    # === Global FID Directories (MFA Style) ===
    global_fid_c = os.path.join(args.output_dir, "fid_global_clean")
    global_fid_a = os.path.join(args.output_dir, "fid_global_adv")
    if os.path.exists(global_fid_c): shutil.rmtree(global_fid_c)
    if os.path.exists(global_fid_a): shutil.rmtree(global_fid_a)
    os.makedirs(global_fid_c, exist_ok=True)
    os.makedirs(global_fid_a, exist_ok=True)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    ATTACK_PARAMS['device'] = f"cuda:{args.gpu}"
    
    print("=== TransFuser DiffAttack (Generation Mode) ===")
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

    attacker = TransFuserDiffAttacker(model, device)
    
    # [Fix] 使用 MFA 的扫描逻辑
    datasets = find_valid_datasets(args.root_dir)
    
    global_stats = {
        'shift': [], 'lat_err': [], 'lon_err': [], 'sim_drift': [], 
        'lpips': [], 'ssim': [], 'clean_len': [], 'success': [], 
        'global_tan': [], 'avg_time': []
    }
    
    start_time = time.time()
    total_processed = 0

    for ds_idx, ds_path in enumerate(datasets):
        # 提取路线名
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
        
        dataset = CARLA_Data(root=[ds_path], config=config, shared_dict=None)
        
        # 预先创建 Dummy Tensors
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

            rgb_raw = data['rgb'].to(device, dtype=torch.float32)
            gt_wp = data['ego_waypoint'].to(device, dtype=torch.float32)
            
            # 兼容性 resize (MFA 逻辑)
            if rgb_raw.shape[-2:] != (160, 704):
                 rgb = torch.nn.functional.interpolate(rgb_raw, size=(160, 704), mode='bilinear', align_corners=False)
            else:
                 rgb = rgb_raw

            batch = {
                'lidar': lidar,
                'ego_waypoint': gt_wp,
                'target_point': target_point,
                'target_point_image': tp_img,
                'speed': speed,
                'bev_label': dummy_label, 
                'depth': dummy_depth, 
                'semantic': dummy_semantic,
                'bev_points': data.get('bev_points').long().to(device) if 'bev_points' in data else None,
                'cam_points': data.get('cam_points').long().to(device) if 'cam_points' in data else None
            }

            try:
                if ATTACK_PARAMS.get('use_patch_attack', False):
                    rgb_adv, p_clean, p_adv, drift, lp, ss = attacker.run_patch_attack(rgb, batch)
                else:
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
                
                # === 图片保存 (MFA Style) ===
                filename = f"{ds_name}_{batch_idx:04d}.png"
                
                # 1. 保存到路线文件夹
                Image.fromarray(rgb[0].permute(1,2,0).cpu().numpy().astype(np.uint8)).save(
                    os.path.join(clean_save_dir, filename)
                )
                Image.fromarray(rgb_adv[0].permute(1,2,0).cpu().numpy().astype(np.uint8)).save(
                    os.path.join(adv_save_dir, filename)
                )

                # 2. 保存到全局 FID 文件夹
                Image.fromarray(rgb[0].permute(1,2,0).cpu().numpy().astype(np.uint8)).save(
                    os.path.join(global_fid_c, filename)
                )
                Image.fromarray(rgb_adv[0].permute(1,2,0).cpu().numpy().astype(np.uint8)).save(
                    os.path.join(global_fid_a, filename)
                )

                # 3. 定期保存可视化
                if batch_idx % 20 == 0:
                    vis_filename = f"{ds_name}_{batch_idx:04d}_vis.png"
                    save_visualization_unified(
                        rgb[0], rgb_adv[0], clean_wp, adv_wp, gt_wp[0].cpu().numpy(), 
                        batch['target_point'][0].cpu().numpy(), 
                        os.path.join(vis_save_dir, vis_filename), 
                        drift, title_prefix=f"{ds_name} #{batch_idx}"
                    )
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
        print("MACRO AVERAGE METRICS (DiffAttack Generation):")
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
        except: 
            print("  > FID Calculation Failed")
        print("="*60)

if __name__ == "__main__":
    main()