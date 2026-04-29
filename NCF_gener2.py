import os
# 优化显存分配
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
import glob
import json
import gzip
import math
import argparse
import time
import shutil
import gc
from datetime import timedelta
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import hydra
from omegaconf import OmegaConf
from transformers import AutoProcessor, AutoConfig
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torchvision.utils import save_image
import torchvision.transforms as T

# [引入] 隐蔽性与评估库
try:
    import lpips
except ImportError:
    print("请先安装 lpips: pip install lpips")
    sys.exit(1)

try:
    from torch_fidelity import calculate_metrics
except ImportError:
    calculate_metrics = None

# ================= 配置 =================
REPO_ROOT = "/home/pc/simlingo" # 请根据实际情况修改
CHECKPOINT_PATH = f"{REPO_ROOT}/outputs/simlingo/checkpoints/epoch=013.ckpt/pytorch_model.pt"
CONFIG_PATH = f"{REPO_ROOT}/outputs/simlingo/.hydra/config.yaml"

# === White-box NCF 攻击参数 (标准两阶段：IR + NS) ===
ATTACK_CONFIG = {
    "batch_size": 1,
    "device": "cuda:0",
    
    # === Phase 1: Initialization Reset (IR) ===
    "num_reset": 20,         # 对应原论文的 eta，尝试多少种随机风格
    
    # === Phase 2: Neighborhood Search (NS) ===
    "num_iter": 100,         # 梯度下降迭代次数
    "lr_T": 0.05,            # T 矩阵的学习率
    "lr_mu": 0.05,           # 均值偏移的学习率
    "momentum": 0.9,         # 动量
    
    # === 约束参数 ===
    "lambda_sim": 10.0,      # 攻击 Loss 权重
    "lambda_reg": 0.1,       # 正则化权重
    "lambda_mu": 0.05,       # 均值约束权重
    
    # === 限制幅度 ===
    "epsilon": 16.0 / 255,   # RGB 空间的 L_inf 约束
    "mu_limit": 0.3,         # Lab 空间均值偏移限制 (放宽以允许更明显的换色)
    
    # === Loss 选择 ===
    "loss_type": "cosine",   # 'mse' or 'cosine'
    
    # === 区域控制 ===
    "road_region": [0.35, 0.95],
    "images_size": 448,
}

# ================= 全局变量更新 =================
CMD_MAPPING = {
    0: "Straight", 
    1: "Turn Left",
    2: "Turn Right",
    3: "Go Straight",
    4: "Follow Lane",
    5: "Change Lane Left",
    6: "Change Lane Right"
}

def get_cmd_text(cmd_id):
    return CMD_MAPPING.get(cmd_id, "Follow Lane")

sys.path.append(REPO_ROOT)
try:
    from simlingo_training.utils.custom_types import DrivingInput, LanguageLabel
    from team_code.simlingo_utils import get_camera_intrinsics, get_camera_extrinsics
except ImportError as e:
    print(f"SimLingo 环境加载失败: {e}")
    sys.exit(1)

# ================= Logger =================
class Logger(object):
    def __init__(self, filename="default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a")
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
    def flush(self): pass

# ================= SSIM 模块 =================
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

# ================= 辅助函数 =================
def world_to_ego(route_global, ego_x, ego_y, ego_theta_rad):
    route_global = np.asarray(route_global, dtype=float)
    if route_global.ndim != 2 or route_global.shape[1] < 2: raise ValueError("route_global must be (N,2)")
    yaw = float(ego_theta_rad)
    diff = route_global - np.array([float(ego_x), float(ego_y)])
    dx = diff[:, 0]; dy = diff[:, 1]
    c, s = np.cos(yaw), np.sin(yaw)
    local_x = dx * c + dy * s
    local_y = -dx * s + dy * c
    return np.stack([local_x, local_y], axis=-1)

# ================= 核心：可微分的色彩转换 (Float32) =================
# NCF 攻击变量计算必须保持 Float32 以确保数值稳定
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

# ================= 攻击主类 =================
class SimLingoNCFAttacker:
    def __init__(self, config):
        self.cfg = config
        self.device = torch.device(config["device"])
        
        print(f"Loading Config: {CONFIG_PATH}")
        with open(CONFIG_PATH, 'r') as f: self.hydra_cfg = OmegaConf.load(f)
        
        variant = self.hydra_cfg.model.vision_model.variant
        self.processor = AutoProcessor.from_pretrained(variant, trust_remote_code=True)
        self.tokenizer = getattr(self.processor, 'tokenizer', self.processor)
        self.tokenizer.padding_side = "left"
        self.tokenizer.add_special_tokens({'additional_special_tokens': ['<WAYPOINTS>', '<TARGET_POINT>']})
        
        self.tmp_config = AutoConfig.from_pretrained(variant, trust_remote_code=True)
        self.num_image_token = int((448 // self.tmp_config.vision_config.patch_size) ** 2 * (0.5 ** 2))

        print("Loading SimLingo Model...")
        # 【关键修复】：恢复 BFloat16 默认类型，适配模型权重
        torch.set_default_dtype(torch.bfloat16)
        
        cache_dir = f"pretrained/{variant.split('/')[-1]}"
        self.model = hydra.utils.instantiate(
            self.hydra_cfg.model, 
            cfg_data_module=self.hydra_cfg.data_module, 
            processor=self.processor, 
            cache_dir=cache_dir,
            _recursive_=False
        ).to(self.device)
        
        state_dict = torch.load(CHECKPOINT_PATH, map_location=self.device)
        if 'state_dict' in state_dict: state_dict = state_dict['state_dict']
        new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        self.model.load_state_dict(new_state_dict, strict=False)
        self.model.eval()
        
        for param in self.model.parameters(): 
            param.requires_grad = False
        
        self.vision_model = self._hunt_vision_model(self.model)
        
        # 评估模型使用 Float32
        self.lpips_vgg = lpips.LPIPS(net='vgg').to(self.device).float().requires_grad_(False)
        self.ssim_loss = SSIM().to(self.device).float()

        # 统计量 (Float32)
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=self.device, dtype=torch.float32).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=self.device, dtype=torch.float32).view(1, 3, 1, 1)

    def _hunt_vision_model(self, model):
        def is_viable_backbone(mod):
            has_embed = hasattr(mod, 'embeddings') or hasattr(mod, 'patch_embed') or hasattr(mod, 'patch_embedding')
            try:
                forward_func = getattr(mod, 'forward', None)
                return has_embed and callable(forward_func)
            except: return False

        def recursive_search(module, depth=0):
            if depth > 6: return None
            if is_viable_backbone(module): return module
            candidates = ['vision_model', 'vision_tower', 'image_encoder', 'model', 'backbone']
            for name in candidates:
                if hasattr(module, name):
                    res = recursive_search(getattr(module, name), depth+1)
                    if res: return res
            if 'Wrapper' in module.__class__.__name__ or 'Encoder' in module.__class__.__name__:
                for child in module.children():
                    res = recursive_search(child, depth+1)
                    if res: return res
            return None

        target = recursive_search(model)
        if target is None:
            if hasattr(model, 'vision_model'): 
                target = list(model.vision_model.children())[0]
        
        if target is None:
             raise ValueError("CRITICAL: Failed to locate Vision Transformer!")
        print(f">>> [TARGET LOCKED] Found Vision Backbone: {type(target).__name__}")
        return target
    
    def load_measurements(self, image_path):
        img_dir = os.path.dirname(image_path)
        root_dir = os.path.dirname(img_dir)
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        
        meas_path = os.path.join(root_dir, 'measurements', f"{base_name}.json.gz")
        if not os.path.exists(meas_path): meas_path = os.path.join(root_dir, 'measurements', f"{base_name}.json")
        boxes_path = os.path.join(root_dir, 'boxes', f"{base_name}.json.gz")
        if not os.path.exists(boxes_path): boxes_path = os.path.join(root_dir, 'boxes', f"{base_name}.json")

        speed = 0.0; cmd_id = 4; traffic_light_state = "Green"
        tp1 = np.array([10.0, 0.0]); tp2 = np.array([20.0, 0.0]); gt_local = None

        if os.path.exists(meas_path):
            try:
                open_func = gzip.open if meas_path.endswith('.gz') else open
                with open_func(meas_path, 'rt', encoding='utf-8') as f:
                    data = json.load(f)
                speed = float(data.get('speed', speed))
                cmd_id = int(data.get('command', cmd_id))
                ego_x = ego_y = 0.0
                pos_global = data.get('pos_global', None)
                if pos_global: ego_x, ego_y = float(pos_global[0]), float(pos_global[1])
                else: 
                    ego_matrix = data.get('ego_matrix', None)
                    if ego_matrix: ego_x, ego_y = float(ego_matrix[0][3]), float(ego_matrix[1][3])
                ego_theta = float(data.get('theta', 0.0))
                if abs(ego_theta) > 2 * math.pi: ego_theta = math.radians(ego_theta)
                if 'route' in data:
                    route = np.array(data['route'], dtype=float)
                    if route.ndim >= 2: route = route[:, :2]
                    if np.max(np.abs(route)) > 1000.0: gt_local = world_to_ego(route, ego_x, ego_y, ego_theta)
                    else: gt_local = route
                if gt_local is not None and len(gt_local) > 0:
                    dists = np.linalg.norm(gt_local, axis=1)
                    idx1 = np.where(dists > 10.0)[0]
                    tp1 = gt_local[idx1[0]] if len(idx1) > 0 else gt_local[-1]
                    idx2 = np.where(dists > 20.0)[0]
                    tp2 = gt_local[idx2[0]] if len(idx2) > 0 else gt_local[-1]
            except Exception as e: print(f"Error loading measurements {meas_path}: {e}")

        if os.path.exists(boxes_path):
            try:
                open_func = gzip.open if boxes_path.endswith('.gz') else open
                with open_func(boxes_path, 'rt', encoding='utf-8') as f:
                    boxes_data = json.load(f)
                    if isinstance(boxes_data, list):
                        for item in boxes_data:
                            if item.get('class') == 'ego_info':
                                raw_tl = item.get('traffic_light_state', 'None')
                                if raw_tl != "None": traffic_light_state = raw_tl
                                break
            except Exception as e: print(f"Error loading boxes {boxes_path}: {e}")
        return speed, cmd_id, traffic_light_state, tp1, tp2, gt_local

    def prepare_inputs(self, image_path):
        speed, cmd_id, tl_state, tp1, tp2, gt_route = self.load_measurements(image_path)
        cmd_text = get_cmd_text(cmd_id)
        
        # [Modified for Center Crop]
        # 读取原始图片并记录尺寸
        raw_pil = Image.open(image_path).convert('RGB')
        orig_W, orig_H = raw_pil.size
        
        # 1. 计算 Center Crop 坐标
        crop_size = min(orig_W, orig_H)
        left = (orig_W - crop_size) // 2
        top = (orig_H - crop_size) // 2
        right = left + crop_size
        bottom = top + crop_size
        crop_coords = (left, top, right, bottom, crop_size)
        
        # 2. 执行裁剪
        img_cropped = raw_pil.crop((left, top, right, bottom))
        
        # 3. Resize 到 448x448 (从正方形 resize 到 正方形，无畸变)
        input_pil = img_cropped.resize((448, 448))
        
        # 基础输入为 Float32 (用于攻击计算)，后续需要转 BFloat16 (用于模型推理)
        img_tensor = T.ToTensor()(input_pil).unsqueeze(0).to(self.device).float()
        
        prompt_tp = "Target waypoint: <TARGET_POINT><TARGET_POINT>."
        base_prompt = (f"Current speed: {speed:.2f} m/s. Command: {cmd_text}. "
                       f"Traffic light: {tl_state}. {prompt_tp} Predict the waypoints.")
        image_tokens = '<img>' + '<IMG_CONTEXT>' * self.num_image_token * 2 + '</img>'
        final_prompt = image_tokens + "\n" + base_prompt
        tp_ids = self.tokenizer.convert_tokens_to_ids('<TARGET_POINT>')
        tp_vals = np.stack([tp1, tp2]).astype(np.float32)
        placeholder = {tp_ids: tp_vals}
        tokens = self.tokenizer([final_prompt], padding=True, return_tensors="pt")
        ll = LanguageLabel(
            phrase_ids=tokens["input_ids"].to(self.device), 
            phrase_valid=tokens["attention_mask"].bool().to(self.device), 
            phrase_mask=tokens["attention_mask"].bool().to(self.device), 
            placeholder_values=[placeholder], language_string=[final_prompt], loss_masking=None
        )
        intrinsics = get_camera_intrinsics(448, 448, fov=110).unsqueeze(0).to(self.device).float().view(1, 3, 3)
        extrinsics = get_camera_extrinsics().unsqueeze(0).to(self.device).float().view(1, 4, 4)
        
        # Meta 信息使用 BFloat16 适配模型
        meta = {
            "image_sizes": torch.tensor([[448, 448]]).to(self.device),
            "camera_intrinsics": intrinsics, "camera_extrinsics": extrinsics,
            "vehicle_speed": torch.tensor([[speed]]).to(self.device).bfloat16(),
            "target_point": torch.from_numpy(tp1).bfloat16().unsqueeze(0).to(self.device),
            "prompt": ll, "prompt_inference": ll
        }
        # 返回新增 crop_coords
        return img_tensor, meta, raw_pil, gt_route, speed, tp1, (orig_W, orig_H), crop_coords

    def normalize(self, img_01):
        # 输入 Float32，输出 Float32
        norm_img = (img_01 - self.mean) / self.std
        return norm_img.unsqueeze(1).unsqueeze(1).repeat(1, 1, 2, 1, 1, 1)

    # 【关键修改】参考 FGSM 代码的 forward_vision_only
    def forward_vision_only(self, img_tensor_normalized, return_attn=False):
        """
        处理维度展平和类型转换，确保梯度流。
        """
        # 展平维度 [1, 1, 2, 3, H, W] -> [2, 3, H, W] (如果 batch>1 或 views>1)
        # 这里 SimLingo 输入是 6D [B, T, V, C, H, W]
        # img_tensor_normalized 是 Float32
        if img_tensor_normalized.dim() == 6:
            B, T, N, C, H, W = img_tensor_normalized.shape
            pixel_values = img_tensor_normalized.view(B * T * N, C, H, W)
        else:
            pixel_values = img_tensor_normalized

        outputs = None; attn_weights = None
        
        # 强制转换为 BFloat16 送入模型 (梯度会穿过这个 cast)
        pixel_values_bf16 = pixel_values.bfloat16()
        
        try:
            outputs = self.vision_model(pixel_values=pixel_values_bf16, output_attentions=return_attn)
        except TypeError:
            try: outputs = self.vision_model(pixel_values_bf16)
            except: 
                child = list(self.vision_model.children())[0]
                outputs = child(pixel_values_bf16)

        if hasattr(outputs, 'last_hidden_state'): features = outputs.last_hidden_state
        elif hasattr(outputs, 'pooler_output'): features = outputs.pooler_output
        elif isinstance(outputs, tuple): features = outputs[0]
        else: features = outputs
        
        # 特征转回 Float32 以便计算 Loss
        features = features.float()
        
        if return_attn:
            if hasattr(outputs, 'attentions') and outputs.attentions is not None:
                attn_weights = outputs.attentions[-1]
            elif isinstance(outputs, tuple):
                for item in outputs:
                    if isinstance(item, (list, tuple)) and len(item) > 0:
                        if isinstance(item[-1], torch.Tensor) and item[-1].dim() == 4:
                            attn_weights = item[-1]; break
            if attn_weights is not None:
                attn_weights = attn_weights.float()
            return features, attn_weights
        
        return features

    # ================= 核心攻击逻辑：White-box NCF =================
    # ================= 核心攻击逻辑：White-box NCF (修复 Dtype 问题) =================
    def ncf_attack(self, img_tensor, clean_features):
        """
        NCF White-box Attack: Optimize T and delta_mu.
        All calculations inside this function MUST be in Float32.
        """
        batch_size, _, H, W = img_tensor.shape
        device = self.device
        cfg = self.cfg
        
        # 1. 预计算 (No Grad) - 确保 lab_clean 是 float32
        with torch.no_grad():
            # img_tensor 应该是 float32 (由 prepare_inputs 保证)
            lab_clean = rgb_to_lab_differentiable(img_tensor)
            mu_clean = torch.mean(lab_clean, dim=[2, 3], keepdim=True) 
            clean_feat_target = clean_features.detach() 
            # [B, 3, H, W] -> [B, H, W, 3] 用于矩阵乘法
            lab_centered = (lab_clean - mu_clean).permute(0, 2, 3, 1).reshape(batch_size, -1, 3)

        # ------------------------------------------------------------------
        # Phase 1: Initialization Reset (IR) - 随机搜索
        # ------------------------------------------------------------------
        num_search = cfg.get('num_search', 20)
        
        # 【关键修复】：显式指定 dtype=torch.float32
        best_T_init = torch.eye(3, device=device, dtype=torch.float32).unsqueeze(0).repeat(batch_size, 1, 1)
        best_mu_init = torch.zeros((batch_size, 3, 1, 1), device=device, dtype=torch.float32)
        best_score_init = 1.0 
        
        for _ in range(num_search):
            with torch.no_grad():
                # 1. 随机生成 T (单位矩阵 + 噪声) -> Float32
                cand_T = torch.eye(3, device=device, dtype=torch.float32).unsqueeze(0).repeat(batch_size, 1, 1)
                
                # 随机数也要 Float32
                scale = torch.rand(batch_size, 3, device=device, dtype=torch.float32) * 1.0 + 0.5 
                cand_T[:, 0, 0] = scale[:, 0]
                cand_T[:, 1, 1] = scale[:, 1]
                cand_T[:, 2, 2] = scale[:, 2]
                
                noise_T = torch.randn_like(cand_T, dtype=torch.float32) * 0.1
                noise_T[:, 0, 1:] = 0; noise_T[:, 1:, 0] = 0 
                cand_T = cand_T + noise_T
                
                # 2. 随机生成 mu 偏移 -> Float32
                cand_mu = torch.zeros((batch_size, 3, 1, 1), device=device, dtype=torch.float32)
                cand_mu[:, 0] = (torch.rand(batch_size, 1, 1, device=device, dtype=torch.float32) - 0.5) * 0.4 
                cand_mu[:, 1:] = (torch.rand(batch_size, 2, 1, 1, device=device, dtype=torch.float32) - 0.5) * 0.6 
                
                # 3. 快速评估
                # Float32 @ Float32 -> OK
                lab_adv_flat = torch.bmm(lab_centered, cand_T)
                lab_adv = lab_adv_flat.reshape(batch_size, H, W, 3).permute(0, 3, 1, 2)
                lab_adv = lab_adv + mu_clean + cand_mu
                
                adv_img = lab_to_rgb_differentiable(lab_adv)
                adv_img = torch.clamp(adv_img, 0, 1)
                
                delta = adv_img - img_tensor
                delta = torch.clamp(delta, -cfg['epsilon'], cfg['epsilon'])
                adv_img = torch.clamp(img_tensor + delta, 0, 1)
                
                # 前向传播 (Mix Precision: Float32 -> BFloat16 inside)
                norm_adv = self.normalize(adv_img)
                feat_adv = self.forward_vision_only(norm_adv)
                
                score = F.cosine_similarity(feat_adv.flatten(1), clean_feat_target.flatten(1)).mean().item()
                
                if score < best_score_init:
                    best_score_init = score
                    best_T_init = cand_T.clone()
                    best_mu_init = cand_mu.clone()

        # ------------------------------------------------------------------
        # Phase 2: Neighborhood Search (NS) - 梯度下降优化
        # ------------------------------------------------------------------
        # 确保优化变量是 Float32
        T_matrix = best_T_init.detach().float().requires_grad_(True)
        delta_mu = best_mu_init.detach().float().requires_grad_(True)
        
        optimizer = torch.optim.SGD([
            {'params': [T_matrix], 'lr': cfg['lr_T']},
            {'params': [delta_mu], 'lr': cfg['lr_mu']}
        ], momentum=cfg['momentum'])
        
        best_adv_img = img_tensor.clone().detach()
        best_metric = float('inf') 
        
        for i in range(cfg['num_iter']):
            # --- A. 应用 NCF 变换 ---
            # lab_centered (Float32) @ T_matrix (Float32)
            lab_adv_flat = torch.bmm(lab_centered, T_matrix) 
            lab_adv = lab_adv_flat.reshape(batch_size, H, W, 3).permute(0, 3, 1, 2) 
            
            delta_mu_clamped = torch.clamp(delta_mu, -cfg['mu_limit'], cfg['mu_limit'])
            lab_adv = lab_adv + mu_clean + delta_mu_clamped
            
            adv_img = lab_to_rgb_differentiable(lab_adv)
            
            # Epsilon 约束
            delta_rgb = adv_img - img_tensor
            delta_rgb = torch.clamp(delta_rgb, -cfg['epsilon'], cfg['epsilon'])
            adv_img_constrained = torch.clamp(img_tensor + delta_rgb, 0, 1)
            
            # --- B. 前向传播 ---
            norm_input = self.normalize(adv_img_constrained)
            # forward_vision_only 会处理转 BFloat16 的逻辑
            feat_adv = self.forward_vision_only(norm_input)
            
            # --- C. 计算 Loss ---
            if cfg['loss_type'] == 'cosine':
                sim_loss = F.cosine_similarity(feat_adv.flatten(1), clean_feat_target.flatten(1)).mean()
                attack_loss = sim_loss 
            else:
                mse = F.mse_loss(feat_adv, clean_feat_target)
                attack_loss = -mse
                
            # 正则化 (Float32)
            I = torch.eye(3, device=device, dtype=torch.float32).unsqueeze(0)
            reg_T = torch.norm(T_matrix - I, p='fro')
            reg_mu = torch.norm(delta_mu, p=2)
            
            loss = cfg['lambda_sim'] * attack_loss + \
                   cfg['lambda_reg'] * reg_T + \
                   cfg['lambda_mu'] * reg_mu
            
            # --- D. 反向传播 ---
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_([T_matrix, delta_mu], 1.0)
            optimizer.step()
            
            # --- E. 记录最佳 ---
            current_score = attack_loss.item() 
            if i == 0: 
                best_metric = current_score
                best_adv_img = adv_img_constrained.detach()
            else:
                if current_score < best_metric:
                    best_metric = current_score
                    best_adv_img = adv_img_constrained.detach()

        return best_adv_img

    def run(self, img_path, output_dir, fid_clean_dir, fid_adv_dir, file_prefix=""):
        img_name = os.path.basename(img_path).split('.')[0]
        # 读取时多获取了 orig_size 和 crop_coords
        img_clean, meta, pil_clean, gt_route, clean_speed, target_point, (orig_W, orig_H), crop_coords = self.prepare_inputs(img_path)
        
        # 1. 计算原始特征 (Float32 return)
        with torch.no_grad():
            norm_clean = self.normalize(img_clean)
            clean_feat = self.forward_vision_only(norm_clean)
            
            # 【修复点 1】：全模型推理 (Baseline) 必须给 BFloat16
            full_input = DrivingInput(camera_images=norm_clean.bfloat16(), **meta)
            pred_speed_clean, pred_route_clean, _ = self.model(full_input)

        # 2. 执行 NCF 白盒攻击 (在 448x448 的 Center Crop 上进行)
        start_t = time.time()
        img_adv = self.ncf_attack(img_clean, clean_feat)
        attack_time = time.time() - start_t
        
        # 3. 计算指标 (依然基于 448x448，监控攻击对 SimLingo 的影响)
        with torch.no_grad():
            norm_adv = self.normalize(img_adv)
            adv_feat = self.forward_vision_only(norm_adv)
            
            # 【修复点 2】：Adv Inference 必须给 BFloat16
            adv_input = DrivingInput(camera_images=norm_adv.bfloat16(), **meta)
            pred_speed_adv, pred_route_adv, _ = self.model(adv_input)
            
            # --- 后续计算统一转回 Float32 ---
            pred_route_clean = pred_route_clean.float()
            pred_route_adv = pred_route_adv.float()
            pred_speed_clean = pred_speed_clean.float()
            pred_speed_adv = pred_speed_adv.float()
            
            clean_end = pred_route_clean[0, -1] 
            adv_end = pred_route_adv[0, -1]
            
            shift = torch.norm(pred_route_adv - pred_route_clean, p=2).item()
            spd_err = torch.norm(pred_speed_adv - pred_speed_clean, p=2).item()
            lat_error = abs(adv_end[1].item() - clean_end[1].item())
            lon_error = abs(adv_end[0].item() - clean_end[0].item())
            clean_route_len = abs(clean_end[0].item())
            is_success = 1.0 if shift >= 1.0 else 0.0
            
            feat_drift = F.mse_loss(clean_feat, adv_feat).item() * 1000
            
            img_adv_norm = (img_adv * 2 - 1).float() 
            img_clean_norm = (img_clean * 2 - 1).float()
            
            lpips_val = self.lpips_vgg(img_adv_norm, img_clean_norm).mean().item()
            ssim_val = self.ssim_loss(img_adv_norm, img_clean_norm).item()

            # ================= [关键修改] Paste Back 逻辑 =================
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            if file_prefix:
                save_name = f"{file_prefix}_{base_name}.png"
            else:
                save_name = f"{base_name}.png"

            # 1. 准备原始背景 (Clean Canvas)
            # 加载原图全尺寸转 Tensor [0, 1]
            raw_tensor = T.ToTensor()(pil_clean).unsqueeze(0).to(self.device).float()
            
            # 2. 解包 Crop 坐标
            left, top, right, bottom, crop_size = crop_coords
            
            # 3. 放大生成的 448x448 对抗样本到 crop_size
            img_adv_upscaled = F.interpolate(img_adv, size=(crop_size, crop_size), mode='bilinear', align_corners=False)
            img_clean_upscaled = F.interpolate(img_clean, size=(crop_size, crop_size), mode='bilinear', align_corners=False)
            
            # 4. 贴回
            final_full_adv = raw_tensor.clone()
            final_full_clean = raw_tensor.clone()
            
            final_full_adv[:, :, top:bottom, left:right] = img_adv_upscaled
            final_full_clean[:, :, top:bottom, left:right] = img_clean_upscaled
            
            # 定义安全保存为 PNG 的函数
            def save_safe_png(tensor, path):
                # tensor: [0, 1]
                ndarr = tensor[0].mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
                Image.fromarray(ndarr).save(path, format='PNG')

            # 1. 保存 Clean (FID 目录) - 原始尺寸
            save_safe_png(final_full_clean.cpu().float(), os.path.join(fid_clean_dir, save_name))
            
            # 2. 保存 Adv (FID 目录) - 原始尺寸
            save_safe_png(final_full_adv.cpu().float(), os.path.join(fid_adv_dir, save_name))

            # 3. [最重要] 保存 Adv 到数据集输出目录 (这是给 TransFuser 用的) - 原始尺寸
            save_safe_png(final_full_adv.cpu().float(), os.path.join(output_dir, save_name))
            # =========================================================================

        self.save_unified_visualization(img_clean, img_adv, pred_route_clean, pred_route_adv, gt_route, 
                                      target_point, pred_speed_clean, pred_speed_adv, output_dir, img_name)
        
        # 返回值增加 attack_time
        return shift, spd_err, feat_drift, lpips_val, ssim_val, lat_error, lon_error, clean_route_len, is_success, attack_time

    def save_unified_visualization(self, recon_01, adv_01, route_clean, route_adv, 
                                   gt_route, target_point, speed_clean, speed_adv, save_dir, img_name):
        try:
            recon_np = recon_01[0].permute(1, 2, 0).cpu().float().numpy()
            adv_np = adv_01[0].permute(1, 2, 0).cpu().float().numpy()
            recon_np = np.clip(recon_np, 0, 1)
            adv_np = np.clip(adv_np, 0, 1)
            diff_vis = np.clip(np.abs(adv_np - recon_np) * 15.0, 0, 1)
            
            c_pts = route_clean[0].detach().cpu().float().numpy()
            a_pts = route_adv[0].detach().cpu().float().numpy()
            sc_val = speed_clean.float().mean().item() if speed_clean.numel() > 1 else speed_clean.item()
            sa_val = speed_adv.float().mean().item() if speed_adv.numel() > 1 else speed_adv.item()

            fig = plt.figure(figsize=(16, 14))
            gs = fig.add_gridspec(2, 2, hspace=0.2, wspace=0.15)
            
            ax1 = fig.add_subplot(gs[0, 0])
            ax1.imshow(recon_np)
            ax1.set_title("1. Clean (Center Crop)", fontsize=14, fontweight='bold')
            ax1.axis('off')
            
            ax2 = fig.add_subplot(gs[0, 1])
            ax2.imshow(adv_np)
            ax2.set_title("2. Adv (NCF Crop)", fontsize=14, fontweight='bold')
            ax2.axis('off')
            
            ax3 = fig.add_subplot(gs[1, 0])
            ax3.imshow(diff_vis)
            ax3.set_title("3. Noise (x15)", fontsize=14, fontweight='bold')
            ax3.axis('off')

            ax4 = fig.add_subplot(gs[1, 1])
            if gt_route is not None:
                ax4.plot(gt_route[:, 1], gt_route[:, 0], 'g-', alpha=0.3, linewidth=5, label='Ground Truth')
            ax4.plot(c_pts[:, 1], c_pts[:, 0], 'b-o', markersize=5, linewidth=2, label=f'Clean ({sc_val:.1f} m/s)')
            ax4.plot(a_pts[:, 1], a_pts[:, 0], 'r--^', markersize=6, linewidth=2, label=f'Adv ({sa_val:.1f} m/s)')
            
            if target_point is not None:
                tp_np = np.array(target_point)
                tx, ty = (tp_np[0], tp_np[1]) if tp_np.ndim == 1 else (tp_np[0, 0], tp_np[0, 1])
                ax4.scatter(ty, tx, c='gold', marker='*', s=300, edgecolors='black', label='Target', zorder=5)

            ax4.set_xlim(-12, 12); ax4.set_ylim(-2, 40)
            ax4.set_xlabel("Lateral (m)"); ax4.set_ylabel("Forward (m)")
            ax4.grid(True, linestyle=':', alpha=0.6)
            ax4.legend(loc='upper right')
            ax4.set_title("4. Trajectory", fontsize=14, fontweight='bold')

            plt.tight_layout()
            save_path = os.path.join(save_dir, f"{img_name}_vis.png")
            plt.savefig(save_path, dpi=120, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"Vis error: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True, 
                        help="Root directory containing multiple dataset folders")
    parser.add_argument("--output_dir", type=str, default="ncf_transfer_results") 
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--iter', type=int, default=None)
    args = parser.parse_args()

    if args.iter is not None:
        ATTACK_CONFIG["num_iter"] = args.iter

    # 1. 确定本次运行的序号文件夹
    base_dir = args.output_dir
    os.makedirs(base_dir, exist_ok=True)

    existing_folders = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    max_id = 0
    for folder in existing_folders:
        if folder.startswith("NCF-Gener-"):
            try:
                idx = int(folder.split("-")[-1])
                if idx > max_id: max_id = idx
            except ValueError: continue
    
    new_folder_name = f"NCF-Gener-{max_id + 1}"
    current_run_dir = os.path.join(base_dir, new_folder_name)
    os.makedirs(current_run_dir, exist_ok=True)

    # 2. 定义 Log 和 Config 路径
    log_path = os.path.join(current_run_dir, "attack_log.txt")
    config_path = os.path.join(current_run_dir, "attack_config.txt")

    # 3. 保存参数配置
    with open(config_path, 'w') as f:
        f.write(f"=== Execution Info (Run ID: {max_id + 1}) ===\n")
        f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Data Root: {args.data_root}\n")
        f.write(f"Output Directory: {current_run_dir}\n\n")
        f.write("=== White-Box NCF Attack Parameters ===\n")
        json.dump(ATTACK_CONFIG, f, indent=4)
        f.write("\n")

    # 4. 重定向日志输出
    sys.stdout = Logger(log_path)
    
    print(f"Results will be saved to: {current_run_dir}")
    print("-" * 60)
    print("NCF Generation Config:")
    print(json.dumps(ATTACK_CONFIG, indent=2))
    
    attacker = SimLingoNCFAttacker(ATTACK_CONFIG)
    
    # 5. 设置 FID 文件夹
    global_fid_clean = os.path.join(current_run_dir, "global_fid_clean")
    global_fid_adv = os.path.join(current_run_dir, "global_fid_adv")
    if os.path.exists(global_fid_clean): shutil.rmtree(global_fid_clean)
    if os.path.exists(global_fid_adv): shutil.rmtree(global_fid_adv)
    os.makedirs(global_fid_clean); os.makedirs(global_fid_adv)

    # 扫描数据集
    if not os.path.exists(args.data_root):
        print(f"Error: Data root {args.data_root} does not exist."); sys.exit(1)

    all_subdirs = sorted([d for d in os.listdir(args.data_root) if os.path.isdir(os.path.join(args.data_root, d))])
    valid_datasets = []
    for d in all_subdirs:
        if os.path.exists(os.path.join(args.data_root, d, "rgb")):
            valid_datasets.append(d)
    
    if not valid_datasets:
        print(f"No valid datasets found in {args.data_root}"); sys.exit(1)
        
    print(f"\nFound {len(valid_datasets)} Datasets: {valid_datasets}\n")
    overall_start_time = time.time()
    total_samples_processed = 0
    
    global_stats = {
        'shift': [], 'spd_err': [], 'sim_drift': [], 
        'road_lpips': [], 'road_ssim': [], 
        'global_tan': [], 'global_angle': [], 
        'success_rate': [], 'avg_time': []
    }

    # === 循环数据集 ===
    for dataset_idx, dataset_name in enumerate(valid_datasets):
        print(f"\n>>> Processing Dataset [{dataset_idx+1}/{len(valid_datasets)}]: {dataset_name}")
        
        current_rgb_dir = os.path.join(args.data_root, dataset_name, "rgb")
        current_out_dir = os.path.join(current_run_dir, dataset_name)
        os.makedirs(current_out_dir, exist_ok=True)
        
        # 兼容 jpg/png
        images = sorted(glob.glob(os.path.join(current_rgb_dir, "*.png")))
        if not images:
            images = sorted(glob.glob(os.path.join(current_rgb_dir, "*.jpg")))

        ds_totals = {
            'shift': 0.0, 'spd_err': 0.0, 'sim_drift': 0.0, 
            'road_lpips': 0.0, 'road_ssim': 0.0,
            'success_count': 0.0,
            'sum_lat_offset': 0.0,
            'sum_clean_len': 0.0,
            'total_time': 0.0
        }
        count = 0

        for img_path in images:
            t0 = time.time()
            try:
                # 接收攻击时间
                res = attacker.run(img_path, current_out_dir, global_fid_clean, global_fid_adv, file_prefix=dataset_name)
                step_time = time.time() - t0
                
                # 解包 10 个返回值
                shift, spd, drift, lp, ss, lat, lon, clen, succ, atk_time = res
                
                # 单图 Tan
                current_tan = lat / (clen + 1e-6)
                
                # 累加
                ds_totals['shift'] += shift
                ds_totals['spd_err'] += spd
                ds_totals['sim_drift'] += drift
                ds_totals['road_lpips'] += lp
                ds_totals['road_ssim'] += ss
                ds_totals['sum_lat_offset'] += lat 
                ds_totals['sum_clean_len'] += clen 
                ds_totals['success_count'] += succ
                ds_totals['total_time'] += step_time
                
                count += 1
                # 打印信息增加 AtkTime
                print(f"[{dataset_name}][{os.path.basename(img_path)}] Shift: {shift:.2f}m (Lat:{lat:.2f}, Lon:{lon:.2f}) | Spd Diff: {spd:.2f} | Tan: {current_tan:.2f} | AtkTime: {atk_time:.2f}s | Total: {step_time:.2f}s")

            except KeyboardInterrupt: sys.exit(1)
            except Exception as e: 
                print(f"Error: {e}")
                import traceback; traceback.print_exc()
            finally: torch.cuda.empty_cache()

        if count > 0:
            avg_shift = ds_totals['shift'] / count
            avg_spd = ds_totals['spd_err'] / count
            avg_drift = ds_totals['sim_drift'] / count
            avg_lpips = ds_totals['road_lpips'] / count
            avg_ssim = ds_totals['road_ssim'] / count
            avg_time_per_img = ds_totals['total_time'] / count
            succ_rate = (ds_totals['success_count'] / count) * 100
            
            dataset_global_tan = ds_totals['sum_lat_offset'] / (ds_totals['sum_clean_len'] + 1e-6)
            dataset_global_angle = math.degrees(math.atan(dataset_global_tan))
            
            global_stats['shift'].append(avg_shift)
            global_stats['spd_err'].append(avg_spd)
            global_stats['sim_drift'].append(avg_drift)
            global_stats['road_lpips'].append(avg_lpips)
            global_stats['road_ssim'].append(avg_ssim)
            global_stats['global_tan'].append(dataset_global_tan)
            global_stats['global_angle'].append(dataset_global_angle)
            global_stats['success_rate'].append(succ_rate)
            global_stats['avg_time'].append(avg_time_per_img)
            
            total_samples_processed += count

            print("-" * 50)
            print(f"Dataset Summary: {dataset_name} ({count} images)")
            print(f"  > Avg Time/Image:               {avg_time_per_img:.3f} s")
            print(f"  > Route Shift (Avg):            {avg_shift:.4f} m")
            print(f"  > Speed Diff (Avg):             {avg_spd:.4f} m/s")
            print(f"  > Global Tan (Steering Drift):  {dataset_global_tan:.4f} ({dataset_global_angle:.2f}°)")
            print(f"  > Success Rate:                 {succ_rate:.2f} %")
            print(f"  > SimLingo Drift (Avg):         {avg_drift:.4f}")
            print(f"  > Road-Region LPIPS:            {avg_lpips:.4f}")
            print(f"  > Road-Region SSIM:             {avg_ssim:.4f}")
            print("-" * 50)
        
        gc.collect(); torch.cuda.empty_cache()

    print("\n" + "="*60)
    print("ALL DATASETS PROCESSED. Calculating Global FID...")
    final_fid = 0.0
    try:
        del attacker; gc.collect(); torch.cuda.empty_cache(); torch.set_default_dtype(torch.float32)
        if len(os.listdir(global_fid_clean)) > 0 and calculate_metrics is not None:
            metrics = calculate_metrics(input1=global_fid_clean, input2=global_fid_adv, cuda=True, isc=False, fid=True, verbose=False)
            final_fid = metrics['frechet_inception_distance']
            print(f"  > GLOBAL FID SCORE: {final_fid:.4f}")
    except Exception as e: print(f"  > FID Calculation Failed: {e}")

    total_elapsed_time = time.time() - overall_start_time
    if len(global_stats['shift']) > 0:
        def get_avg(k): return sum(global_stats[k]) / len(global_stats[k])
        
        print("\nMACRO AVERAGE METRICS (Average across all datasets):")
        print(f"  > Total Samples:        {total_samples_processed}")
        print(f"  > Total Elapsed Time:   {timedelta(seconds=int(total_elapsed_time))}")
        print(f"  > Avg Time Per Image:   {get_avg('avg_time'):.3f} s")
        print("-" * 40)
        print(f"  > Avg Global Tan:       {get_avg('global_tan'):.4f}")
        print(f"  > Avg Global Angle:     {get_avg('global_angle'):.2f}°")
        print(f"  > Avg Route Shift:      {get_avg('shift'):.4f} m")
        print(f"  > Avg Speed Diff:       {get_avg('spd_err'):.4f} m/s")
        print(f"  > Avg Success Rate:     {get_avg('success_rate'):.2f} %")
        print(f"  > Avg SimLingo Drift:   {get_avg('sim_drift'):.4f}")
        print(f"  > Avg LPIPS:            {get_avg('road_lpips'):.4f}")
        print(f"  > Avg SSIM:             {get_avg('road_ssim'):.4f}")
        print(f"  > Global FID (All):     {final_fid:.4f}")
        print("="*60)

if __name__ == "__main__":
    main()