import os
# 禁用 Tokenizers 并行，防止死锁
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# 优化显存分配
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import sys
import glob
import json
import gzip
import math
import argparse
import shutil
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

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
from diffusers import StableDiffusionPipeline, DDIMScheduler
from diffusers.models.attention_processor import Attention
from torchvision.utils import save_image
import gc

import time
from datetime import timedelta

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
REPO_ROOT = "/home/pc/simlingo"
CHECKPOINT_PATH = f"{REPO_ROOT}/outputs/simlingo/checkpoints/epoch=013.ckpt/pytorch_model.pt"
CONFIG_PATH = f"{REPO_ROOT}/outputs/simlingo/.hydra/config.yaml"

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# === 灰盒 DiffAttack 参数 (V8 策略迁移 - MFA 风格) ===
ATTACK_PARAMS = {
    "steps": 20,            # SD 采样步数
    "start_step": 1,        # 从第几步开始攻击 (越大越早，不仅能改纹理还能改结构)
    "iterations": 60,       # 优化迭代次数
    "lr": 0.05,             # 学习率
    
    # === 灰盒攻击权重 (MFA 风格) ===
    "w_feature": 3.0,       # [核心] 道路区域特征破坏权重
    "w_cosine": 1.5,        # [核心] 余弦相似度攻击
    "w_attn_focus": 4.5,    # [核心] 注意力聚焦攻击权重
    
    # === 结构约束 ===
    "w_attn": 450.0,       # SD Attention Control (保持原图结构)
    "attn_temperature": 4.5,# 注意力权重温度参数
    "epsilon_latent": 0.05, # 潜空间截断范围
    "sd_model": "Manojb/stable-diffusion-2-base",
    "device": "cuda:0",
}

# ================= 全局变量更新 =================
# CARLA 标准指令映射 (与 openloopv2.py 保持一致)
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
    """根据指令ID获取对应的文本描述"""
    return CMD_MAPPING.get(cmd_id, "Follow Lane")

sys.path.append(REPO_ROOT)
try:
    from simlingo_training.utils.custom_types import DrivingInput, LanguageLabel
    from team_code.simlingo_utils import get_camera_intrinsics, get_camera_extrinsics
except ImportError as e:
    print(f"环境加载失败: {e}")
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

# ================= 特征提取器 (SimLingo Drift 用于评估) =================
class SimLingoFeatureExtractor:
    def __init__(self, simlingo_model):
        self.features = []
        self.hooks = []
        target_module = None
        candidates = []
        for name, module in simlingo_model.named_modules():
            if "vision" in name or "visual" in name:
                if name.endswith(".layers") or name.endswith(".resblocks") or name.endswith(".blocks"):
                    if len(list(module.children())) > 0:
                        try:
                            last_layer = list(module.children())[-1]
                            candidates.append(last_layer)
                        except: continue
        if len(candidates) > 0:
            target_module = candidates[-1]
            self.hooks.append(target_module.register_forward_hook(self.hook_fn))
        else:
            try:
                if hasattr(simlingo_model, 'vision_model'): target_module = simlingo_model.vision_model
                elif hasattr(simlingo_model, 'model'): target_module = simlingo_model.model.vision_model
                if target_module: self.hooks.append(target_module.register_forward_hook(self.hook_fn))
            except: pass

    def hook_fn(self, module, input, output):
        data = None
        if isinstance(output, torch.Tensor): data = output
        elif hasattr(output, 'last_hidden_state'): data = output.last_hidden_state
        elif isinstance(output, tuple): data = output[0]
        if data is not None: self.features.append(data.detach().clone())

    def clear(self): self.features = []
    def remove(self): 
        for h in self.hooks: h.remove()

# ================= UNet 特征提取器 (DiffAttack 结构约束) =================
class UNetFeatureExtractor:
    def __init__(self, unet):
        self.features = []
        self.hooks = []
        targets = [unet.mid_block, unet.up_blocks[1]] 
        for module in targets:
            self.hooks.append(module.register_forward_hook(self.hook_fn))
    def hook_fn(self, m, i, o): self.features.append(o.detach())
    def clear(self): self.features = []
    def remove(self): 
        for h in self.hooks: h.remove()

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

# [关键] 道路加权特征损失
def get_road_focused_feature_loss(feat_adv, feat_clean, road_ratio=0.45):
    if feat_adv.dim() != 3: return F.mse_loss(feat_adv, feat_clean)
    B, N, D = feat_adv.shape
    side = int(math.sqrt(N))
    if side * side == N:
        h_tokens = w_tokens = side
        patch_adv, patch_clean = feat_adv, feat_clean
        cls_adv, cls_clean = None, None
    else:
        side = int(math.sqrt(N - 1))
        if side * side == N - 1:
            h_tokens = w_tokens = side
            cls_adv, patch_adv = feat_adv[:, :1, :], feat_adv[:, 1:, :]
            cls_clean, patch_clean = feat_clean[:, :1, :], feat_clean[:, 1:, :]
        else:
            return F.mse_loss(feat_adv, feat_clean)
            
    patch_adv_2d = patch_adv.view(B, h_tokens, w_tokens, D)
    patch_clean_2d = patch_clean.view(B, h_tokens, w_tokens, D)
    
    weight = torch.ones(h_tokens, device=feat_adv.device)
    road_start = int(h_tokens * (1.0 - road_ratio))
    weight[road_start:] = 3.0 
    weight = weight.view(1, h_tokens, 1, 1)
    
    diff = (patch_adv_2d - patch_clean_2d) ** 2
    weighted_diff = diff * weight
    loss = weighted_diff.mean()
    if cls_adv is not None: loss = loss + F.mse_loss(cls_adv, cls_clean)
    return loss

# [MFA风格] 注意力加权特征损失
def get_attention_weighted_feature_loss(feat_adv, feat_clean, attn_weights, temperature=2.0):
    if attn_weights is None: return F.mse_loss(feat_adv, feat_clean)
    if attn_weights.dim() == 4: attn = attn_weights[:, :, 0, 1:].mean(dim=1)
    elif attn_weights.dim() == 3: attn = attn_weights[:, 0, 1:]
    elif attn_weights.dim() == 2: attn = attn_weights[:, 1:] if attn_weights.size(1) > 1 else attn_weights
    else: return F.mse_loss(feat_adv, feat_clean)
    
    attn = F.softmax(attn * temperature, dim=-1)
    if feat_adv.dim() == 3 and feat_adv.size(1) > attn.size(1):
        cls_adv, patch_adv = feat_adv[:, :1, :], feat_adv[:, 1:, :]
        cls_clean, patch_clean = feat_clean[:, :1, :], feat_clean[:, 1:, :]
        diff = (patch_adv - patch_clean) ** 2
        diff_per_patch = diff.mean(dim=-1)
        weighted_loss = (diff_per_patch * attn).sum(dim=-1).mean()
        cls_loss = F.mse_loss(cls_adv, cls_clean)
        return weighted_loss + cls_loss
    else: return F.mse_loss(feat_adv, feat_clean)

# ================= DiffAttack Controller =================
class DiffAttackController:
    def __init__(self):
        self.loss = 0
        self.reference_maps = {}
        self.mode = "store" 
        self.cur_step = 0
        self.layer_idx = 0
        
    def reset(self):
        self.loss = 0
        self.cur_step = 0
        self.layer_idx = 0
        
    def step(self):
        self.cur_step += 1
        self.layer_idx = 0

    def __call__(self, attn_probs, is_cross, place_in_unet):
        if is_cross: return attn_probs
        key = f"{place_in_unet}_{self.layer_idx}"
        if self.mode == "store":
            self.reference_maps[key] = attn_probs.detach().clone()
        elif self.mode == "loss":
            if key in self.reference_maps:
                ref = self.reference_maps[key]
                if attn_probs.shape == ref.shape:
                    self.loss += F.mse_loss(attn_probs, ref)
        elif self.mode == "replace":
            if key in self.reference_maps:
                ref = self.reference_maps[key]
                if attn_probs.shape == ref.shape:
                    return ref
        self.layer_idx += 1
        return attn_probs

class P2PAttentionProcessor:
    def __init__(self, place_in_unet, is_cross, controller):
        self.place_in_unet = place_in_unet; self.is_cross = is_cross; self.controller = controller
    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None, scale=1.0):
        query = attn.to_q(hidden_states); query = attn.head_to_batch_dim(query)
        if self.is_cross: key = attn.to_k(encoder_hidden_states); value = attn.to_v(encoder_hidden_states)
        else: key = attn.to_k(hidden_states); value = attn.to_v(hidden_states)
        key = attn.head_to_batch_dim(key); value = attn.head_to_batch_dim(value)
        attention_scores = torch.baddbmm(torch.empty(query.shape[0], query.shape[1], key.shape[1], dtype=query.dtype, device=query.device), query, key.transpose(-1, -2), beta=0, alpha=attn.scale)
        attention_probs = attention_scores.softmax(dim=-1)
        attention_probs = self.controller(attention_probs, self.is_cross, self.place_in_unet)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)
        hidden_states = attn.to_out[0](hidden_states); hidden_states = attn.to_out[1](hidden_states)
        return hidden_states

def register_attention_control(unet, controller):
    def get_place(name):
        if "down" in name: return "down"
        elif "mid" in name: return "mid"
        elif "up" in name: return "up"
        return "none"
    for name, module in unet.named_modules():
        if isinstance(module, Attention):
            is_cross = True if "attn2" in name else False 
            place = get_place(name)
            module.set_processor(P2PAttentionProcessor(place, is_cross, controller))

# ================= 主程序 (Grey-Box Version) =================

class DiffAttackerGreyBox:
    def __init__(self):
        self.device = torch.device(ATTACK_PARAMS["device"])
        print(f"Loading Config: {CONFIG_PATH}")
        with open(CONFIG_PATH, 'r') as f: self.cfg = OmegaConf.load(f)
        
        self.lpips_vgg = lpips.LPIPS(net='vgg').to(self.device); self.lpips_vgg.requires_grad_(False)
        self.ssim_loss = SSIM().to(self.device)

        variant = self.cfg.model.vision_model.variant
        self.processor = AutoProcessor.from_pretrained(variant, trust_remote_code=True)
        self.tokenizer = getattr(self.processor, 'tokenizer', self.processor)
        self.tokenizer.padding_side = "left"
        self.tokenizer.add_special_tokens({'additional_special_tokens': ['<WAYPOINTS>', '<TARGET_POINT>']})

        self.tmp_config = AutoConfig.from_pretrained(variant, trust_remote_code=True)
        image_size = self.tmp_config.force_image_size or self.tmp_config.vision_config.image_size
        self.num_image_token = int((image_size // self.tmp_config.vision_config.patch_size) ** 2 * (0.5 ** 2))

        print("Loading SimLingo Model...")
        torch.set_default_dtype(torch.bfloat16)
        cache_dir = f"pretrained/{variant.split('/')[-1]}"
        self.model = hydra.utils.instantiate(self.cfg.model, cfg_data_module=self.cfg.data_module, processor=self.processor, cache_dir=cache_dir, _recursive_=False).to(self.device)
        state_dict = torch.load(CHECKPOINT_PATH, map_location=self.device)
        if 'state_dict' in state_dict: state_dict = state_dict['state_dict']
        new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        self.model.load_state_dict(new_state_dict, strict=False)
        self.model.eval()
        for param in self.model.parameters(): param.requires_grad = False

        # [灰盒关键] 寻找 Vision Model
        print(f">>> [Grey-Box] Hunting for Vision Backbone...")
        self.vision_model = None
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

        self.vision_model = recursive_search(self.model)
        if self.vision_model is None:
            if hasattr(self.model, 'vision_model'): 
                self.vision_model = list(self.model.vision_model.children())[0]
        
        if self.vision_model is None:
             raise ValueError("CRITICAL: Failed to locate underlying Vision Transformer model!")
        print(f">>> [Grey-Box] TARGET LOCKED: {self.vision_model.__class__.__name__}")

        print(f"Loading Stable Diffusion...")
        self.sd_pipe = StableDiffusionPipeline.from_pretrained(ATTACK_PARAMS['sd_model'], torch_dtype=torch.float16).to(self.device)
        self.sd_pipe.scheduler = DDIMScheduler.from_config(self.sd_pipe.scheduler.config)
        self.sd_pipe.scheduler.set_timesteps(ATTACK_PARAMS['steps'])
        self.sd_pipe.safety_checker = None
        self.sd_pipe.enable_attention_slicing()
        self.sd_pipe.vae.requires_grad_(False); self.sd_pipe.unet.requires_grad_(False)
        
        # Init Extractors
        self.simlingo_extractor = SimLingoFeatureExtractor(self.model)
        self.unet_extractor = UNetFeatureExtractor(self.sd_pipe.unet)
        
    def load_measurements(self, image_path):
        """
        加载 Measurements 和 Boxes 数据
        """
        img_dir = os.path.dirname(image_path)
        root_dir = os.path.dirname(img_dir)
        base_name = os.path.splitext(os.path.basename(image_path))[0] # 使用 splitext 兼容 png/jpg
        
        meas_path = os.path.join(root_dir, 'measurements', f"{base_name}.json.gz")
        if not os.path.exists(meas_path):
            meas_path = os.path.join(root_dir, 'measurements', f"{base_name}.json")
            
        boxes_path = os.path.join(root_dir, 'boxes', f"{base_name}.json.gz")
        if not os.path.exists(boxes_path):
            boxes_path = os.path.join(root_dir, 'boxes', f"{base_name}.json")

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
                    idx1 = np.where(dists > 10.0)[0]; tp1 = gt_local[idx1[0]] if len(idx1) > 0 else gt_local[-1]
                    idx2 = np.where(dists > 20.0)[0]; tp2 = gt_local[idx2[0]] if len(idx2) > 0 else gt_local[-1]
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
                                if raw_tl != "None": traffic_light_state = raw_tl; break
            except Exception as e: print(f"Error loading boxes {boxes_path}: {e}")

        return speed, cmd_id, traffic_light_state, tp1, tp2, gt_local
    
    def prepare_base_components(self, image_path):
        real_speed, cmd_id, tl_state, tp1, tp2, gt_route = self.load_measurements(image_path)
        cmd_text = get_cmd_text(cmd_id)
        
        # [修改] 读取图片并保存原始尺寸，这里不进行resize，放在 run_diff_attack 里做
        raw_image = Image.open(image_path).convert('RGB')
        orig_W, orig_H = raw_image.size
        
        image_sizes = torch.tensor([[448, 448]]).to(self.device) 
        
        prompt_tp = "Target waypoint: <TARGET_POINT><TARGET_POINT>."
        base_prompt = (
            f"Current speed: {real_speed:.2f} m/s. "
            f"Command: {cmd_text}. "
            f"Traffic light: {tl_state}. "
            f"{prompt_tp} Predict the waypoints."
        )
        
        image_tokens = '<img>' + '<IMG_CONTEXT>' * self.num_image_token * 2 + '</img>'
        final_prompt = image_tokens + "\n" + base_prompt
        
        tp_token_id = self.tokenizer.convert_tokens_to_ids('<TARGET_POINT>')
        target_points_np = np.stack([tp1, tp2])
        placeholder_dict = {tp_token_id: target_points_np}
        
        tokens = self.tokenizer([final_prompt], padding=True, return_tensors="pt")
        ll = LanguageLabel(
            phrase_ids=tokens["input_ids"].to(self.device), 
            phrase_valid=tokens["attention_mask"].bool().to(self.device),
            phrase_mask=tokens["attention_mask"].bool().to(self.device), 
            placeholder_values=[placeholder_dict], 
            language_string=[final_prompt], 
            loss_masking=None
        )
        
        intrinsics = get_camera_intrinsics(448, 448, fov=110).unsqueeze(0).to(self.device).float().view(1, 3, 3)
        extrinsics = get_camera_extrinsics().unsqueeze(0).to(self.device).float().view(1, 4, 4)
        tp_tensor = torch.from_numpy(tp1).bfloat16().unsqueeze(0).to(self.device)
        
        return {
            "image_sizes": image_sizes, 
            "camera_intrinsics": intrinsics, 
            "camera_extrinsics": extrinsics, 
            "vehicle_speed": torch.tensor([[real_speed]]).to(self.device).bfloat16(), 
            "target_point": tp_tensor, 
            "prompt": ll, 
            "prompt_inference": ll
        }, raw_image, gt_route, tp1, (orig_W, orig_H)

    def differentiable_processing(self, sd_output_01):
        # sd_output_01: [B, 3, 512, 512] (Center Crop)
        # 强制 Resize 到 448x448 (InternVL 标准输入)
        # 这里的输入已经是 Center Crop，所以直接 resize 不会破坏几何结构
        img_448 = F.interpolate(sd_output_01, size=(448, 448), mode='bilinear', align_corners=False)
        mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
        return ((img_448 - mean) / std).unsqueeze(1).unsqueeze(1).repeat(1, 1, 2, 1, 1, 1).bfloat16()
    
    def get_vision_features_grad(self, img_tensor_448_normed, return_attn=False):
        pixel_values = img_tensor_448_normed[:, 0, 0, :, :, :] 
        attn_weights = None
        try:
            outputs = self.vision_model(pixel_values=pixel_values, output_attentions=return_attn)
        except TypeError:
            try:
                outputs = self.vision_model(pixel_values, output_attentions=return_attn)
            except TypeError:
                try:
                    outputs = self.vision_model(pixel_values=pixel_values)
                except:
                    outputs = self.vision_model(pixel_values)
        
        if hasattr(outputs, 'last_hidden_state'): features = outputs.last_hidden_state
        elif hasattr(outputs, 'pooler_output'): features = outputs.pooler_output
        elif isinstance(outputs, tuple): features = outputs[0]
        else: features = outputs
        
        if return_attn:
            if hasattr(outputs, 'attentions') and outputs.attentions is not None:
                attn_weights = outputs.attentions[-1]
            elif isinstance(outputs, tuple) and len(outputs) > 1:
                for item in outputs[1:]:
                    if isinstance(item, (list, tuple)) and len(item) > 0:
                        if isinstance(item[-1], torch.Tensor) and item[-1].dim() == 4:
                            attn_weights = item[-1]; break
            return features, attn_weights
        return features

    def get_text_embeddings(self):
        return self.sd_pipe.encode_prompt(prompt="", device=self.device, num_images_per_prompt=1, do_classifier_free_guidance=False, negative_prompt=None)[0]

    def ddim_inversion(self, latents, start_step_idx):
        if not hasattr(self, 'null_embs'): 
            self.null_embs = self.get_text_embeddings().detach()
        all_timesteps = self.sd_pipe.scheduler.timesteps
        start_index = len(all_timesteps) - 1
        stop_index = len(all_timesteps) - 1 - start_step_idx
        curr_latents = latents.clone()

        # print(f"   [DDIM Inversion] Inverting from step {start_index} back to {stop_index}...")

        with torch.no_grad():
            for i in range(start_index, stop_index, -1):
                t = all_timesteps[i]
                t_next = all_timesteps[i-1]
                noise_pred = self.sd_pipe.unet(curr_latents, t, encoder_hidden_states=self.null_embs).sample
                alpha_prod_t = self.sd_pipe.scheduler.alphas_cumprod[t]
                alpha_prod_t_next = self.sd_pipe.scheduler.alphas_cumprod[t_next]
                beta_prod_t = 1 - alpha_prod_t
                pred_original_sample = (curr_latents - beta_prod_t ** (0.5) * noise_pred) / alpha_prod_t ** (0.5)
                beta_prod_t_next = 1 - alpha_prod_t_next
                curr_latents = alpha_prod_t_next ** (0.5) * pred_original_sample + beta_prod_t_next ** (0.5) * noise_pred
        return curr_latents

    def run_diff_attack(self, image_path, output_dir, fid_clean_dir, fid_adv_dir, file_prefix=""):
        base_params, raw_pil, gt_route, target_point, (orig_W, orig_H) = self.prepare_base_components(image_path)
        
        # === [Modified for Center Crop] ===
        # 计算中心裁剪区域
        crop_size = min(orig_W, orig_H)
        left = (orig_W - crop_size) // 2
        top = (orig_H - crop_size) // 2
        right = left + crop_size
        bottom = top + crop_size
        
        # 1. Crop Center
        img_cropped = raw_pil.crop((left, top, right, bottom))
        
        # 2. Resize to 512x512 for Stable Diffusion
        # SD 必须使用 512x512，现在输入的是正方形，比例正常
        img_512 = img_cropped.resize((512, 512), Image.BILINEAR)
        img_tensor = (torch.from_numpy(np.array(img_512)).float() / 127.5 - 1.0)
        img_tensor_sd = img_tensor.permute(2, 0, 1).unsqueeze(0).to(self.device).half()
        
        if not hasattr(self, 'null_embs'): self.null_embs = self.get_text_embeddings().detach()
        
        with torch.no_grad():
            latents_orig = self.sd_pipe.vae.encode(img_tensor_sd).latent_dist.sample() * 0.18215
            recon_clean_img_01 = (self.sd_pipe.vae.decode(latents_orig / 0.18215).sample / 2 + 0.5).clamp(0, 1)

            # 1. 提取 Clean ViT 特征 (Target) - 注意 differentiable_processing 会将其 resize 到 448
            clean_inputs = self.differentiable_processing(recon_clean_img_01.float())
            clean_feat_raw, clean_attn = self.get_vision_features_grad(clean_inputs, return_attn=True)
            clean_vision_emb = clean_feat_raw.detach() 
            
            # 2. 完整 SimLingo 推理 (For Reference Metrics)
            # SimLingo 也是基于 Center Crop 进行推理
            self.simlingo_extractor.clear()
            clean_full_input = DrivingInput(camera_images=clean_inputs, **base_params)
            speed_clean, route_clean, _ = self.model(clean_full_input)
            speed_clean, route_clean = speed_clean.float(), route_clean.float()
            clean_sim_feats = [f.clone() for f in self.simlingo_extractor.features]
            
            # 3. Clean UNet Inversion
            t_idx = ATTACK_PARAMS["start_step"] 
            initial_noisy_latents = self.ddim_inversion(latents_orig, t_idx)
            target_t_index = len(self.sd_pipe.scheduler.timesteps) - t_idx - 1
            t = self.sd_pipe.scheduler.timesteps[target_t_index]
            
            self.unet_extractor.clear()
            _ = self.sd_pipe.unet(initial_noisy_latents.half(), t, encoder_hidden_states=self.null_embs)
            clean_unet_feats = [f.clone() for f in self.unet_extractor.features]
            
        # === 初始化 Attention Controller ===
        controller = DiffAttackController()
        register_attention_control(self.sd_pipe.unet, controller)
        
        controller.mode = "store"; controller.reset()
        with torch.no_grad(): 
            _ = self.sd_pipe.unet(initial_noisy_latents.half(), t, encoder_hidden_states=self.null_embs)
            
        controller.mode = "loss"
        noisy_latents = initial_noisy_latents.detach().float().clone().requires_grad_(True)
        optimizer = torch.optim.AdamW([noisy_latents], lr=ATTACK_PARAMS['lr'])

        # === 攻击循环 ===
        for i in range(ATTACK_PARAMS['iterations']):
            optimizer.zero_grad()
            l_in = noisy_latents.to(self.sd_pipe.unet.dtype) 
            
            controller.reset() 
            noise_pred = self.sd_pipe.unet(l_in, t, encoder_hidden_states=self.null_embs).sample
            loss_attn = controller.loss

            alpha_prod_t = self.sd_pipe.scheduler.alphas_cumprod[t]
            beta_prod_t = 1 - alpha_prod_t
            pred_latents_z0 = (l_in - beta_prod_t ** 0.5 * noise_pred) / alpha_prod_t ** 0.5
            
            vae_input = (pred_latents_z0 / 0.18215).to(self.sd_pipe.vae.dtype)
            decoded_img = self.sd_pipe.vae.decode(vae_input).sample
            adv_img_01 = (decoded_img / 2 + 0.5).clamp(0, 1)
            
            # 这里 differentiable_processing 会处理 512(SD) -> 448(SimLingo) 的转换
            adv_inputs = self.differentiable_processing(adv_img_01.float())
            adv_vision_emb = self.get_vision_features_grad(adv_inputs)
            
            loss_feature_dist = get_road_focused_feature_loss(adv_vision_emb, clean_vision_emb)
            
            adv_flat = adv_vision_emb.view(adv_vision_emb.size(0), -1)
            clean_flat = clean_vision_emb.view(clean_vision_emb.size(0), -1)
            cos_sim = F.cosine_similarity(adv_flat, clean_flat, dim=-1).mean()
            loss_cosine = cos_sim
            
            loss_attn_focus = get_attention_weighted_feature_loss(
                adv_vision_emb, clean_vision_emb, clean_attn,
                temperature=ATTACK_PARAMS['attn_temperature']
            )

            attack_loss = -ATTACK_PARAMS['w_feature'] * loss_feature_dist \
                        - ATTACK_PARAMS['w_cosine'] * loss_cosine \
                        - ATTACK_PARAMS['w_attn_focus'] * loss_attn_focus
            
            total_loss = attack_loss + ATTACK_PARAMS['w_attn'] * loss_attn

            if torch.isnan(total_loss): break
            total_loss.backward()
            optimizer.step()

            with torch.no_grad():
                eps = ATTACK_PARAMS['epsilon_latent']
                delta = noisy_latents - initial_noisy_latents
                noisy_latents.copy_(initial_noisy_latents + torch.clamp(delta, -eps, eps))
                
                cur_adv_input = adv_img_01.detach().float() * 2 - 1
                cur_clean_input = recon_clean_img_01.detach().float() * 2 - 1
                cur_lpips = self.lpips_vgg(cur_adv_input, cur_clean_input).mean().item()
                cur_ssim = self.ssim_loss(cur_adv_input, cur_clean_input).item()
                
                print(f"\rIter [{i+1:02d}/{ATTACK_PARAMS['iterations']}] "
                      f"Loss: {total_loss.item():.4f} | "
                      f"LPIPS: {cur_lpips:.4f} | "
                      f"SSIM: {cur_ssim:.4f}", end="", flush=True)
                if i == ATTACK_PARAMS['iterations'] - 1: print()

        # === 最终生成与指标计算 ===
        with torch.no_grad():
            self.unet_extractor.clear()
            latents_for_gen = noisy_latents.detach().clone().to(self.sd_pipe.unet.dtype)
            
            # UNet Feats
            controller.mode = "loss"; controller.reset()
            _ = self.sd_pipe.unet(latents_for_gen, t, encoder_hidden_states=self.null_embs)
            adv_unet_feats = [f.clone() for f in self.unet_extractor.features]
            self.unet_extractor.clear()
            
            # Resume Denoising with Attention Replacement
            if not hasattr(self, 'null_embs'): self.null_embs = self.get_text_embeddings().detach()
            t_idx = ATTACK_PARAMS["start_step"]
            start_index = len(self.sd_pipe.scheduler.timesteps) - t_idx - 1
            
            controller.mode = "replace"; controller.reset()
            for i in range(start_index, len(self.sd_pipe.scheduler.timesteps)):
                t_curr = self.sd_pipe.scheduler.timesteps[i]
                controller.layer_idx = 0 
                noise_pred = self.sd_pipe.unet(latents_for_gen, t_curr, encoder_hidden_states=self.null_embs).sample
                latents_for_gen = self.sd_pipe.scheduler.step(noise_pred, t_curr, latents_for_gen).prev_sample
                controller.step()

            # VAE Decode (512x512) - 这是攻击后的 Crop
            final_img = self.sd_pipe.vae.decode(latents_for_gen.to(self.sd_pipe.vae.dtype) / 0.18215).sample
            final_img_01_512 = (final_img / 2 + 0.5).clamp(0, 1)

            # ================= [Modified for Center Crop] 贴回逻辑 =================
            # 1. 准备原始背景 (Full Canvas)
            raw_tensor = (torch.from_numpy(np.array(raw_pil)).float() / 127.5 - 1.0).permute(2, 0, 1).unsqueeze(0).to(self.device)
            final_full_adv = raw_tensor.clone()
            final_full_clean = raw_tensor.clone()
            
            # 2. 放大 Crop (512 -> crop_size)
            final_crop_upscaled = F.interpolate(final_img, size=(crop_size, crop_size), mode='bilinear', align_corners=False)
            recon_crop_upscaled = F.interpolate(self.sd_pipe.vae.decode(latents_orig / 0.18215).sample, size=(crop_size, crop_size), mode='bilinear', align_corners=False)
            
            # 3. 贴回中间位置
            final_full_adv[:, :, top:bottom, left:right] = final_crop_upscaled
            final_full_clean[:, :, top:bottom, left:right] = recon_crop_upscaled
            
            # 4. 归一化到 [0, 1] 用于保存和显示
            final_full_adv_01 = (final_full_adv / 2 + 0.5).clamp(0, 1)
            final_full_clean_01 = (final_full_clean / 2 + 0.5).clamp(0, 1)
            # ====================================================================
            
            # [评估] SimLingo 需要 448x448
            # 为了与 MFA 逻辑保持一致，SimLingo 的评估基于 Center Crop (即 final_img)
            # 这样 SimLingo 看到的是核心区域，且比例正确
            self.simlingo_extractor.clear()
            final_input = DrivingInput(camera_images=self.differentiable_processing(final_img_01_512.float()), **base_params)
            speed_final, route_final, _ = self.model(final_input)
            adv_sim_feats = [f.clone() for f in self.simlingo_extractor.features]
            
            unet_dist = 0.0
            if len(adv_unet_feats) == len(clean_unet_feats):
                for fa, fc in zip(adv_unet_feats, clean_unet_feats):
                    unet_dist += F.mse_loss(fa.float(), fc.float()).item()
                unet_dist /= len(adv_unet_feats)
            
            sim_drift = 0.0
            if len(adv_sim_feats) > 0:
                sim_drift = F.mse_loss(adv_sim_feats[-1].float(), clean_sim_feats[-1].float()).item() * 1000

            clean_end = route_clean[0, -1] 
            adv_end = route_final[0, -1]
            f_dist = torch.norm(route_final.float() - route_clean, p=2).item()
            f_spd_err = torch.norm(speed_final.float() - speed_clean, p=2).item()
            lat_error = abs(adv_end[1].item() - clean_end[1].item())
            lon_error = abs(adv_end[0].item() - clean_end[0].item())
            clean_route_len = abs(clean_end[0].item())
            is_success = 1.0 if f_dist >= 1.0 else 0.0

            # 内部指标 (LPIPS/SSIM) 使用 512 版本计算 (即评估 Crop 区域的质量)
            road_lpips = self.lpips_vgg(final_img_01_512.float()*2-1, recon_clean_img_01.float()*2-1).mean().item()
            road_ssim = self.ssim_loss(final_img_01_512.float()*2-1, recon_clean_img_01.float()*2-1).item()

            # ================= 保存为 PNG =================
            base_name = os.path.basename(image_path).split('.')[0]
            save_name = f"{file_prefix}_{base_name}.png" if file_prefix else f"{base_name}.png"

            def save_safe_png(tensor, path):
                # tensor: [0, 1] -> [0, 255]
                ndarr = tensor.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
                Image.fromarray(ndarr).save(path, format='PNG')

            # 保存原始尺寸的 Clean 重建图 (用于 FID)
            save_safe_png(final_full_clean_01[0].cpu().float(), os.path.join(fid_clean_dir, save_name))
            
            # 保存原始尺寸的 Adversarial (用于 FID 和 迁移攻击)
            save_safe_png(final_full_adv_01[0].cpu().float(), os.path.join(fid_adv_dir, save_name))
            
            # 保存到数据集输出目录 (TransFuser 将读取此目录)
            save_safe_png(final_full_adv_01[0].cpu().float(), os.path.join(output_dir, save_name))
            # =========================================================
            
            # 可视化使用 Crop 区域以便观察细节
            self.save_unified_visualization(recon_clean_img_01, final_img_01_512, route_clean, route_final, gt_route, target_point, speed_clean, speed_final, output_dir, base_name)

        return f_dist, f_spd_err, sim_drift, road_lpips, road_ssim, lat_error, lon_error, clean_route_len, is_success
    
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
            ax2.set_title("2. Adv (Center Crop)", fontsize=14, fontweight='bold')
            ax2.axis('off')
            
            ax3 = fig.add_subplot(gs[1, 0])
            ax3.imshow(diff_vis)
            ax3.set_title("3. Noise (15x)", fontsize=14, fontweight='bold')
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

            ax4.set_xlim(-12, 12)
            ax4.set_ylim(-2, 40)
            ax4.set_xlabel("Lateral (m)")
            ax4.set_ylabel("Forward (m)")
            ax4.grid(True, linestyle=':', alpha=0.6)
            ax4.legend(loc='upper right')
            ax4.set_title("4. Trajectory (Feature Attack)", fontsize=14, fontweight='bold')

            plt.tight_layout()
            save_path = os.path.join(save_dir, f"{img_name}_vis.png")
            plt.savefig(save_path, dpi=120, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"Vis error: {e}")

def get_next_log_filename(output_dir):
    existing_logs = [f for f in os.listdir(output_dir) if f.startswith("attack_log_") and f.endswith(".txt")]
    if not existing_logs:
        return os.path.join(output_dir, "attack_log_1.txt")
    
    max_index = max(int(f.split("_")[2].split(".")[0]) for f in existing_logs)
    return os.path.join(output_dir, f"attack_log_{max_index + 1}.txt")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True, 
                        help="Root directory containing multiple dataset folders")
    parser.add_argument("--output_dir", type=str, default="da_allresults") 
    args = parser.parse_args()

    # 1. 确定本次运行的序号文件夹 (DA-gray-N)
    base_dir = args.output_dir
    os.makedirs(base_dir, exist_ok=True)

    existing_folders = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    max_id = 0
    for folder in existing_folders:
        if folder.startswith("DA-gray-"):
            try:
                idx = int(folder.split("-")[-1])
                if idx > max_id: max_id = idx
            except ValueError: continue
    
    new_folder_name = f"DA-gray-{max_id + 1}"
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
        f.write("=== Attack Parameters ===\n")
        json.dump(ATTACK_PARAMS, f, indent=4)
        f.write("\n")

    # 4. 重定向日志输出
    sys.stdout = Logger(log_path)
    
    print(f"Results will be saved to: {current_run_dir}")
    print(f"Config saved to: {config_path}")
    print("-" * 60)
    print("Grey-Box DiffAttack Config (Transfer Mode):")
    print(json.dumps(ATTACK_PARAMS, indent=2))
    
    attacker = DiffAttackerGreyBox()
    
    # 5. 设置 FID 文件夹
    global_fid_clean = os.path.join(current_run_dir, "global_fid_clean")
    global_fid_adv = os.path.join(current_run_dir, "global_fid_adv")
    if os.path.exists(global_fid_clean): shutil.rmtree(global_fid_clean)
    if os.path.exists(global_fid_adv): shutil.rmtree(global_fid_adv)
    os.makedirs(global_fid_clean); os.makedirs(global_fid_adv)

    # 扫描数据集
    if not os.path.exists(args.data_root):
        print(f"Error: Data root {args.data_root} does not exist."); sys.exit(1)

    # 修改：直接处理 data_root 下的子文件夹，或者如果 data_root 本身就是包含 rgb 的文件夹
    # 这里的逻辑假设 data_root 下面有多个场景子文件夹 (如 town05_tiny, town01_val 等)
    # 如果 test_complex 下直接就是 rgb 文件夹，则需要调整
    
    # 检测 data_root 是否直接包含 rgb
    if os.path.exists(os.path.join(args.data_root, "rgb")):
        # 如果 args.data_root 本身就是一个场景文件夹
        valid_datasets = [os.path.basename(args.data_root)]
        # 此时修正后续循环中的路径逻辑，需要临时把 args.data_root 视为父目录
        actual_root = os.path.dirname(args.data_root)
        valid_datasets = [os.path.basename(args.data_root)]
        scan_root = actual_root
    else:
        # 否则假设 data_root 下包含多个场景文件夹
        scan_root = args.data_root
        all_subdirs = sorted([d for d in os.listdir(scan_root) if os.path.isdir(os.path.join(scan_root, d))])
        valid_datasets = []
        for d in all_subdirs:
            if os.path.exists(os.path.join(scan_root, d, "rgb")):
                valid_datasets.append(d)
    
    if not valid_datasets:
        print(f"No valid datasets found in {args.data_root}"); sys.exit(1)
        
    print(f"\nFound {len(valid_datasets)} Datasets: {valid_datasets}\n")
    overall_start_time = time.time()
    total_samples_processed = 0
    
    # 全局统计容器
    global_stats = {
        'shift': [], 'spd_err': [], 'sim_drift': [], 
        'road_lpips': [], 'road_ssim': [], 
        'global_tan': [], 'global_angle': [], 
        'success_rate': [], 'avg_time': []
    }

    # === 循环数据集 ===
    for dataset_idx, dataset_name in enumerate(valid_datasets):
        print(f"\n>>> Processing Dataset [{dataset_idx+1}/{len(valid_datasets)}]: {dataset_name}")
        
        current_rgb_dir = os.path.join(scan_root, dataset_name, "rgb")
        current_out_dir = os.path.join(current_run_dir, dataset_name)
        os.makedirs(current_out_dir, exist_ok=True)
        
        # 兼容 JPG/PNG 输入
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
                # file_prefix 传空字符串，因为 MFA 代码中也没用 prefix
                res = attacker.run_diff_attack(img_path, current_out_dir, global_fid_clean, global_fid_adv, file_prefix=dataset_name)
                step_time = time.time() - t0
                
                lat_err = res[5]
                lon_err = res[6]
                clean_len = res[7]
                is_succ = res[8]
                
                current_tan = lat_err / (clean_len + 1e-6)
                
                ds_totals['shift'] += res[0]
                ds_totals['spd_err'] += res[1]
                ds_totals['sim_drift'] += res[2]
                ds_totals['road_lpips'] += res[3]
                ds_totals['road_ssim'] += res[4]
                ds_totals['sum_lat_offset'] += lat_err 
                ds_totals['sum_clean_len'] += clean_len 
                ds_totals['success_count'] += is_succ
                ds_totals['total_time'] += step_time
                
                count += 1
                print(f"[{dataset_name}][{os.path.basename(img_path)}] Shift: {res[0]:.2f}m (Lat:{lat_err:.2f}, Lon:{lon_err:.2f}) | Spd Diff: {res[1]:.2f} | Tan: {current_tan:.2f} | Time: {step_time:.2f}s")

            except KeyboardInterrupt: sys.exit(1)
            except Exception as e: 
                print(f"Error: {e}")
                import traceback; traceback.print_exc()
            finally: torch.cuda.empty_cache()

        # === 数据集结算 ===
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

    # === 全局 FID 计算 ===
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

    # === 宏观平均报告 ===
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