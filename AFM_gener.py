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
import torch.optim as optim
import numpy as np
from PIL import Image
import hydra
from omegaconf import OmegaConf
from transformers import AutoProcessor, AutoConfig
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from diffusers.models import AutoencoderKL
from torchvision.utils import save_image
from tqdm import tqdm

import time
from datetime import timedelta

# === 引入依赖 ===
try:
    from sit import SiT_models
except ImportError:
    print("【警告】找不到 sit.py，请确保 SiT 模型代码在路径中")

try:
    import lpips
except ImportError:
    print("请先安装 lpips: pip install lpips")
    sys.exit(1)

try:
    from torch_fidelity import calculate_metrics
except ImportError:
    calculate_metrics = None

# === 全局常量 ===
TARGET_CLASS_IDX = 817 

# === SSIM 模块 ===
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
    mu1_sq, mu2_sq, mu1_mu2 = mu1.pow(2), mu2.pow(2), mu1*mu2
    sigma1_sq = F.conv2d(img1*img1, window, padding=window_size//2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding=window_size//2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding=window_size//2, groups=channel) - mu1_mu2
    C1, C2 = 0.01**2, 0.03**2
    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean() if size_average else ssim_map.mean(1).mean(1).mean(1)

class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIM, self).__init__()
        self.window_size = window_size; self.channel = 3; self.window = create_window(window_size, self.channel)
        self.size_average = size_average
    def forward(self, img1, img2):
        img1 = (img1 + 1) / 2.0; img2 = (img2 + 1) / 2.0 
        self.window = self.window.to(img1.device).type_as(img1)
        return _ssim(img1, img2, self.window, self.window_size, self.channel, self.size_average)

# === 配置路径 ===
REPO_ROOT = "/home/pc/simlingo"  
CHECKPOINT_PATH = f"{REPO_ROOT}/outputs/simlingo/checkpoints/epoch=013.ckpt/pytorch_model.pt"
CONFIG_PATH = f"{REPO_ROOT}/outputs/simlingo/.hydra/config.yaml"
sys.path.append(REPO_ROOT)

try:
    from simlingo_training.utils.custom_types import DrivingInput, LanguageLabel
    from team_code.simlingo_utils import get_camera_intrinsics, get_camera_extrinsics
except ImportError as e:
    print(f"环境加载失败: {e}"); sys.exit(1)

# === 灰盒攻击参数 ===
ATTACK_PARAMS = {
    "iterations": 50,           # 减少迭代次数
    "lr_z": 0.05,                # 提高学习率0.05
    "lr_u": 0.05,              
    "t_limit": 0.05,             # 保持时间范围
    "sample_steps": 1,           # 保持单步采样
    "active_steps": [0, 1],      
    "epsilon_latent": 0.03,      # 保持微小扰动
    "epsilon_u": 0.03,          
    "w_feature": 3.0,            # 提高特征攻击权重
    "w_cosine": 1.5,             # 提高余弦相似度攻击
    "w_attn_focus": 4.5,         # 保持注意力聚焦权重
    "w_anchor": 6.0,           # 适度锚点约束
    "attn_temperature": 4.5,     # 提高温度，更聚焦关键区域
    "sit_model": "SiT-XL/2",
    "sit_ckpt": "sit_xl_2_meanflow_ema-002.pt",
    "device": "cuda:1", 
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

class Logger(object):
    def __init__(self, filename="default.log"):
        self.terminal = sys.stdout; self.log = open(filename, "a")
    def write(self, message): self.terminal.write(message); self.log.write(message); self.log.flush() 
    def flush(self): pass

def world_to_ego(route_global, ego_x, ego_y, ego_theta_rad):
    route_global = np.asarray(route_global, dtype=float); yaw = float(ego_theta_rad)
    diff = route_global - np.array([float(ego_x), float(ego_y)])
    dx, dy = diff[:, 0], diff[:, 1]
    c, s = np.cos(yaw), np.sin(yaw)
    return np.stack([dx * c + dy * s, -dx * s + dy * c], axis=-1)

# ... (Helper functions: get_sky_mask, get_road_focused_feature_loss, etc. 保持不变) ...
def get_sky_mask(img_tensor, sky_ratio=0.35):
    """生成天空区域掩码 (上部 sky_ratio 区域)"""
    bs, c, h, w = img_tensor.shape
    mask = torch.zeros((bs, 1, h, w), device=img_tensor.device)
    sky_h = int(h * sky_ratio)
    mask[:, :, :sky_h, :] = 1.0
    return mask

def get_latent_sky_mask(latent_tensor, sky_ratio=0.35):
    """生成潜空间的天空区域掩码 (latent是32x32时对应256x256图像)"""
    bs, c, h, w = latent_tensor.shape
    mask = torch.zeros_like(latent_tensor)
    sky_h = int(h * sky_ratio)
    mask[:, :, :sky_h, :] = 1.0
    return mask

def get_latent_road_mask(latent_tensor, road_ratio=0.4):
    """生成潜空间的道路区域掩码 (下部 road_ratio 区域)"""
    bs, c, h, w = latent_tensor.shape
    mask = torch.zeros_like(latent_tensor)
    road_start = int(h * (1.0 - road_ratio))
    mask[:, :, road_start:, :] = 1.0
    return mask

def get_road_mask(img_tensor, road_ratio=0.4):
    """生成道路区域掩码 (下部 road_ratio 区域)"""
    bs, c, h, w = img_tensor.shape
    mask = torch.zeros((bs, 1, h, w), device=img_tensor.device)
    road_start = int(h * (1.0 - road_ratio))
    mask[:, :, road_start:, :] = 1.0
    return mask

def get_sky_preserve_loss(img_adv, img_clean, sky_ratio=0.35):
    """计算天空区域的保护损失 (强制天空区域不变)"""
    sky_mask = get_sky_mask(img_adv, sky_ratio)
    diff = (img_adv - img_clean) * sky_mask
    return (diff ** 2).mean()

def get_road_focused_feature_loss(feat_adv, feat_clean, road_ratio=0.45):
    """
    针对 ViT patch tokens 的道路区域加权攻击 (自动计算空间尺寸)
    feat_adv: [B, N, D]
    """
    if feat_adv.dim() != 3:
        return F.mse_loss(feat_adv, feat_clean)
    
    B, N, D = feat_adv.shape
    
    # 判断是否有 CLS token
    # 通常 ViT 的 N = H*W + 1 (CLS) 或 N = H*W (No CLS)
    # 我们假设是正方形输入
    num_patches = N
    has_cls = False
    
    # 尝试开根号看是否是整数
    side = int(math.sqrt(N))
    if side * side == N:
        h_tokens = w_tokens = side
        patch_adv = feat_adv
        patch_clean = feat_clean
        cls_adv, cls_clean = None, None
    else:
        # 假设有个 CLS token
        side = int(math.sqrt(N - 1))
        if side * side == N - 1:
            h_tokens = w_tokens = side
            has_cls = True
            cls_adv, patch_adv = feat_adv[:, :1, :], feat_adv[:, 1:, :]
            cls_clean, patch_clean = feat_clean[:, :1, :], feat_clean[:, 1:, :]
        else:
            # 无法还原空间结构，回退到普通 MSE
            return F.mse_loss(feat_adv, feat_clean)
            
    # 重塑为空间形态 [B, H, W, D]
    patch_adv_2d = patch_adv.view(B, h_tokens, w_tokens, D)
    patch_clean_2d = patch_clean.view(B, h_tokens, w_tokens, D)
    
    # 创建权重: 道路区域(下部)权重更高
    weight = torch.ones(h_tokens, device=feat_adv.device)
    road_start = int(h_tokens * (1.0 - road_ratio))
    weight[road_start:] = 3.0  # [建议] 进一步加大道路区域的权重 (2.0 -> 3.0)
    weight = weight.view(1, h_tokens, 1, 1)
    
    # 加权 MSE
    diff = (patch_adv_2d - patch_clean_2d) ** 2
    weighted_diff = diff * weight
    
    loss = weighted_diff.mean()
    
    if cls_adv is not None:
        loss = loss + F.mse_loss(cls_adv, cls_clean)
    
    return loss

def get_attention_weighted_feature_loss(feat_adv, feat_clean, attn_weights, temperature=2.0):
    """
    使用注意力权重加权的特征损失
    attn_weights: [B, N] 或 [B, H, N, N] - CLS token 对各 patch 的注意力
    """
    if attn_weights is None:
        return F.mse_loss(feat_adv, feat_clean)
    
    # 处理不同形状的注意力
    if attn_weights.dim() == 4:
        # [B, H, N, N] -> 取 CLS (第0个token) 对所有 patch 的注意力，平均所有 head
        attn = attn_weights[:, :, 0, 1:].mean(dim=1)  # [B, N-1]
    elif attn_weights.dim() == 3:
        attn = attn_weights[:, 0, 1:]  # [B, N-1]
    elif attn_weights.dim() == 2:
        attn = attn_weights[:, 1:] if attn_weights.size(1) > 1 else attn_weights
    else:
        return F.mse_loss(feat_adv, feat_clean)
    
    # 温度缩放，增强高注意力区域
    attn = F.softmax(attn * temperature, dim=-1)  # [B, N-1]
    
    # 分离 CLS 和 patch tokens
    if feat_adv.dim() == 3 and feat_adv.size(1) > attn.size(1):
        cls_adv, patch_adv = feat_adv[:, :1, :], feat_adv[:, 1:, :]
        cls_clean, patch_clean = feat_clean[:, :1, :], feat_clean[:, 1:, :]
        
        # Patch-wise 加权损失
        diff = (patch_adv - patch_clean) ** 2  # [B, N-1, D]
        diff_per_patch = diff.mean(dim=-1)  # [B, N-1]
        weighted_loss = (diff_per_patch * attn).sum(dim=-1).mean()
        
        # CLS token 损失
        cls_loss = F.mse_loss(cls_adv, cls_clean)
        
        return weighted_loss + cls_loss
    else:
        return F.mse_loss(feat_adv, feat_clean)

# === 特征提取器 (用于 Sim-Drift 指标 - 全模型) ===
class SimLingoFeatureExtractor:
    def __init__(self, simlingo_model):
        self.features = []
        self.hooks = []
        target_module = None
        candidates = []
        # 宽泛搜索用于显示的层
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
                # Fallback mechanism
                if hasattr(simlingo_model, 'vision_model'): target_module = simlingo_model.vision_model
                elif hasattr(simlingo_model, 'model'): target_module = simlingo_model.model.vision_model
                if target_module: 
                    self.hooks.append(target_module.register_forward_hook(self.hook_fn))
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

# ============================
# 主攻击类 (Grey-Box SiT Version V3)
# ============================

class MFAttackerSiT:
    def __init__(self, args):
        self.device = torch.device(ATTACK_PARAMS["device"])
        self.args = args
        
        print(">>> Loading SimLingo Config...")
        with open(CONFIG_PATH, 'r') as f: self.cfg = OmegaConf.load(f)
        variant = self.cfg.model.vision_model.variant
        self.processor = AutoProcessor.from_pretrained(variant, trust_remote_code=True)
        self.tokenizer = getattr(self.processor, 'tokenizer', self.processor)
        self.tokenizer.padding_side = "left"
        self.tokenizer.add_special_tokens({'additional_special_tokens': ['<WAYPOINTS>', '<TARGET_POINT>']})
        
        self.tmp_config = AutoConfig.from_pretrained(variant, trust_remote_code=True)
        self.num_image_token = int((self.tmp_config.vision_config.image_size // self.tmp_config.vision_config.patch_size) ** 2 * (0.5 ** 2))
        
        cache_dir = f"pretrained/{variant.split('/')[-1]}"
        self.simlingo = hydra.utils.instantiate(self.cfg.model, cfg_data_module=self.cfg.data_module, processor=self.processor, cache_dir=cache_dir, _recursive_=False)
        
        state_dict = torch.load(CHECKPOINT_PATH, map_location='cpu')
        if 'state_dict' in state_dict: state_dict = state_dict['state_dict']
        self.simlingo.load_state_dict({k.replace('module.', ''): v for k, v in state_dict.items()}, strict=False)
        self.simlingo.to(dtype=torch.bfloat16).to(self.device).eval()
        
        # =========================================================
        # [灰盒核心] 智能寻找 Vision Backbone
        # =========================================================
        print(f">>> [Grey-Box] Hunting for the actual Vision Backbone...")
        self.vision_model = None
        # ... (递归搜索 Vision Backbone 的逻辑保持不变) ...
        def is_viable_backbone(mod):
            has_embed = hasattr(mod, 'embeddings') or hasattr(mod, 'patch_embed') or hasattr(mod, 'patch_embedding')
            try:
                forward_func = getattr(mod, 'forward', None)
                return has_embed and callable(forward_func)
            except: return False

        def recursive_search(module, depth=0):
            if depth > 5: return None
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

        self.vision_model = recursive_search(self.simlingo)
        if self.vision_model is None:
            try:
                print(">>> [Grey-Box] Recursive search failed, attempting brute force unwrap...")
                if hasattr(self.simlingo, 'vision_model'):
                    self.vision_model = list(self.simlingo.vision_model.children())[0]
            except: pass

        if self.vision_model is None:
             raise ValueError("CRITICAL: Failed to locate underlying Vision Transformer model!")
        print(f">>> [Grey-Box] TARGET LOCKED: {self.vision_model.__class__.__name__}")
        # =========================================================

        self.simlingo_extractor = SimLingoFeatureExtractor(self.simlingo)
        
        print(">>> Loading Aux Models...")
        self.vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema").to(self.device).eval()
        self.lpips_vgg = lpips.LPIPS(net='vgg').to(self.device).eval()
        self.ssim_loss = SSIM().to(self.device)

        print(">>> Loading SiT (Flow Model)...")
        self.sit_model = SiT_models[ATTACK_PARAMS['sit_model']](input_size=32, num_classes=1000, qk_norm=False, finetune=True).to(self.device).eval()
        ckpt = torch.load(ATTACK_PARAMS['sit_ckpt'], map_location=self.device)
        self.sit_model.load_state_dict({k.replace("module.", ""): v for k, v in (ckpt["ema"] if "ema" in ckpt else ckpt).items()})

    # [更新] 加载 Measurements 和 Boxes (红绿灯)
    def load_measurements(self, image_path):
        """
        加载 Measurements 和 Boxes 数据，返回构建 Prompt 所需的所有元数据。
        与 openloopv2.py 逻辑保持一致。
        """
        img_dir = os.path.dirname(image_path)
        root_dir = os.path.dirname(img_dir)
        base_name = os.path.splitext(os.path.basename(image_path))[0] 
        
        # --- 1. 路径设置 ---
        meas_path = os.path.join(root_dir, 'measurements', f"{base_name}.json.gz")
        if not os.path.exists(meas_path):
            meas_path = os.path.join(root_dir, 'measurements', f"{base_name}.json")
            
        boxes_path = os.path.join(root_dir, 'boxes', f"{base_name}.json.gz")
        if not os.path.exists(boxes_path):
            boxes_path = os.path.join(root_dir, 'boxes', f"{base_name}.json")

        # --- 默认值 ---
        speed = 0.0
        cmd_id = 4
        traffic_light_state = "Green"
        tp1 = np.array([10.0, 0.0])
        tp2 = np.array([20.0, 0.0])
        gt_local = None

        # --- 2. 读取 Measurements ---
        if os.path.exists(meas_path):
            try:
                open_func = gzip.open if meas_path.endswith('.gz') else open
                with open_func(meas_path, 'rt', encoding='utf-8') as f:
                    data = json.load(f)
                
                speed = float(data.get('speed', speed))
                cmd_id = int(data.get('command', cmd_id))
                
                # 位姿处理
                ego_x = ego_y = 0.0
                pos_global = data.get('pos_global', None)
                if pos_global: 
                    ego_x, ego_y = float(pos_global[0]), float(pos_global[1])
                else: 
                    ego_matrix = data.get('ego_matrix', None)
                    if ego_matrix:
                        ego_x, ego_y = float(ego_matrix[0][3]), float(ego_matrix[1][3])
                        
                ego_theta = float(data.get('theta', 0.0))
                if abs(ego_theta) > 2 * math.pi: ego_theta = math.radians(ego_theta)
                
                # 路由处理 (GT Route)
                if 'route' in data:
                    route = np.array(data['route'], dtype=float)
                    if route.ndim >= 2: route = route[:, :2]
                    if np.max(np.abs(route)) > 1000.0:
                        gt_local = world_to_ego(route, ego_x, ego_y, ego_theta)
                    else:
                        gt_local = route
                
                # 提取 Target Points
                if gt_local is not None and len(gt_local) > 0:
                    dists = np.linalg.norm(gt_local, axis=1)
                    idx1 = np.where(dists > 10.0)[0]
                    tp1 = gt_local[idx1[0]] if len(idx1) > 0 else gt_local[-1]
                    idx2 = np.where(dists > 20.0)[0]
                    tp2 = gt_local[idx2[0]] if len(idx2) > 0 else gt_local[-1]
            except Exception as e:
                print(f"Error loading measurements {meas_path}: {e}")

        # --- 3. 读取 Boxes ---
        if os.path.exists(boxes_path):
            try:
                open_func = gzip.open if boxes_path.endswith('.gz') else open
                with open_func(boxes_path, 'rt', encoding='utf-8') as f:
                    boxes_data = json.load(f)
                    if isinstance(boxes_data, list):
                        for item in boxes_data:
                            if item.get('class') == 'ego_info':
                                raw_tl = item.get('traffic_light_state', 'None')
                                if raw_tl != "None":
                                    traffic_light_state = raw_tl
                                break
            except Exception as e:
                print(f"Error loading boxes {boxes_path}: {e}")

        return speed, cmd_id, traffic_light_state, tp1, tp2, gt_local

    # [更新] 整合 Command 和 Traffic Light 到 Prompt
    def prepare_simlingo_input(self, image_tensor, speed, tp1, tp2, command_text, traffic_light):
        IMG_TOKENS = '<img>' + '<IMG_CONTEXT>' * self.num_image_token * 2 + '</img>'
        prompt = (
            f"{IMG_TOKENS}\n"
            f"Current speed: {speed:.2f} m/s. "
            f"Command: {command_text}. "
            f"Traffic light: {traffic_light}. "
            f"Target waypoint: <TARGET_POINT><TARGET_POINT>. "
            f"Predict the waypoints."
        )
        
        tp_np = np.stack([tp1, tp2]).astype(np.float32)
        placeholder_dict = {self.tokenizer.convert_tokens_to_ids('<TARGET_POINT>'): tp_np}
        tokens = self.tokenizer([prompt], padding=True, return_tensors="pt")
        ll = LanguageLabel(
            phrase_ids=tokens["input_ids"].to(self.device), 
            phrase_valid=tokens["attention_mask"].bool().to(self.device), 
            phrase_mask=tokens["attention_mask"].bool().to(self.device), 
            placeholder_values=[placeholder_dict], 
            language_string=[prompt], 
            loss_masking=None
        )
        intrinsics = get_camera_intrinsics(448, 448, fov=110).unsqueeze(0).to(self.device).float().view(1, 3, 3)
        extrinsics = get_camera_extrinsics().unsqueeze(0).to(self.device).float().view(1, 4, 4)
        return DrivingInput(camera_images=image_tensor.unsqueeze(1).unsqueeze(1).repeat(1, 1, 2, 1, 1, 1).bfloat16(), image_sizes=torch.tensor([[448, 448]]).to(self.device), camera_intrinsics=intrinsics, camera_extrinsics=extrinsics, vehicle_speed=torch.tensor([[speed]]).to(self.device).bfloat16(), target_point=torch.from_numpy(tp1).bfloat16().unsqueeze(0).to(self.device), prompt=ll, prompt_inference=ll)

    def transform_for_simlingo(self, img_tensor):
        # [Modified for Center Crop]
        # 输入 img_tensor 已经是经过 Center Crop 并缩放到 256x256 的正方形
        # 我们只需要将其上采样到 448x448 适配模型输入
        img_448 = F.interpolate((img_tensor + 1) / 2, size=(448, 448), mode='bilinear', align_corners=False)
        mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
        return (img_448 - mean) / std

    # [灰盒关键] 鲁棒的前向传播 - 返回特征和注意力
    def get_vision_features_grad(self, img_tensor, return_attn=False):
        # 1. 预处理 (归一化 + Resize 448)
        pixel_values = self.transform_for_simlingo(img_tensor).to(dtype=torch.bfloat16)
        
        # 2. 尝试调用 (带注意力输出)
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
        
        # 3. 解析特征输出
        if hasattr(outputs, 'last_hidden_state'):
            features = outputs.last_hidden_state
        elif hasattr(outputs, 'pooler_output'):
            features = outputs.pooler_output
        elif isinstance(outputs, tuple):
            features = outputs[0]
        else:
            features = outputs
        
        # 4. 解析注意力输出 (如果有)
        if return_attn:
            if hasattr(outputs, 'attentions') and outputs.attentions is not None:
                # 取最后一层的注意力
                attn_weights = outputs.attentions[-1]  # [B, H, N, N]
            elif isinstance(outputs, tuple) and len(outputs) > 1:
                # 尝试从 tuple 中获取
                for item in outputs[1:]:
                    if isinstance(item, (list, tuple)) and len(item) > 0:
                        if isinstance(item[-1], torch.Tensor) and item[-1].dim() == 4:
                            attn_weights = item[-1]
                            break
            return features, attn_weights
        
        return features

    # SiT Inversion
    def invert_latents(self, z, class_labels, t_limit, num_steps):
        batch_size = z.shape[0]
        time_steps = torch.linspace(0.0, t_limit, num_steps + 1, device=self.device)
        with torch.no_grad():
            for i in range(num_steps):
                t_cur, t_next = time_steps[i], time_steps[i + 1]
                u = self.sit_model(z, torch.full((batch_size,), t_cur, device=self.device), torch.full((batch_size,), t_next, device=self.device), y=class_labels)
                if u.shape[1] == 8: u, _ = u.chunk(2, dim=1)
                z = z + (t_next - t_cur) * u
        return z

    # SiT Sampling
    def differentiable_flow_sampler(self, z, class_labels, t_limit, delta_u_dict=None, num_steps=2):
        batch_size = z.shape[0]
        time_steps = torch.linspace(t_limit, 0.0, num_steps + 1, device=self.device)
        for i in range(num_steps):
            t_cur, t_next = time_steps[i], time_steps[i + 1]
            u = self.sit_model(z, torch.full((batch_size,), t_next, device=self.device), torch.full((batch_size,), t_cur, device=self.device), y=class_labels)
            if u.shape[1] == 8: u, _ = u.chunk(2, dim=1)
            
            if delta_u_dict is not None and str(i) in delta_u_dict:
                u = u + delta_u_dict[str(i)]
            
            z = z - (t_cur - t_next) * u
            if torch.isnan(z).any(): z = torch.nan_to_num(z, nan=0.0)
        return z

    def run_attack(self, image_path, save_dir, fid_clean_dir, fid_adv_dir, file_prefix=""):
            real_speed, cmd_id, tl_state, tp1, tp2, gt_route = self.load_measurements(image_path)
            command_text = get_cmd_text(cmd_id)
            # 获取纯文件名，不带路径和后缀
            base_filename = os.path.splitext(os.path.basename(image_path))[0]
            
            # === [Modified for Center Crop] 读取图片并进行 Center Crop 预处理 ===
            raw_img_pil = Image.open(image_path).convert('RGB')
            orig_W, orig_H = raw_img_pil.size
            
            # 计算中心裁剪区域 (取短边作为正方形边长)
            crop_size = min(orig_W, orig_H)
            left = (orig_W - crop_size) // 2
            top = (orig_H - crop_size) // 2
            right = left + crop_size
            bottom = top + crop_size
            
            # 裁剪出中间的正方形
            img_cropped = raw_img_pil.crop((left, top, right, bottom))
            
            # 将裁剪后的正方形缩放到 256x256 供 VAE/SiT 攻击使用
            # 这样 VAE 处理的是几何比例正常的图像
            input_pil = img_cropped.resize((256, 256), Image.BILINEAR)
            img_t = (torch.from_numpy(np.array(input_pil)).permute(2,0,1).float().unsqueeze(0).to(self.device) / 127.5 - 1.0)

            y_tensor = torch.tensor([TARGET_CLASS_IDX], device=self.device)
            t_limit = ATTACK_PARAMS['t_limit']

            # === 1. 原始状态预测与特征提取 ===
            with torch.no_grad():
                self.simlingo_extractor.clear()
                z_real = self.vae.encode(img_t).latent_dist.sample() * 0.18215
                recon_img = torch.clamp(self.vae.decode(z_real / 0.18215).sample, -1, 1)
                
                # Full Clean Inference (for Reference Metrics ONLY)
                # 此时 transform_for_simlingo 接收的是正方形输入，SimLingo 看到的也是无畸变图像
                s_clean, r_clean, _ = self.simlingo(self.prepare_simlingo_input(self.transform_for_simlingo(recon_img), real_speed, tp1, tp2, command_text, tl_state))
                clean_feats_log = [f.clone() for f in self.simlingo_extractor.features] 
                
                # 获取干净样本的视觉特征 (Detached) 用于 Loss 锚点
                clean_vision_emb = self.get_vision_features_grad(recon_img).detach()

                # SiT Inversion
                z_mid = self.invert_latents(z_real, y_tensor, t_limit, ATTACK_PARAMS['sample_steps'])
                real_z_std = z_real.std()

            # === 2. 攻击优化 (Grey-Box Loop) ===
            delta_z = nn.Parameter(torch.randn_like(z_mid) * 1e-4)
            delta_u_seq = nn.ParameterDict({str(i): nn.Parameter(torch.zeros_like(z_mid)) for i in range(ATTACK_PARAMS['sample_steps'])})
            optimizer = optim.Adam([{'params': [delta_z], 'lr': ATTACK_PARAMS['lr_z']}, {'params': delta_u_seq.parameters(), 'lr': ATTACK_PARAMS['lr_u']}])

            # 获取干净图像的注意力图 (用于引导攻击)
            with torch.no_grad():
                _, clean_attn = self.get_vision_features_grad(recon_img, return_attn=True)
            img_name = os.path.basename(image_path)
            pbar = tqdm(range(ATTACK_PARAMS['iterations']), desc=f"GreyBox Attack {img_name}", leave=False)
            for i in pbar:
                optimizer.zero_grad()
                z_start = z_mid + delta_z
                z_f = self.differentiable_flow_sampler(z_start, y_tensor, t_limit, delta_u_seq, ATTACK_PARAMS['sample_steps'])
                
                if torch.isnan(z_f).any(): break

                img_adv = torch.clamp(self.vae.decode(z_f / 0.18215).sample, -1, 1)
                
                # [灰盒关键] 仅通过 Vision Model 计算梯度
                adv_vision_emb = self.get_vision_features_grad(img_adv)
                
                # === 多目标攻击损失 ===
                loss_feature_dist = get_road_focused_feature_loss(adv_vision_emb, clean_vision_emb)
                
                adv_flat = adv_vision_emb.view(adv_vision_emb.size(0), -1)
                clean_flat = clean_vision_emb.view(clean_vision_emb.size(0), -1)
                cos_sim = F.cosine_similarity(adv_flat, clean_flat, dim=-1).mean()
                loss_cosine = cos_sim 
                
                loss_attn_focus = get_attention_weighted_feature_loss(
                    adv_vision_emb, clean_vision_emb, clean_attn, 
                    temperature=ATTACK_PARAMS['attn_temperature']
                )
                
                loss_anchor = torch.abs(z_f.std() - real_z_std)
                
                with torch.no_grad():
                    loss_ssim = 1.0 - self.ssim_loss(img_adv, recon_img)

                attack_loss = -ATTACK_PARAMS['w_feature'] * loss_feature_dist \
                            - ATTACK_PARAMS['w_cosine'] * loss_cosine \
                            - ATTACK_PARAMS['w_attn_focus'] * loss_attn_focus
                
                loss_total = attack_loss + ATTACK_PARAMS['w_anchor'] * loss_anchor
                loss_total.backward()
                optimizer.step()
                
                with torch.no_grad():
                    delta_z.data.clamp_(-ATTACK_PARAMS['epsilon_latent'], ATTACK_PARAMS['epsilon_latent'])
                    for k in delta_u_seq: 
                        delta_u_seq[k].data.clamp_(-ATTACK_PARAMS['epsilon_u'], ATTACK_PARAMS['epsilon_u'])
                
                pbar.set_postfix({'FeatDist': f"{loss_feature_dist.item():.4f}", 'SSIM': f"{1-loss_ssim.item():.3f}"})

           # === 3. 最终生成与完整指标计算 ===
            with torch.no_grad():
                self.simlingo_extractor.clear() 
                # 1. 生成最终对抗样本 (VAE Latent -> Image, 256x256 crop)
                final_zf = self.differentiable_flow_sampler(z_mid + delta_z, y_tensor, t_limit, delta_u_seq, ATTACK_PARAMS['sample_steps'])
                img_f = torch.clamp(self.vae.decode(torch.nan_to_num(final_zf) / 0.18215).sample, -1, 1)

                # ================= [Modified for Center Crop] 贴回逻辑 =================
                # --- 保存逻辑：将攻击后的 256x256 正方形插值回 crop_size 并贴回原图 ---
                
                # 1. 准备原始背景 (Clean Background)
                # 我们将原始大图转换为 Tensor [-1, 1]
                raw_img_tensor = (torch.from_numpy(np.array(raw_img_pil)).permute(2,0,1).float().unsqueeze(0).to(self.device) / 127.5 - 1.0)
                
                # 2. 放大对抗样本 (256x256 -> crop_size x crop_size)
                img_f_upscaled = F.interpolate(img_f, size=(crop_size, crop_size), mode='bilinear', align_corners=False)
                recon_img_upscaled = F.interpolate(recon_img, size=(crop_size, crop_size), mode='bilinear', align_corners=False)
                
                # 3. 创建最终输出画布 (复制原始背景)
                # 这样非 Crop 区域（如左右两边）保持原始图像不变，中间变成对抗样本
                final_full_adv = raw_img_tensor.clone()
                final_full_clean = raw_img_tensor.clone()
                
                # 4. 贴回中间位置 (Paste Back)
                final_full_adv[:, :, top:bottom, left:right] = img_f_upscaled
                final_full_clean[:, :, top:bottom, left:right] = recon_img_upscaled
                # ====================================================================

                # 为了计算 SimLingo 的指标，依然生成 512 版本 (保持原有逻辑兼容性) - 这里是对 Crop 的 512
                img_f_512 = F.interpolate(img_f, size=(512, 512), mode='bilinear', align_corners=False)
                recon_img_512 = F.interpolate(recon_img, size=(512, 512), mode='bilinear', align_corners=False)

                # [评估阶段] 完整运行 SimLingo (检查攻击在代理模型上的效果)
                # 注意：这里 SimLingo 接收的是 img_f (Center Crop)，因为在 Center Crop 设置下我们关心的是模型对核心区域的反应
                s_f, r_f, _ = self.simlingo(self.prepare_simlingo_input(self.transform_for_simlingo(img_f), real_speed, tp1, tp2, command_text, tl_state))
                adv_feats_log = [f.clone() for f in self.simlingo_extractor.features]

                # --- 详细误差拆解 ---
                clean_end = r_clean[0, -1] 
                adv_end = r_f[0, -1]
                
                # 1. 总体偏移 (Shift)
                shift = torch.norm(r_f.float() - r_clean.float(), p=2).item()
                spd_err = torch.norm(s_f.float() - s_clean.float(), p=2).item()
                
                # 2. 横向误差 (Adv vs Clean)
                lat_error = abs(adv_end[1].item() - clean_end[1].item())
                
                # 3. 纵向误差 (Adv vs Clean)
                lon_error = abs(adv_end[0].item() - clean_end[0].item())
                
                # 4. 原始路程长度 (用于 Tan 分母)
                clean_route_len = abs(clean_end[0].item())
                
                is_success = 1.0 if shift >= 1.0 else 0.0

                final_ssim = self.ssim_loss(img_f_512, recon_img_512).item()
                final_lpips = self.lpips_vgg(img_f_512, recon_img_512).mean().item()
                sim_drift = F.mse_loss(adv_feats_log[-1].float(), clean_feats_log[-1].float()).item() * 1000 if adv_feats_log else 0
                
                # ================= [Modified for Center Crop] 保存完整的宽图 =================
                # 获取文件名 (兼容 jpg/png 输入，输出强制为 png)
                base_name = os.path.splitext(os.path.basename(image_path))[0]
                
                # 1. 先定义保存函数
                def save_safe_png(tensor, path):
                    """tensor: [-1, 1] -> [0, 255] PNG"""
                    img_01 = (tensor + 1) / 2
                    ndarr = img_01[0].mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
                    Image.fromarray(ndarr).save(path, format='PNG')

                # 2. 计算文件名
                if file_prefix:
                    save_name = f"{file_prefix}_{base_name}.png"
                else:
                    save_name = f"{base_name}.png"
                
                # 3. 保存文件 (使用贴回后的完整图像)
                save_safe_png(final_full_clean, os.path.join(fid_clean_dir, save_name))
                save_safe_png(final_full_adv, os.path.join(fid_adv_dir, save_name))
                save_safe_png(final_full_adv, os.path.join(save_dir, save_name))
                # =========================================================

                # 可视化 (使用 Crop 部分以便观察噪声细节)
                self.save_unified_visualization((recon_img+1)/2, (img_f+1)/2, r_clean, r_f, gt_route, tp1, s_clean, s_f, save_dir, img_name)

            # 返回值顺序: 
            # 0:Shift, 1:Spd, 2:Drift, 3:LPIPS, 4:SSIM, 
            # 5:Lat_Err, 6:Lon_Err, 7:Clean_Len, 8:Succ
            return shift, spd_err, sim_drift, final_lpips, final_ssim, lat_error, lon_error, clean_route_len, is_success
    
    def save_unified_visualization(self, recon_01, adv_01, rc, ra, gt, tp, sc, sa, save_dir, name):
        try:
            r_np = recon_01[0].cpu().float().numpy().transpose(1, 2, 0)
            a_np = adv_01[0].cpu().float().numpy().transpose(1, 2, 0)
            r_np = np.clip(r_np, 0, 1); a_np = np.clip(a_np, 0, 1)
            diff = np.abs(a_np - r_np) * 15.0; diff = np.clip(diff, 0, 1)
            sc_val = sc.float().mean().item() if sc.numel() > 1 else sc.item()
            sa_val = sa.float().mean().item() if sa.numel() > 1 else sa.item()

            fig = plt.figure(figsize=(16, 14))
            gs = fig.add_gridspec(2, 2, hspace=0.2, wspace=0.15)
            
            ax1 = fig.add_subplot(gs[0, 0]); ax1.imshow(r_np); ax1.set_title("1. Clean (Center Crop)", fontsize=14, fontweight='bold'); ax1.axis('off')
            ax2 = fig.add_subplot(gs[0, 1]); ax2.imshow(a_np); ax2.set_title("2. Adv (Center Crop)", fontsize=14, fontweight='bold'); ax2.axis('off')
            ax3 = fig.add_subplot(gs[1, 0]); ax3.imshow(diff); ax3.set_title("3. Noise", fontsize=14, fontweight='bold'); ax3.axis('off')

            ax4 = fig.add_subplot(gs[1, 1])
            c_pts = rc[0].cpu().float().numpy(); a_pts = ra[0].cpu().float().numpy()
            if gt is not None: ax4.plot(gt[:, 1], gt[:, 0], 'g-', alpha=0.3, linewidth=5, label='Ground Truth')
            ax4.plot(c_pts[:, 1], c_pts[:, 0], 'b-o', markersize=5, linewidth=2, label=f'Clean ({sc_val:.1f} m/s)')
            ax4.plot(a_pts[:, 1], a_pts[:, 0], 'r--^', markersize=6, linewidth=2, label=f'Adv ({sa_val:.1f} m/s)')
            
            tp_np = np.array(tp)
            tx, ty = (tp_np[0], tp_np[1]) if tp_np.ndim == 1 else (tp_np[0, 0], tp_np[0, 1])
            ax4.scatter(ty, tx, c='gold', marker='*', s=300, edgecolors='black', label='Target', zorder=5)

            ax4.set_xlim(-12, 12); ax4.set_ylim(-2, 40)
            ax4.set_xlabel("Lateral (m)"); ax4.set_ylabel("Forward (m)")
            ax4.grid(True, linestyle=':', alpha=0.6); ax4.legend(loc='upper right')
            ax4.set_title("4. Trajectory (Feature Attack)", fontsize=14, fontweight='bold')

            plt.tight_layout()
            save_path = os.path.join(save_dir, f"{name}_vis.png")
            plt.savefig(save_path, dpi=120, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"Vis error: {e}")

def get_next_log_filename(output_dir):
    """生成递增的日志文件名"""
    existing_logs = glob.glob(os.path.join(output_dir, "attack_log_*.txt"))
    indices = [int(os.path.basename(log).split("_")[2].split(".")[0]) for log in existing_logs if "attack_log_" in log]
    next_index = max(indices) + 1 if indices else 1
    return os.path.join(output_dir, f"attack_log_{next_index}.txt")

import gc

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True, 
                        help="Root directory containing multiple dataset folders")
    parser.add_argument("--output_dir", type=str, default="mfa_allresults") 
    args = parser.parse_args()

    # 1. 确定本次运行的序号文件夹 (MFA-gray-N)
    base_dir = args.output_dir
    os.makedirs(base_dir, exist_ok=True)

    existing_folders = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    max_id = 0
    for folder in existing_folders:
        if folder.startswith("MFA-gray-"):
            try:
                idx = int(folder.split("-")[-1])
                if idx > max_id: max_id = idx
            except ValueError: continue
    
    new_folder_name = f"MFA-gray-{max_id + 1}"
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
    print("Grey-Box MFA (SiT) Config:")
    print(json.dumps(ATTACK_PARAMS, indent=2))
    
    attacker = MFAttackerSiT(args)
    
    # 5. 设置 FID 文件夹 (在当前运行目录下)
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
        
        current_rgb_dir = os.path.join(args.data_root, dataset_name, "rgb")
        current_out_dir = os.path.join(current_run_dir, dataset_name)
        os.makedirs(current_out_dir, exist_ok=True)
        
        # 优先查找 png，如果没有则查找 jpg
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
                res = attacker.run_attack(img_path, current_out_dir, global_fid_clean, global_fid_adv, file_prefix=dataset_name)
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
                succ_str = "YES" if is_succ > 0.5 else "NO"
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