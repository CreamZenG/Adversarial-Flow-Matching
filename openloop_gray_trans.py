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
# === 在 import 区域添加 ===
from torchvision.transforms.functional import center_crop

# === 在全局常量区 (Global Constants) 添加 ===
# 例如：裁剪为 512x512 或 448x448，请根据你的原始图片尺寸修改
# 格式为 (Height, Width)
CROP_SIZE = (160, 160)

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
    "lr_z": 0.08,                # 提高学习率0.05
    "lr_u": 0.05,              
    "t_limit": 0.05,             # 保持时间范围
    "sample_steps": 1,           # 保持单步采样
    "active_steps": [0, 1],      
    "epsilon_latent": 0.05,      # 保持微小扰动
    "epsilon_u": 0.05,          
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
        base_name = os.path.basename(image_path).replace('.jpg', '')
        
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
        # 强制 Resize 到 448x448 (InternVL 标准输入)
        img_448 = F.interpolate((img_tensor + 1) / 2, size=(448, 448), mode='bilinear')
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

    def run_attack(self, image_path, save_dir, fid_clean_dir, fid_adv_dir, file_prefix="", external_adv_path=None):
            real_speed, cmd_id, tl_state, tp1, tp2, gt_route = self.load_measurements(image_path)
            command_text = get_cmd_text(cmd_id)
            img_name = os.path.basename(image_path).split('.')[0]
            
            # --- [修改开始] 加载 Clean 图片并执行中心裁剪 ---
            clean_pil_full = Image.open(image_path).convert('RGB')
            clean_tensor_full = torch.from_numpy(np.array(clean_pil_full)).permute(2,0,1).float().unsqueeze(0).to(self.device)
            
            # 执行中心裁剪
            clean_tensor_cropped = center_crop(clean_tensor_full, CROP_SIZE)
            
            # 后续所有操作基于 clean_tensor_cropped
            # transform_for_simlingo 会负责后续 Resize 到 448 (如果裁剪尺寸不是 448) 并归一化
            with torch.no_grad():
                self.simlingo_extractor.clear()
                # 注意：这里输入变成了 clean_tensor_cropped
                s_clean, r_clean, _ = self.simlingo(
                    self.prepare_simlingo_input(
                        self.transform_for_simlingo((clean_tensor_cropped/127.5)-1.0), 
                        real_speed, tp1, tp2, command_text, tl_state
                    )
                )
                clean_feats_log = [f.clone() for f in self.simlingo_extractor.features]

            # ==========================================
            # [修改] 分支逻辑：如果有外部对抗样本，直接加载并评估
            # ==========================================
            if external_adv_path and os.path.exists(external_adv_path):
                # 1. 加载 TransFuser 生成的对抗样本
                tf_adv_pil = Image.open(external_adv_path).convert('RGB')
                tf_adv_tensor = torch.from_numpy(np.array(tf_adv_pil)).permute(2,0,1).float().unsqueeze(0).to(self.device)
                
                # --- [修改开始] 对对抗样本执行中心裁剪 ---
                # 确保裁剪尺寸与 Clean 样本一致
                tf_adv_cropped = center_crop(tf_adv_tensor, CROP_SIZE)
                
                # 2. Resize 到 SimLingo 需要的 448x448 (如果 CROP_SIZE 已经是 448，这一步其实就是恒等变换，但保留以防万一)
                img_f_448 = F.interpolate(tf_adv_cropped, size=(448, 448), mode='bilinear', align_corners=False)
                
                # 3. 归一化到 [-1, 1] 准备给模型
                img_f = (img_f_448 / 127.5) - 1.0
                # --- [修改结束] ---
                
                pass

            else:
                # === 原有的灰盒攻击逻辑 (如果没有外部图片) ===
                # ... (这里省略掉原有的 SiT/VAE 优化代码，保持不变) ...
                # 如果你需要同时保留原有功能，就把原来的优化代码放在这个 else 块里
                # 为了简洁，这里假设你主要跑迁移，这里是原来的 Attack Loop
                
                # (原代码的 VAE Encode, SiT Inversion, Optimization Loop ...)
                # 最终得到 img_f
                
                # 占位符，实际使用时请保留原有逻辑
                img_f = img_t # 仅作演示
                pass

            # === 3. 最终生成与完整指标计算 (通用部分) ===
            with torch.no_grad():
                self.simlingo_extractor.clear() 
                
                # 推理对抗样本 (img_f 已经是裁剪并 resize 过的)
                simlingo_in = self.transform_for_simlingo(img_f)
                
                s_f, r_f, _ = self.simlingo(
                    self.prepare_simlingo_input(simlingo_in, real_speed, tp1, tp2, command_text, tl_state)
                )
                adv_feats_log = [f.clone() for f in self.simlingo_extractor.features]
                # --- 详细误差拆解 ---
                clean_end = r_clean[0, -1] 
                adv_end = r_f[0, -1]
                
                # 1. 总体偏移 (Shift)
                shift = torch.norm(r_f.float() - r_clean.float(), p=2).item()
                spd_err = torch.norm(s_f.float() - s_clean.float(), p=2).item()
                
                # 2. 横向/纵向误差
                lat_error = abs(adv_end[1].item() - clean_end[1].item())
                lon_error = abs(adv_end[0].item() - clean_end[0].item())
                clean_route_len = abs(clean_end[0].item())
                
                is_success = 1.0 if shift >= 1.0 else 0.0

                # 计算 SSIM/LPIPS (需要 Resize 到相同尺寸)
                # 使用 512x512 作为对比基准
                img_f_512 = F.interpolate(img_f, size=(512, 512), mode='bilinear')
                
                # --- [修改开始] 基准图必须来自裁剪后的 clean tensor ---
                # 原代码: recon_img_512 = F.interpolate(clean_tensor_full, size=(512, 512), mode='bilinear')
                # 新代码: 使用 clean_tensor_cropped
                recon_img_512 = F.interpolate(clean_tensor_cropped, size=(512, 512), mode='bilinear')
                # --- [修改结束] ---
                # 归一化到 [-1, 1] 如果之前 clean_tensor_full 是 [0, 255]
                if recon_img_512.max() > 1.0: recon_img_512 = (recon_img_512 / 127.5) - 1.0

                final_ssim = self.ssim_loss(img_f_512, recon_img_512).item()
                final_lpips = self.lpips_vgg(img_f_512, recon_img_512).mean().item()
                
                sim_drift = 0.0
                if adv_feats_log and clean_feats_log:
                    sim_drift = F.mse_loss(adv_feats_log[-1].float(), clean_feats_log[-1].float()).item() * 1000
                
                # 保存图片
                base_name = os.path.basename(image_path).split('.')[0]
                save_name = f"{file_prefix}_{base_name}.jpg"
                
                def save_safe_jpg(tensor, path):
                    # Tensor [-1, 1] -> [0, 255]
                    img_01 = (tensor + 1) / 2
                    ndarr = img_01[0].mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
                    Image.fromarray(ndarr).save(path, format='JPEG', quality=95)

                save_safe_jpg(recon_img_512, os.path.join(fid_clean_dir, save_name))
                save_safe_jpg(img_f_512, os.path.join(fid_adv_dir, save_name))

                self.save_unified_visualization((recon_img_512+1)/2, (img_f_512+1)/2, r_clean, r_f, gt_route, tp1, s_clean, s_f, save_dir, img_name)
                
                # 保存单独的矢量图
                self.save_individual_images(
                    clean_tensor=recon_img_512, 
                    adv_tensor=img_f_512, 
                    route_clean=r_clean, 
                    route_adv=r_f, 
                    gt_route=gt_route, 
                    tp=tp1, 
                    speed_clean=s_clean, 
                    speed_adv=s_f, 
                    save_dir=save_dir, 
                    name=img_name
                )

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
            
            ax1 = fig.add_subplot(gs[0, 0]); ax1.imshow(r_np); ax1.set_title("1. Clean", fontsize=14, fontweight='bold'); ax1.axis('off')
            ax2 = fig.add_subplot(gs[0, 1]); ax2.imshow(a_np); ax2.set_title("2. Adv (Grey-Box)", fontsize=14, fontweight='bold'); ax2.axis('off')
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

    def save_individual_images(self, clean_tensor, adv_tensor, route_clean, route_adv, gt_route, tp, speed_clean, speed_adv, save_dir, name):
        """
        单独保存矢量图：干净图像、攻击后图像、噪声图像、轨迹图
        所有图像尺寸按照 SimLingo 输入尺寸 (448x448) 保存
        """
        try:
            # 创建子目录
            svg_dir = os.path.join(save_dir, "svg_outputs")
            os.makedirs(svg_dir, exist_ok=True)
            
            # Resize 到 SimLingo 标准输入尺寸 448x448
            clean_448 = F.interpolate(clean_tensor, size=(448, 448), mode='bilinear', align_corners=False)
            adv_448 = F.interpolate(adv_tensor, size=(448, 448), mode='bilinear', align_corners=False)
            
            # 归一化到 [0, 1]
            clean_01 = (clean_448 + 1) / 2 if clean_448.min() < 0 else clean_448 / 255.0 if clean_448.max() > 1 else clean_448
            adv_01 = (adv_448 + 1) / 2 if adv_448.min() < 0 else adv_448 / 255.0 if adv_448.max() > 1 else adv_448
            
            clean_01 = torch.clamp(clean_01, 0, 1)
            adv_01 = torch.clamp(adv_01, 0, 1)
            
            # 转换为 numpy
            clean_np = clean_01[0].cpu().float().numpy().transpose(1, 2, 0)
            adv_np = adv_01[0].cpu().float().numpy().transpose(1, 2, 0)
            
            # 计算噪声图 (放大 15 倍以便可视化)
            noise_np = np.abs(adv_np - clean_np) * 15.0
            noise_np = np.clip(noise_np, 0, 1)
            
            # 1. 保存干净图像 (SVG)
            fig_clean = plt.figure(figsize=(4.48, 4.48), dpi=100)
            ax = fig_clean.add_axes([0, 0, 1, 1])
            ax.imshow(clean_np)
            ax.axis('off')
            plt.savefig(os.path.join(svg_dir, f"{name}_clean.svg"), format='svg', bbox_inches='tight', pad_inches=0)
            plt.close(fig_clean)
            
            # 2. 保存攻击后图像 (SVG)
            fig_adv = plt.figure(figsize=(4.48, 4.48), dpi=100)
            ax = fig_adv.add_axes([0, 0, 1, 1])
            ax.imshow(adv_np)
            ax.axis('off')
            plt.savefig(os.path.join(svg_dir, f"{name}_adv.svg"), format='svg', bbox_inches='tight', pad_inches=0)
            plt.close(fig_adv)
            
            # 3. 保存噪声图像 (SVG)
            fig_noise = plt.figure(figsize=(4.48, 4.48), dpi=100)
            ax = fig_noise.add_axes([0, 0, 1, 1])
            ax.imshow(noise_np)
            ax.axis('off')
            plt.savefig(os.path.join(svg_dir, f"{name}_noise.svg"), format='svg', bbox_inches='tight', pad_inches=0)
            plt.close(fig_noise)
            
            # 4. 保存轨迹图 (SVG 矢量格式，正方形)
            sc_val = speed_clean.float().mean().item() if speed_clean.numel() > 1 else speed_clean.item()
            sa_val = speed_adv.float().mean().item() if speed_adv.numel() > 1 else speed_adv.item()
            
            fig_traj = plt.figure(figsize=(8, 8))
            ax = fig_traj.add_subplot(111)
            
            c_pts = route_clean[0].cpu().float().numpy()
            a_pts = route_adv[0].cpu().float().numpy()
            
            if gt_route is not None:
                ax.plot(gt_route[:, 1], gt_route[:, 0], 'g-', alpha=0.3, linewidth=5, label='Ground Truth')
            
            ax.plot(c_pts[:, 1], c_pts[:, 0], 'b-o', markersize=5, linewidth=2, label=f'Clean ({sc_val:.1f} m/s)')
            ax.plot(a_pts[:, 1], a_pts[:, 0], 'r--^', markersize=6, linewidth=2, label=f'Adv ({sa_val:.1f} m/s)')
            
            tp_np = np.array(tp)
            tx, ty = (tp_np[0], tp_np[1]) if tp_np.ndim == 1 else (tp_np[0, 0], tp_np[0, 1])
            ax.scatter(ty, tx, c='gold', marker='*', s=300, edgecolors='black', label='Target', zorder=5)
            
            ax.set_xlim(-12, 12)
            ax.set_ylim(-2, 40)
            ax.set_xlabel("Lateral (m)")
            ax.set_ylabel("Forward (m)")
            ax.grid(True, linestyle=':', alpha=0.6)
            ax.legend(loc='upper right')
            ax.set_title("Trajectory (AFM Attack)", fontsize=14, fontweight='bold')
            
            plt.tight_layout()
            plt.savefig(os.path.join(svg_dir, f"{name}_trajectory.svg"), format='svg', bbox_inches='tight')
            plt.close(fig_traj)
            
        except Exception as e:
            print(f"Individual SVG save error: {e}")

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
    parser.add_argument("--external_adv_dir", type=str, default=None, 
                    help="Path to adversarial images generated by TransFuser")
    
    args = parser.parse_args()

    # 1. 确定本次运行的序号文件夹
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

    # 2. Log Config
    log_path = os.path.join(current_run_dir, "attack_log.txt")
    config_path = os.path.join(current_run_dir, "attack_config.txt")

    with open(config_path, 'w') as f:
        f.write(f"Run ID: {max_id + 1}\n")
        json.dump(ATTACK_PARAMS, f, indent=4)

    sys.stdout = Logger(log_path)
    
    attacker = MFAttackerSiT(args)
    
    global_fid_clean = os.path.join(current_run_dir, "global_fid_clean")
    global_fid_adv = os.path.join(current_run_dir, "global_fid_adv")
    if os.path.exists(global_fid_clean): shutil.rmtree(global_fid_clean)
    if os.path.exists(global_fid_adv): shutil.rmtree(global_fid_adv)
    os.makedirs(global_fid_clean); os.makedirs(global_fid_adv)

    if not os.path.exists(args.data_root):
        print(f"Error: Data root {args.data_root} does not exist."); sys.exit(1)

    all_subdirs = sorted([d for d in os.listdir(args.data_root) if os.path.isdir(os.path.join(args.data_root, d))])
    valid_datasets = []
    for d in all_subdirs:
        if os.path.exists(os.path.join(args.data_root, d, "rgb")):
            valid_datasets.append(d)
    
    print(f"\nFound {len(valid_datasets)} Datasets: {valid_datasets}\n")
    
    # === [修复] 初始化统计变量 ===
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
        
        all_files = sorted(os.listdir(current_rgb_dir))
        images = [os.path.join(current_rgb_dir, f) for f in all_files if f.lower().endswith(('.jpg', '.png'))]
        
        if not images: continue

        ds_totals = {
            'shift': 0.0, 'spd_err': 0.0, 'sim_drift': 0.0, 
            'road_lpips': 0.0, 'road_ssim': 0.0,
            'success_count': 0.0,
            'sum_lat_offset': 0.0,
            'sum_clean_len': 0.0,
            'total_time': 0.0
        }
        count = 0

        # [MFA优化] 建立外部对抗样本索引
        # 支持两种目录结构:
        # 1. 扁平结构: fid_global_adv/RouteName_0000.png (生成端格式)
        # 2. 分层结构: RouteName/adv_rgb/RouteName_0000.png
        adv_map = {}
        if args.external_adv_dir:
            print(f"    Scanning external dir: {args.external_adv_dir}")
            try:
                # 优先尝试分层结构: external_adv_dir/dataset_name/adv_rgb/
                layered_dir = os.path.join(args.external_adv_dir, dataset_name, "adv_rgb")
                if os.path.isdir(layered_dir):
                    scan_dir = layered_dir
                    filter_prefix = None  # 分层结构不需要过滤
                    print(f"    Using layered structure: {layered_dir}")
                else:
                    scan_dir = args.external_adv_dir
                    filter_prefix = dataset_name + "_"  # 扁平结构需要按路线名过滤
                    print(f"    Using flat structure: {scan_dir}, filtering by prefix '{filter_prefix}'")
                
                ext_files = os.listdir(scan_dir)
                for f in ext_files:
                    if not f.lower().endswith((".jpg", ".png")):
                        continue
                    
                    stem = os.path.splitext(f)[0]
                    full_path = os.path.join(scan_dir, f)
                    
                    # 扁平结构：只处理当前路线的文件
                    if filter_prefix:
                        if not stem.startswith(filter_prefix):
                            continue  # 跳过其他路线的文件
                        # 生成端命名: RouteName_0000 -> 提取末尾索引 0000
                        idx_str = stem[len(filter_prefix):]  # 去掉前缀后得到 "0000"
                        if idx_str.isdigit():
                            # 原图可能是 00000.jpg (5位) 或其他格式
                            adv_map[idx_str] = full_path  # 4位: "0000"
                            adv_map[idx_str.zfill(5)] = full_path  # 5位: "00000"
                            adv_map[idx_str.lstrip("0") or "0"] = full_path  # 无前导零
                    else:
                        # 分层结构：直接用 stem 作为 key
                        if "_" in stem:
                            parts = stem.rsplit("_", 1)
                            if len(parts) == 2 and parts[1].isdigit():
                                idx_str = parts[1]
                                adv_map[idx_str] = full_path
                                adv_map[idx_str.zfill(5)] = full_path
                                adv_map[idx_str.lstrip("0") or "0"] = full_path
                                continue
                        adv_map[stem] = full_path
                        
                print(f"    Mapped {len(adv_map)} adv images for route '{dataset_name}'")
            except Exception as e:
                print(f"    Failed scanning external adv dir: {e}")

        for i, img_path in enumerate(images):
            t0 = time.time()
            try:
                ext_adv_path = None
                if args.external_adv_dir:
                    # 通过原图的 basename 去匹配外部对抗样本
                    # 原图格式: 00000.jpg -> base_name = "00000"
                    base_name = os.path.splitext(os.path.basename(img_path))[0]
                    
                    # 尝试多种格式匹配
                    ext_adv_path = adv_map.get(base_name)  # 精确匹配
                    if ext_adv_path is None and base_name.isdigit():
                        # 尝试去掉/添加前导零
                        ext_adv_path = adv_map.get(base_name.lstrip("0") or "0")
                        if ext_adv_path is None:
                            ext_adv_path = adv_map.get(base_name.zfill(4))
                        if ext_adv_path is None:
                            ext_adv_path = adv_map.get(base_name.zfill(5))
                    
                    if ext_adv_path is None:
                        if i < 5: 
                            print(f"    [Warn] Missing adv for '{base_name}'. Available keys sample: {list(adv_map.keys())[:5]}")
                        continue  # 跳过未找到对应对抗样本的帧

                res = attacker.run_attack(
                    img_path, 
                    current_out_dir, 
                    global_fid_clean, 
                    global_fid_adv, 
                    file_prefix=dataset_name,
                    external_adv_path=ext_adv_path 
                )
                
                step_time = time.time() - t0
                
                # 解包返回值
                # 0:Shift, 1:Spd, 2:Drift, 3:LPIPS, 4:SSIM, 5:Lat, 6:Lon, 7:Len, 8:Succ
                ds_totals['shift'] += res[0]
                ds_totals['spd_err'] += res[1]
                ds_totals['sim_drift'] += res[2]
                ds_totals['road_lpips'] += res[3]
                ds_totals['road_ssim'] += res[4]
                ds_totals['sum_lat_offset'] += res[5]
                ds_totals['sum_clean_len'] += res[7]
                ds_totals['success_count'] += res[8]
                ds_totals['total_time'] += step_time
                count += 1
                
                # 实时打印简略信息
                print(f"[{dataset_name}][{i}] Shift: {res[0]:.2f}m | Lat: {res[5]:.2f} | SSIM: {res[4]:.3f} | Succ: {int(res[8])}")

            except KeyboardInterrupt: sys.exit(1)
            except Exception as e: 
                print(f"Error processing {img_path}: {e}")
            finally: torch.cuda.empty_cache()

        # === 数据集结算 (恢复完整的打印) ===
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

    # === 宏观平均报告 (恢复完整的打印) ===
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