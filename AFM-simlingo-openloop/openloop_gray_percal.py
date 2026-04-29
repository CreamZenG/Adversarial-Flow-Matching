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
import warnings
from math import pi, cos
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

# === PerC-AL 攻击参数 ===
# 相比分类任务，回归任务通常需要更精细的扰动
ATTACK_PARAMS = {
    "max_iterations": 400,    # 迭代次数
    "alpha_l_init": 10,    # 任务损失更新步长原版224*224设置的1,448*448我们设置2
    "alpha_c_init": 5,     # 颜色损失更新步长
    "device": "cuda:0",
    
    # === MFA 注意力权重 (用于计算梯度) ===
    # 目标：最大化特征差异，集中在路面区域
    "w_feature": 3.0,       # MSE 权重
    "w_cosine": 1.5,        # 余弦相似度权重
    "w_attn_focus": 4.5,    # 注意力加权权重 (破坏高关注区域)
    "attn_temp": 4.5        # 注意力温度
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

# =========================================================================
# === 可微分颜色空间函数 (Differential Color Functions) - Fixed Dtypes ===
# =========================================================================

def rgb2xyz(rgb_image, device):
    # [Fix] 强制输入为 float32，防止 BFloat16 导致的类型不匹配和精度问题
    rgb_image = rgb_image.float()
    
    # [Fix] 强制变换矩阵为 float32
    mt = torch.tensor([[0.4124, 0.3576, 0.1805], 
                   [0.2126, 0.7152, 0.0722],
                   [0.0193, 0.1192, 0.9504]], dtype=torch.float32).to(device)
                   
    mask1 = (rgb_image > 0.0405).float()
    mask1_no = 1 - mask1
    temp_img = mask1 * (((rgb_image + 0.055 ) / 1.055 ) ** 2.4)
    temp_img = temp_img + mask1_no * (rgb_image / 12.92)    
    temp_img = 100 * temp_img

    # matmul: float32 * float32
    res = torch.matmul(mt, temp_img.permute(1, 0, 2,3).contiguous().view(3, -1)).view(3, rgb_image.size(0),rgb_image.size(2), rgb_image.size(3)).permute(1, 0, 2,3)
    return res

def xyz_lab(xyz_image, device):
    xyz_image = xyz_image.float() # Ensure float32
    mask_value_0 = (xyz_image == 0).float().to(device)
    mask_value_0_no = 1 - mask_value_0
    xyz_image = xyz_image + 0.0001 * mask_value_0
    mask1 = (xyz_image > 0.008856).float()     
    mask1_no = 1 - mask1
    res = mask1 * (xyz_image) ** (1 / 3)
    res = res + mask1_no * ((7.787 * xyz_image) + (16 / 116))
    res = res * mask_value_0_no
    return res    

def rgb2lab_diff(rgb_image, device):
    '''
    Function to convert a batch of image tensors from RGB space to CIELAB space.
    '''
    rgb_image = rgb_image.to(device).float() # Ensure float32
    xyz_image = rgb2xyz(rgb_image, device)
    
    xn = 95.0489
    yn = 100.0
    zn = 108.8840
    
    x = xyz_image[:,0, :, :]
    y = xyz_image[:,1, :, :]
    z = xyz_image[:,2, :, :]

    L = 116 * xyz_lab(y / yn, device) - 16
    a = 500 * (xyz_lab(x / xn, device) - xyz_lab(y / yn, device))
    b = 200 * (xyz_lab(y / yn, device) - xyz_lab(z / zn, device))
    
    res = torch.stack([L, a, b], dim=1)
    return res

def degrees(n): return n * (180. / np.pi)
def radians(n): return n * (np.pi / 180.)

def hpf_diff(x, y):
    x = x.float(); y = y.float()
    mask1 = ((x == 0) * (y == 0)).float()
    mask1_no = 1 - mask1

    tmphp = degrees(torch.atan2(x * mask1_no, y * mask1_no))
    tmphp1 = tmphp * (tmphp >= 0).float()
    tmphp2 = (360 + tmphp) * (tmphp < 0).float()

    return tmphp1 + tmphp2

def dhpf_diff(c1, c2, h1p, h2p):
    mask1  = ((c1 * c2) == 0).float()
    mask1_no  = 1 - mask1
    res1 = (h2p - h1p) * mask1_no * (torch.abs(h2p - h1p) <= 180).float()
    res2 = ((h2p - h1p) - 360) * ((h2p - h1p) > 180).float() * mask1_no
    res3 = ((h2p - h1p) + 360) * ((h2p - h1p) < -180).float() * mask1_no
    return res1 + res2 + res3

def ahpf_diff(c1, c2, h1p, h2p):
    mask1 = ((c1 * c2) == 0).float()
    mask1_no = 1 - mask1
    mask2 = (torch.abs(h2p - h1p) <= 180).float()
    mask2_no = 1 - mask2
    mask3 = (torch.abs(h2p + h1p) < 360).float()
    mask3_no = 1 - mask3

    res1 = (h1p + h2p) * mask1_no * mask2
    res2 = (h1p + h2p + 360.) * mask1_no * mask2_no * mask3 
    res3 = (h1p + h2p - 360.) * mask1_no * mask2_no * mask3_no
    res = (res1 + res2 + res3) + (res1 + res2 + res3) * mask1
    return res * 0.5

def ciede2000_diff(lab1, lab2, device):
    '''
    CIEDE2000 metric to claculate the color distance map
    '''
    # [Fix] 确保所有计算都在 float32 下进行
    lab1 = lab1.to(device).float()
    lab2 = lab2.to(device).float()
       
    L1 = lab1[:,0,:,:]
    A1 = lab1[:,1,:,:]
    B1 = lab1[:,2,:,:]
    L2 = lab2[:,0,:,:]
    A2 = lab2[:,1,:,:]
    B2 = lab2[:,2,:,:]   
    kL = 1; kC = 1; kH = 1
    
    mask_value_0_input1 = ((A1 == 0) * (B1 == 0)).float()
    mask_value_0_input2 = ((A2 == 0) * (B2 == 0)).float()
    mask_value_0_input1_no = 1 - mask_value_0_input1
    mask_value_0_input2_no = 1 - mask_value_0_input2
    B1 = B1 + 0.0001 * mask_value_0_input1
    B2 = B2 + 0.0001 * mask_value_0_input2 
    
    C1 = torch.sqrt((A1 ** 2.) + (B1 ** 2.))
    C2 = torch.sqrt((A2 ** 2.) + (B2 ** 2.))   
   
    aC1C2 = (C1 + C2) / 2.
    G = 0.5 * (1. - torch.sqrt((aC1C2 ** 7.) / ((aC1C2 ** 7.) + (25 ** 7.))))
    a1P = (1. + G) * A1
    a2P = (1. + G) * A2
    c1P = torch.sqrt((a1P ** 2.) + (B1 ** 2.))
    c2P = torch.sqrt((a2P ** 2.) + (B2 ** 2.))

    h1P = hpf_diff(B1, a1P)
    h2P = hpf_diff(B2, a2P)
    h1P = h1P * mask_value_0_input1_no
    h2P = h2P * mask_value_0_input2_no 
    
    dLP = L2 - L1
    dCP = c2P - c1P
    dhP = dhpf_diff(C1, C2, h1P, h2P)
    dHP = 2. * torch.sqrt(c1P * c2P) * torch.sin(radians(dhP) / 2.)
    mask_0_no = 1 - torch.max(mask_value_0_input1, mask_value_0_input2)
    dHP = dHP * mask_0_no

    aL = (L1 + L2) / 2.
    aCP = (c1P + c2P) / 2.
    aHP = ahpf_diff(C1, C2, h1P, h2P)
    T = 1. - 0.17 * torch.cos(radians(aHP - 39)) + 0.24 * torch.cos(radians(2. * aHP)) + 0.32 * torch.cos(radians(3. * aHP + 6.)) - 0.2 * torch.cos(radians(4. * aHP - 63.))
    dRO = 30. * torch.exp(-1. * (((aHP - 275.) / 25.) ** 2.))
    rC = torch.sqrt((aCP ** 7.) / ((aCP ** 7.) + (25. ** 7.)))    
    sL = 1. + ((0.015 * ((aL - 50.) ** 2.)) / torch.sqrt(20. + ((aL - 50.) ** 2.)))
    
    sC = 1. + 0.045 * aCP
    sH = 1. + 0.015 * aCP * T
    rT = -2. * rC * torch.sin(radians(2. * dRO))

    res_square = ((dLP / (sL * kL)) ** 2.) + ((dCP / (sC * kC)) ** 2.) * mask_0_no + ((dHP / (sH * kH)) ** 2.) * mask_0_no + rT * (dCP / (sC * kC)) * (dHP / (sH * kH)) * mask_0_no
    mask_0 = (res_square <= 0).float()
    mask_0_no = 1 - mask_0
    res_square = res_square + 0.0001 * mask_0    
    res = torch.sqrt(res_square)
    res = res * mask_0_no
    return res

# ================= SimLingo 特征提取器 =================
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

# [灰盒攻击关键] 道路加权特征损失
def get_road_focused_feature_loss(feat_adv, feat_clean, road_ratio=0.45):
    """
    计算对抗特征与原始特征的加权 MSE 损失。
    在 PerC-AL 攻击中，我们的目标是最大化此距离。
    """
    if feat_adv.dim() != 3: return F.mse_loss(feat_adv, feat_clean)
    
    B, N, D = feat_adv.shape
    
    # 尝试还原空间结构
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
    
    # 道路区域权重加倍 (通常位于图像下方)
    weight = torch.ones(h_tokens, device=feat_adv.device)
    road_start = int(h_tokens * (1.0 - road_ratio))
    weight[road_start:] = 3.0 
    weight = weight.view(1, h_tokens, 1, 1)
    
    diff = (patch_adv_2d - patch_clean_2d) ** 2
    weighted_diff = diff * weight
    loss = weighted_diff.mean()
    
    if cls_adv is not None:
        loss = loss + F.mse_loss(cls_adv, cls_clean)
    return loss

def quantization(x):
   """quantize the continus image tensors into 255 levels (8 bit encoding)"""
   x_quan=torch.round(x*255)/255 
   return x

# ================= 主程序 (PerC-AL Version) =================

class SimLingoPerCALAttacker:
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
        
        self.simlingo_extractor = SimLingoFeatureExtractor(self.model)
        
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

    def prepare_base_components(self, image_path):
        # 1. 获取所有数据
        real_speed, cmd_id, tl_state, tp1, tp2, gt_route = self.load_measurements(image_path)
        cmd_text = get_cmd_text(cmd_id)
        
        # 2. 图像加载
        raw_image = Image.open(image_path).convert('RGB')
        raw_image = raw_image.resize((448, 448))
        image_sizes = torch.tensor([[448, 448]]).to(self.device) 
        
        # 3. 构建 Prompt
        prompt_tp = "Target waypoint: <TARGET_POINT><TARGET_POINT>."
        base_prompt = (
            f"Current speed: {real_speed:.2f} m/s. "
            f"Command: {cmd_text}. "
            f"Traffic light: {tl_state}. "
            f"{prompt_tp} Predict the waypoints."
        )
        
        # 4. Tokenization
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
        }, raw_image, gt_route, tp1

    def differentiable_processing(self, image_01):
        """
        Input: [B, 3, H, W] in [0, 1]
        Output: SimLingo compatible normalized 6D tensor
        """
        img_448 = F.interpolate(image_01, size=(448, 448), mode='bilinear', align_corners=False)
        mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
        # return pixel values [B, T, N, C, H, W] for SimLingo (T=1, N=2 cameras typically, we dup)
        return ((img_448 - mean) / std).unsqueeze(1).unsqueeze(1).repeat(1, 1, 2, 1, 1, 1).bfloat16()
    
    def get_vision_features(self, img_tensor_448_normed, return_attn=False):
        # 从 SimLingo 6D tensor 还原到 4D
        pixel_values = img_tensor_448_normed[:, 0, 0, :, :, :] # [B, 3, 448, 448]
        
        outputs = None; attn_weights = None
        try:
            outputs = self.vision_model(pixel_values=pixel_values, output_attentions=return_attn)
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
            elif isinstance(outputs, tuple):
                for item in outputs:
                    if isinstance(item, (list, tuple)) and len(item) > 0:
                        if isinstance(item[-1], torch.Tensor) and item[-1].dim() == 4:
                            attn_weights = item[-1]; break
            return features, attn_weights
        
        return features

    def run_percal_attack(self, image_path, output_dir, fid_clean_dir, fid_adv_dir, file_prefix=""):
        # 1. 准备数据 (Data Prep)
        base_params, raw_pil, gt_route, target_point = self.prepare_base_components(image_path)
        img_np = np.array(raw_pil).astype(np.float32) / 255.0 
        # [0,1] Range Input
        inputs = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).to(self.device).float() 
        batch_size = inputs.shape[0]

        # [Standard PerC-AL] Pre-calculate LAB for Color Loss Anchor
        inputs_LAB = rgb2lab_diff(inputs, self.device)
        
        # [Fix] Define variables outside to ensure scope visibility
        speed_clean = None
        route_clean = None
        clean_sim_feats = []

        # Extract Clean Features (Target)
        with torch.no_grad():
            clean_sim_inputs = self.differentiable_processing(inputs)
            clean_vision_emb, clean_vis_attn = self.get_vision_features(clean_sim_inputs, return_attn=True)
            clean_vision_emb = clean_vision_emb.detach()
            if clean_vis_attn is not None:
                clean_vis_attn = clean_vis_attn.detach().float()
            
            # Full inference for metrics and clean baseline
            self.simlingo_extractor.clear()
            clean_full_input = DrivingInput(camera_images=clean_sim_inputs, **base_params)
            
            # [Fix] Capture clean outputs here
            speed_clean_out, route_clean_out, _ = self.model(clean_full_input)
            speed_clean = speed_clean_out.detach().clone()
            route_clean = route_clean_out.detach().clone()
            
            clean_sim_feats = [f.clone() for f in self.simlingo_extractor.features]

        # 2. 参数初始化 (Parameter Init - Matching perc_al.py)
        alpha_l_init = ATTACK_PARAMS['alpha_l_init']
        alpha_c_init = ATTACK_PARAMS['alpha_c_init']
        max_iterations = ATTACK_PARAMS['max_iterations']
        
        alpha_l_min = alpha_l_init / 100
        alpha_c_min = alpha_c_init / 10
        
        # [Necessary Adaptation] Random init to break zero-gradient symmetry for Feature Loss
        delta = (torch.rand_like(inputs) * 2 - 1) * 1e-5
        delta.requires_grad_(True)
        
        # 3. 攻击循环 (Optimization Loop)
        for i in range(max_iterations):
            # [Standard PerC-AL] Cosine Annealing
            alpha_c = alpha_c_min + 0.5 * (alpha_c_init - alpha_c_min) * (1 + cos(i / max_iterations * pi))
            alpha_l = alpha_l_min + 0.5 * (alpha_l_init - alpha_l_min) * (1 + cos(i / max_iterations * pi))
            
            # --- Phase 1: Task Loss Update ---
            if delta.grad is not None: delta.grad.zero_()
            
            # Forward
            current_inputs = inputs + delta
            adv_sim_inputs = self.differentiable_processing(current_inputs)
            
            # [Suggestion Applied] Force float32 for precision stability
            adv_vision_emb, adv_vis_attn = self.get_vision_features(adv_sim_inputs, return_attn=True)
            adv_vision_emb = adv_vision_emb.float()
            clean_vision_emb_f32 = clean_vision_emb.float()
            if adv_vis_attn is not None:
                adv_vis_attn = adv_vis_attn.float()
            
            # Calculate Loss with MFA components
            # 1. Road-focused feature loss (standard distance)
            feature_dist = get_road_focused_feature_loss(adv_vision_emb, clean_vision_emb_f32)
            
            # 2. Attention-weighted feature loss (focus on high-attention regions)
            loss_attn = get_attention_weighted_feature_loss(adv_vision_emb, clean_vision_emb_f32, clean_vis_attn, temperature=ATTACK_PARAMS['attn_temp'])
            
            # 3. Cosine similarity loss (prevent feature alignment)
            adv_flat = adv_vision_emb.view(adv_vision_emb.size(0), -1)
            clean_flat = clean_vision_emb_f32.view(clean_vision_emb_f32.size(0), -1)
            loss_cos = F.cosine_similarity(adv_flat, clean_flat, dim=-1).mean()
            
            # Combine all losses: maximize distance => minimize (-1 * distance)
            total_objective = (ATTACK_PARAMS['w_feature'] * feature_dist + 
                             ATTACK_PARAMS['w_attn_focus'] * loss_attn - 
                             ATTACK_PARAMS['w_cosine'] * loss_cos)
            loss = -1.0 * total_objective
            loss.backward()
            
            # [Standard PerC-AL] Gradient Normalization & Update
            grad_a = delta.grad.clone()
            delta.grad.zero_()
            
            norm_grad = torch.norm(grad_a.reshape(batch_size, -1), dim=1) + 1e-8
            normalized_grad = (grad_a.permute(1,2,3,0) / norm_grad).permute(3,0,1,2)
            
            # Apply Task Update
            delta.data = delta.data - alpha_l * normalized_grad

            # --- Phase 2: Color Loss Update ---
            # [Standard PerC-AL] Recalculate color on updated delta
            current_adv_img = (inputs + delta).clamp(0, 1) # Clamp only for calculation
            current_adv_lab = rgb2lab_diff(current_adv_img, self.device)
            
            d_map = ciede2000_diff(inputs_LAB, current_adv_lab, self.device)
            if d_map.dim() == 3: d_map = d_map.unsqueeze(1)
            
            color_dis = torch.norm(d_map.reshape(batch_size, -1), dim=1)
            color_loss = color_dis.sum()
            color_loss.backward()
            
            # [Standard PerC-AL] Gradient Normalization & Update
            grad_color = delta.grad.clone()
            delta.grad.zero_()
            
            norm_grad_color = torch.norm(grad_color.reshape(batch_size, -1), dim=1) + 1e-8
            normalized_grad_color = (grad_color.permute(1,2,3,0) / norm_grad_color).permute(3,0,1,2)
            
            # Apply Color Update
            delta.data = delta.data - alpha_c * normalized_grad_color
            
            # --- Phase 3: Constraint & Quantization ---
            # [Standard PerC-AL] Clamp to valid image range
            delta.data = (inputs + delta.data).clamp(0, 1) - inputs
            
            # [Standard PerC-AL] Quantization step
            X_adv_round = quantization(inputs + delta.data)

        # 4. Final Output Generation
        with torch.no_grad():
            # Use the final quantized result
            final_adv_img = X_adv_round.clamp(0, 1)
            
            # --- Metrics Calculation ---
            self.simlingo_extractor.clear()
            final_input = DrivingInput(camera_images=self.differentiable_processing(final_adv_img), **base_params)
            speed_final, route_final, _ = self.model(final_input)
            adv_sim_feats = [f.clone() for f in self.simlingo_extractor.features]
            
            sim_drift = 0.0
            if len(adv_sim_feats) > 0 and len(clean_sim_feats) > 0:
                sim_drift = F.mse_loss(adv_sim_feats[-1].float(), clean_sim_feats[-1].float()).item() * 1000

            # --- 详细误差拆解 ---
            clean_end = route_clean[0, -1] 
            adv_end = route_final[0, -1]
            
            # 1. 总体偏移 (Shift)
            f_dist = torch.norm(route_final.float() - route_clean.float(), p=2).item()
            f_spd_err = torch.norm(speed_final.float() - speed_clean.float(), p=2).item()
            
            # 2. 横向误差 (Adv vs Clean)
            lat_error = abs(adv_end[1].item() - clean_end[1].item())
            
            # 3. 纵向误差 (Adv vs Clean)
            lon_error = abs(adv_end[0].item() - clean_end[0].item())
            
            # 4. 原始路程长度 (用于 Tan 分母)
            clean_route_len = abs(clean_end[0].item())
            
            is_success = 1.0 if f_dist >= 1.0 else 0.0
            
            recon_clean_img_01 = inputs
            final_img_01 = final_adv_img
            
            road_lpips = self.lpips_vgg(final_img_01.float()*2-1, recon_clean_img_01.float()*2-1).mean().item()
            road_ssim = self.ssim_loss(final_img_01.float()*2-1, recon_clean_img_01.float()*2-1).item()

            # 保存图片
            base_name = os.path.basename(image_path).split('.')[0]
            if file_prefix:
                save_name = f"{file_prefix}_{base_name}.jpg"
            else:
                save_name = f"{base_name}.jpg"

            def save_safe_jpg(tensor, path):
                ndarr = tensor[0].mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
                Image.fromarray(ndarr).save(path, format='JPEG', quality=95)

            save_safe_jpg(recon_clean_img_01.cpu().float(), os.path.join(fid_clean_dir, save_name))
            save_safe_jpg(final_img_01.cpu().float(), os.path.join(fid_adv_dir, save_name))
            
            self.save_unified_visualization(recon_clean_img_01, final_img_01, route_clean, route_final, gt_route, target_point, speed_clean, speed_final, output_dir, base_name)

        # 返回值顺序: 
        # 0:Shift, 1:Spd, 2:Drift, 3:LPIPS, 4:SSIM, 
        # 5:Lat_Err, 6:Lon_Err, 7:Clean_Len, 8:Succ
        return f_dist, f_spd_err, sim_drift, road_lpips, road_ssim, lat_error, lon_error, clean_route_len, is_success
    
    def save_unified_visualization(self, recon_01, adv_01, route_clean, route_adv, 
                                   gt_route, target_point, speed_clean, speed_adv, save_dir, img_name):
        try:
            recon_np = recon_01[0].permute(1, 2, 0).cpu().float().numpy()
            adv_np = adv_01[0].permute(1, 2, 0).cpu().float().numpy()
            recon_np = np.clip(recon_np, 0, 1)
            adv_np = np.clip(adv_np, 0, 1)
            # 增强显示噪声，PerC-AL 噪声通常很小
            diff_vis = np.clip((adv_np - recon_np) * 50.0 + 0.5, 0, 1) 
            
            c_pts = route_clean[0].detach().cpu().float().numpy()
            a_pts = route_adv[0].detach().cpu().float().numpy()
            sc_val = speed_clean.float().mean().item() if speed_clean.numel() > 1 else speed_clean.item()
            sa_val = speed_adv.float().mean().item() if speed_adv.numel() > 1 else speed_adv.item()

            # 创建子目录
            clean_img_dir = os.path.join(save_dir, "clean_images")
            adv_img_dir = os.path.join(save_dir, "adv_images")
            noise_img_dir = os.path.join(save_dir, "noise_images")
            traj_img_dir = os.path.join(save_dir, "trajectory_images")
            os.makedirs(clean_img_dir, exist_ok=True)
            os.makedirs(adv_img_dir, exist_ok=True)
            os.makedirs(noise_img_dir, exist_ok=True)
            os.makedirs(traj_img_dir, exist_ok=True)

            # 0. 保存干净图像 (SVG 格式，嵌入位图 448x448)
            clean_path = os.path.join(clean_img_dir, f"{img_name}_clean.svg")
            fig_clean = plt.figure(figsize=(4.48, 4.48), dpi=100)
            plt.imshow(recon_np)
            plt.axis('off')
            plt.tight_layout(pad=0)
            plt.savefig(clean_path, format='svg', bbox_inches='tight', pad_inches=0)
            plt.close(fig_clean)

            # 1. 保存攻击后图像 (SVG 格式)
            adv_path = os.path.join(adv_img_dir, f"{img_name}_adv.svg")
            fig_adv = plt.figure(figsize=(4.48, 4.48), dpi=100)
            plt.imshow(adv_np)
            plt.axis('off')
            plt.tight_layout(pad=0)
            plt.savefig(adv_path, format='svg', bbox_inches='tight', pad_inches=0)
            plt.close(fig_adv)

            # 2. 保存噪声图像 (SVG 格式)
            noise_path = os.path.join(noise_img_dir, f"{img_name}_noise.svg")
            fig_noise = plt.figure(figsize=(4.48, 4.48), dpi=100)
            plt.imshow(diff_vis)
            plt.axis('off')
            plt.tight_layout(pad=0)
            plt.savefig(noise_path, format='svg', bbox_inches='tight', pad_inches=0)
            plt.close(fig_noise)

            # 3. 保存轨迹图像 (SVG 矢量格式)
            traj_path = os.path.join(traj_img_dir, f"{img_name}_traj.svg")
            fig_traj = plt.figure(figsize=(8, 8))
            ax = fig_traj.add_subplot(111)
            
            if gt_route is not None:
                ax.plot(gt_route[:, 1], gt_route[:, 0], 'g-', alpha=0.3, linewidth=5, label='Ground Truth')
            ax.plot(c_pts[:, 1], c_pts[:, 0], 'b-o', markersize=5, linewidth=2, label=f'Clean ({sc_val:.1f} m/s)')
            ax.plot(a_pts[:, 1], a_pts[:, 0], 'r--^', markersize=6, linewidth=2, label=f'Adv ({sa_val:.1f} m/s)')
            
            if target_point is not None:
                tp_np = np.array(target_point)
                tx, ty = (tp_np[0], tp_np[1]) if tp_np.ndim == 1 else (tp_np[0, 0], tp_np[0, 1])
                ax.scatter(ty, tx, c='gold', marker='*', s=300, edgecolors='black', label='Target', zorder=5)

            ax.set_xlim(-12, 12)
            ax.set_ylim(-2, 40)
            ax.set_xlabel("Lateral (m)")
            ax.set_ylabel("Forward (m)")
            ax.grid(True, linestyle=':', alpha=0.6)
            ax.legend(loc='upper right')
            ax.set_title("Trajectory (PerC-AL Attack)", fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig(traj_path, format='svg', bbox_inches='tight')
            plt.close(fig_traj)

        except Exception as e:
            print(f"Vis error: {e}")

def get_next_log_filename(output_dir):
    existing_logs = [f for f in os.listdir(output_dir) if f.startswith("attack_log_") and f.endswith(".txt")]
    if not existing_logs:
        return os.path.join(output_dir, "attack_log_1.txt")
    
    max_index = max(int(f.split("_")[2].split(".")[0]) for f in existing_logs)
    return os.path.join(output_dir, f"attack_log_{max_index + 1}.txt")

import shutil

# ================= MFA Loss Helper =================
def get_attention_weighted_feature_loss(feat_adv, feat_clean, attn_weights, temperature=2.0):
    """
    使用 Clean Attention Map 对 MSE Loss 进行加权。
    """
    if attn_weights is None:
        return F.mse_loss(feat_adv, feat_clean)

    # 适配 Attention 形状: [B, H, N, N] -> CLS token attention [B, N-1]
    if attn_weights.dim() == 4:
        # 取 CLS (token 0) 对所有 patch 的 attention，平均多头
        attn = attn_weights[:, :, 0, 1:].mean(dim=1)
    elif attn_weights.dim() == 3:
        attn = attn_weights[:, 0, 1:]
    else:
        return F.mse_loss(feat_adv, feat_clean)

    # 温度缩放，强化高权重区域
    attn = F.softmax(attn * temperature, dim=-1)

    # 假设 feat 是 [B, N, D], 其中 0 是 CLS
    if feat_adv.dim() == 3 and feat_adv.size(1) > attn.size(1):
        # 分离 CLS 和 Patch Tokens
        patch_adv = feat_adv[:, 1:, :]
        patch_clean = feat_clean[:, 1:, :]

        # 计算 Patch 维度的 MSE [B, N-1]
        diff = (patch_adv - patch_clean) ** 2
        diff_per_patch = diff.mean(dim=-1)

        # 加权
        weighted_loss = (diff_per_patch * attn).sum(dim=-1).mean()
        return weighted_loss
    else:
        return F.mse_loss(feat_adv, feat_clean)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True, 
                        help="Root directory containing multiple dataset folders")
    parser.add_argument("--output_dir", type=str, default="percal_allresults") 
    args = parser.parse_args()

    # 1. 确定本次运行的序号文件夹 (PerCAL-gray-N)
    base_dir = args.output_dir
    os.makedirs(base_dir, exist_ok=True)

    existing_folders = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    max_id = 0
    for folder in existing_folders:
        if folder.startswith("PerCAL-gray-"):
            try:
                idx = int(folder.split("-")[-1])
                if idx > max_id: max_id = idx
            except ValueError: continue
    
    new_folder_name = f"PerCAL-gray-{max_id + 1}"
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
    print("Grey-Box PerC-AL Config:")
    print(json.dumps(ATTACK_PARAMS, indent=2))
    
    attacker = SimLingoPerCALAttacker()
    
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
        
        images = sorted(glob.glob(os.path.join(current_rgb_dir, "*.jpg")))
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

        for img_path in images:
            t0 = time.time()
            try:
                res = attacker.run_percal_attack(img_path, current_out_dir, global_fid_clean, global_fid_adv, file_prefix=dataset_name)
                step_time = time.time() - t0
                
                lat_err = res[5]
                lon_err = res[6]
                clean_len = res[7]
                is_succ = res[8]
                
                # [新增] 提取单张图片的 LPIPS 和 SSIM
                current_lpips = res[3]
                current_ssim = res[4]
                
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
                
                # [修改] 更新 Print 语句，加入 LPIPS 和 SSIM
                print(f"[{dataset_name}][{os.path.basename(img_path)}] "
                      f"Shift: {res[0]:.2f}m (Lat:{lat_err:.2f}) | "
                      f"LPIPS: {current_lpips:.4f} | SSIM: {current_ssim:.4f} | "
                      f"Tan: {current_tan:.2f} | Time: {step_time:.2f}s")
                # [新增] 将当前指标添加到全局统计
                global_stats['shift'].append(res[0])
                global_stats['spd_err'].append(res[1])
                global_stats['sim_drift'].append(res[2])
                global_stats['road_lpips'].append(current_lpips)
                global_stats['road_ssim'].append(current_ssim)
                global_stats['global_tan'].append(current_tan)
                global_stats['success_rate'].append(is_succ)
                global_stats['avg_time'].append(step_time)

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