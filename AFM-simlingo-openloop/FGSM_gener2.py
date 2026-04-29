import os
# 禁用 Tokenizers 并行，防止死锁
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# 优化显存分配
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import sys
import glob
import json
import gzip
import argparse
import shutil
import math
import time
from datetime import timedelta
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import torch
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
import gc

# [引入] 评估库
try:
    import lpips
except ImportError:
    print("请先安装 lpips: pip install lpips")
    sys.exit(1)

try:
    from torch_fidelity import calculate_metrics
except ImportError:
    print("请先安装 torch-fidelity: pip install torch-fidelity")
    sys.exit(1)


# ================= 配置 =================
REPO_ROOT = "/home/pc/simlingo"
CHECKPOINT_PATH = f"{REPO_ROOT}/outputs/simlingo/checkpoints/epoch=013.ckpt/pytorch_model.pt"
CONFIG_PATH = f"{REPO_ROOT}/outputs/simlingo/.hydra/config.yaml"

# === 灰盒 FGSM (MFA风格) 参数 ===
ATTACK_PARAMS = {
    "epsilon": 8/255,       # 扰动强度
    "steps": 40,            # 迭代步数 (实为 PGD/I-FGSM)
    "step_size": 2/255,     # 单步步长
    "device": "cuda:1",
    
    # === MFA 损失权重 (用于计算梯度) ===
    "w_feature": 3.0,       # MSE 权重
    "w_cosine": 1.5,        # 余弦相似度权重
    "w_attn": 6.0,          # 注意力加权权重 (破坏高关注区域)
    "attn_temp": 6.0        # 注意力温度
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

# ================= 坐标转换工具 =================
def world_to_ego(route_global, ego_x, ego_y, ego_theta_rad):
    route_global = np.asarray(route_global, dtype=float)
    if route_global.ndim != 2 or route_global.shape[1] < 2:
        raise ValueError("route_global must be (N,2)")
    yaw = float(ego_theta_rad)
    diff = route_global - np.array([float(ego_x), float(ego_y)])
    dx = diff[:, 0]; dy = diff[:, 1]
    c, s = np.cos(yaw), np.sin(yaw)
    local_x = dx * c + dy * s
    local_y = -dx * s + dy * c
    return np.stack([local_x, local_y], axis=-1)

# ================= SimLingo 特征提取器 (Evaluation Only) =================
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
            if hasattr(simlingo_model, 'vision_model'): target_module = simlingo_model.vision_model
            elif hasattr(simlingo_model, 'model'): target_module = simlingo_model.model.vision_model
            if target_module: self.hooks.append(target_module.register_forward_hook(self.hook_fn))

    def hook_fn(self, module, input, output):
        data = None
        if isinstance(output, torch.Tensor): data = output
        elif hasattr(output, 'last_hidden_state'): data = output.last_hidden_state
        elif isinstance(output, tuple): data = output[0]
        if data is not None: self.features.append(data.detach().clone())

    def clear(self): self.features = []
    def remove(self): 
        for h in self.hooks: h.remove()

# ================= FGSM Grey-Box Attacker =================
class FGSM_GreyBox_Attacker:
    def __init__(self):
        self.device = torch.device(ATTACK_PARAMS["device"])
        print(f"Loading Config: {CONFIG_PATH}")
        with open(CONFIG_PATH, 'r') as f:
            self.cfg = OmegaConf.load(f)
        
        # Metrics Models
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

        print("Loading Model...")
        torch.set_default_dtype(torch.bfloat16)
        cache_dir = f"pretrained/{variant.split('/')[-1]}"
        self.model = hydra.utils.instantiate(
            self.cfg.model, cfg_data_module=self.cfg.data_module,
            processor=self.processor, cache_dir=cache_dir, _recursive_=False
        ).to(self.device)
        
        state_dict = torch.load(CHECKPOINT_PATH, map_location=self.device)
        if 'state_dict' in state_dict: state_dict = state_dict['state_dict']
        new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        self.model.load_state_dict(new_state_dict, strict=False)
        self.model.eval()

        # === [Grey-Box] Vision Model Hunter ===
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
        
        # 冻结所有参数
        for param in self.model.parameters(): param.requires_grad = False
        print(f">>> [Grey-Box] TARGET LOCKED: {self.vision_model.__class__.__name__}")

        self.intrinsics = get_camera_intrinsics(448, 448, fov=110).unsqueeze(0).to(self.device).float().view(1, 3, 3)
        self.extrinsics = get_camera_extrinsics().unsqueeze(0).to(self.device).float().view(1, 4, 4)

        # Init Feature Extractor (For Full Evaluation)
        self.simlingo_extractor = SimLingoFeatureExtractor(self.model)

    def differentiable_processing(self, image_01):
        """
        Input: [1, 3, 448, 448] tensor in [0, 1]
        Output: SimLingo compatible normalized 6D tensor
        """
        mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
        # Repeat to match SimLingo input structure [B, T, N, C, H, W] -> T=1, N=2 (views)
        return ((image_01 - mean) / std).unsqueeze(1).unsqueeze(1).repeat(1, 1, 2, 1, 1, 1).bfloat16()

    # [新增] 仅 Vision Model 前向传播
    def forward_vision_only(self, img_tensor_normalized, return_attn=False):
        # 展平维度 [1, 1, 2, 3, 448, 448] -> [2, 3, 448, 448]
        if img_tensor_normalized.dim() == 6:
            B, T, N, C, H, W = img_tensor_normalized.shape
            pixel_values = img_tensor_normalized.view(B * T * N, C, H, W)
        else:
            pixel_values = img_tensor_normalized

        outputs = None; attn_weights = None
        try:
            outputs = self.vision_model(pixel_values=pixel_values.bfloat16(), output_attentions=return_attn)
        except TypeError:
            outputs = self.vision_model(pixel_values.bfloat16())

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

    def load_measurements(self, image_path):
        """
        加载 Measurements 和 Boxes 数据
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
                
                if 'route' in data:
                    route = np.array(data['route'], dtype=float)
                    if route.ndim >= 2: route = route[:, :2]
                    if np.max(np.abs(route)) > 1000.0:
                        gt_local = world_to_ego(route, ego_x, ego_y, ego_theta)
                    else:
                        gt_local = route
                
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
        real_speed, cmd_id, tl_state, tp1, tp2, gt_route = self.load_measurements(image_path)
        cmd_text = get_cmd_text(cmd_id)
        
        # === [Modified for Center Crop] ===
        raw_image = Image.open(image_path).convert('RGB')
        orig_W, orig_H = raw_image.size
        
        # 1. 计算 Center Crop 坐标
        crop_size = min(orig_W, orig_H)
        left = (orig_W - crop_size) // 2
        top = (orig_H - crop_size) // 2
        right = left + crop_size
        bottom = top + crop_size
        crop_coords = (left, top, right, bottom, crop_size)
        
        # 2. 裁剪
        img_cropped = raw_image.crop((left, top, right, bottom))
        
        # 3. Resize to 448x448
        raw_image_448 = img_cropped.resize((448, 448))
        
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
        tp_tensor = torch.from_numpy(tp1).bfloat16().unsqueeze(0).to(self.device)
        
        return {
            "image_sizes": image_sizes, 
            "camera_intrinsics": self.intrinsics, 
            "camera_extrinsics": self.extrinsics, 
            "vehicle_speed": torch.tensor([[real_speed]]).to(self.device).bfloat16(), 
            "target_point": tp_tensor, 
            "prompt": ll, 
            "prompt_inference": ll
        }, raw_image_448, gt_route, tp1, raw_image, crop_coords
        
    def save_unified_visualization(self, clean_tensor_01, adv_tensor_01, route_clean, route_adv, 
                                   gt_route, target_point,
                                   speed_clean, speed_adv, save_dir, img_name):
        try:
            # clean_tensor_01: [1, 3, 448, 448]
            clean_np = clean_tensor_01[0].detach().cpu().float().numpy().transpose(1, 2, 0)
            adv_np = adv_tensor_01[0].detach().cpu().float().numpy().transpose(1, 2, 0)
            
            clean_np = np.clip(clean_np, 0, 1); adv_np = np.clip(adv_np, 0, 1)
            noise_vis = np.clip(np.abs(adv_np - clean_np) * 15.0, 0, 1)

            clean_pts = route_clean[0].detach().cpu().float().numpy()
            adv_pts = route_adv[0].detach().cpu().float().numpy()
            s_clean = speed_clean.mean().item(); s_adv = speed_adv.mean().item()

            fig = plt.figure(figsize=(12, 10))
            gs = fig.add_gridspec(2, 2)
            ax1 = fig.add_subplot(gs[0, 0]); ax1.imshow(clean_np); ax1.set_title("Clean (Center Crop)"); ax1.axis('off')
            ax2 = fig.add_subplot(gs[0, 1]); ax2.imshow(adv_np); ax2.set_title("FGSM (Center Crop)"); ax2.axis('off')
            ax3 = fig.add_subplot(gs[1, 0]); ax3.imshow(noise_vis); ax3.set_title("Noise Diff (x15)"); ax3.axis('off')
            
            ax4 = fig.add_subplot(gs[1, 1])
            if gt_route is not None:
                viz_len = min(len(gt_route), 20)
                ax4.plot(gt_route[:viz_len, 1], gt_route[:viz_len, 0], 'g-o', linewidth=2, label='GT')
            ax4.plot(clean_pts[:, 1], clean_pts[:, 0], 'b-x', linewidth=2, label=f'Clean ({s_clean:.1f} m/s)')
            ax4.plot(adv_pts[:, 1], adv_pts[:, 0], 'r--^', linewidth=2, label=f'FGSM ({s_adv:.1f} m/s)')
            if target_point is not None: 
                t_np = target_point if isinstance(target_point, np.ndarray) else target_point.cpu().numpy()
                if t_np.ndim > 1: t_np = t_np[0]
                ax4.plot(t_np[1], t_np[0], 'r*', markersize=18, label='Target', zorder=10)
            ax4.plot(0, 0, 'k^', markersize=12, label='Ego')
            l2_dist = np.linalg.norm(clean_pts-adv_pts)
            ax4.set_title(f"Attack Result\nRoute Dev: {l2_dist:.2f}m | Speed Diff: {abs(s_clean-s_adv):.2f} m/s")
            ax4.set_xlabel("Lateral (m)"); ax4.set_ylabel("Longitudinal (m)")
            ax4.legend(loc='upper right'); ax4.grid(True, linestyle='--', alpha=0.5)
            ax4.set_aspect('equal'); ax4.set_xlim(-10, 10); ax4.set_ylim(-2, 30)

            plt.tight_layout(); plt.savefig(os.path.join(save_dir, f"{img_name}_fgsm_final.png"), dpi=100); plt.close()
        except Exception as e: print(f"Viz Error: {e}")

    def run_attack(self, image_path, output_dir, fid_clean_dir, fid_adv_dir, file_prefix=""):
        # 1. 准备数据 (获取 Crop 后的 448x448 图像 和 原始全图)
        base_params, raw_pil_448, gt_route, target_point, raw_pil_full, crop_coords = self.prepare_base_components(image_path)
        
        # 将 raw_pil_448 转换为 Tensor [0, 1] [1, 3, 448, 448]
        img_np_448 = np.array(raw_pil_448).astype(np.float32) / 255.0
        inputs = torch.from_numpy(img_np_448).permute(2, 0, 1).unsqueeze(0).to(self.device).float()
        
        # 2. Clean Prediction (SimLingo)
        # 构造 SimLingo 输入 (6D)
        clean_sim_inputs = self.differentiable_processing(inputs)
        
        clean_input = DrivingInput(camera_images=clean_sim_inputs, **base_params)
        self.simlingo_extractor.clear()
        with torch.no_grad():
            speed_clean, route_clean, _ = self.model(clean_input)
            speed_clean = speed_clean.float(); route_clean = route_clean.float()
        clean_sim_feats_eval = [f.clone() for f in self.simlingo_extractor.features]

        # 3. MFA Anchor (Vision Model Features)
        with torch.no_grad():
            clean_vis_feats, clean_vis_attn = self.forward_vision_only(clean_sim_inputs, return_attn=True)
            clean_vis_feats = clean_vis_feats.float()
            if clean_vis_attn is not None: clean_vis_attn = clean_vis_attn.float()

        # 4. FGSM/PGD Setup
        # 我们针对 inputs (0-1) 进行优化
        origin_image = inputs.clone().detach()
        delta = torch.zeros_like(origin_image).uniform_(-1e-6, 1e-6)
        delta.requires_grad = True 
        
        # 5. Attack Step (Iterative)
        for step in range(ATTACK_PARAMS['steps']):
            # 加扰动并 Clip
            x_adv_01 = torch.clamp(origin_image + delta, 0, 1)
            
            # 转为 SimLingo 输入格式 (自动处理归一化)
            x_adv_sim = self.differentiable_processing(x_adv_01)
            
            # 仅通过 Vision Model 计算梯度
            adv_vis_feats = self.forward_vision_only(x_adv_sim, return_attn=False).float()
            
            loss_mse = F.mse_loss(adv_vis_feats, clean_vis_feats)
            adv_flat = adv_vis_feats.view(adv_vis_feats.size(0), -1)
            clean_flat = clean_vis_feats.view(clean_vis_feats.size(0), -1)
            loss_cos = F.cosine_similarity(adv_flat, clean_flat, dim=-1).mean()
            loss_attn = get_attention_weighted_feature_loss(adv_vis_feats, clean_vis_feats, clean_vis_attn, temperature=ATTACK_PARAMS['attn_temp'])
            
            total_objective = ATTACK_PARAMS['w_feature'] * loss_mse + ATTACK_PARAMS['w_attn'] * loss_attn - ATTACK_PARAMS['w_cosine'] * loss_cos
            
            if delta.grad is not None: delta.grad.zero_()
            total_objective.backward()
            
            # PGD Update
            grad = delta.grad
            delta_update = ATTACK_PARAMS['step_size'] * grad.sign()
            delta.data = torch.clamp(delta + delta_update, -ATTACK_PARAMS['epsilon'], ATTACK_PARAMS['epsilon'])
            delta.data = torch.clamp(origin_image + delta.data, 0, 1) - origin_image

        # 6. Final Evaluation & Paste Back
        with torch.no_grad():
            final_img_01 = torch.clamp(origin_image + delta, 0, 1) # [1, 3, 448, 448]
            x_final_sim = self.differentiable_processing(final_img_01)
            
            self.simlingo_extractor.clear()
            final_input = DrivingInput(camera_images=x_final_sim, **base_params)
            speed_final, route_final, _ = self.model(final_input)
            adv_sim_feats_eval = [f.clone() for f in self.simlingo_extractor.features]
            
            # --- Metrics ---
            clean_end = route_clean[0, -1] 
            adv_end = route_final[0, -1]
            
            final_dist = torch.norm(route_final.float() - route_clean, p=2).item()
            final_spd_diff = torch.norm(speed_final.float() - speed_clean, p=2).item()
            lat_error = abs(adv_end[1].item() - clean_end[1].item())
            lon_error = abs(adv_end[0].item() - clean_end[0].item())
            clean_route_len = abs(clean_end[0].item())
            is_success = 1.0 if final_dist >= 1.0 else 0.0

            road_lpips = self.lpips_vgg(final_img_01 * 2 - 1, origin_image * 2 - 1).mean().item()
            road_ssim = self.ssim_loss(final_img_01 * 2 - 1, origin_image * 2 - 1).item()
            
            sim_drift = 0.0
            if len(adv_sim_feats_eval) > 0 and len(clean_sim_feats_eval) > 0:
                sim_drift = F.mse_loss(adv_sim_feats_eval[-1].float(), clean_sim_feats_eval[-1].float()).item() * 1000

            # ================= [关键修改] Paste Back 逻辑 =================
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            if file_prefix:
                save_name = f"{file_prefix}_{base_name}.png"
            else:
                save_name = f"{base_name}.png"

            # 1. 准备原始背景 (Clean Canvas)
            # 加载原图全尺寸转 Tensor [0, 1]
            raw_tensor = T.ToTensor()(raw_pil_full).unsqueeze(0).to(self.device).float()
            
            # 2. 解包 Crop 坐标
            left, top, right, bottom, crop_size = crop_coords
            
            # 3. 放大生成的 448x448 对抗样本到 crop_size
            img_adv_upscaled = F.interpolate(final_img_01, size=(crop_size, crop_size), mode='bilinear', align_corners=False)
            img_clean_upscaled = F.interpolate(inputs, size=(crop_size, crop_size), mode='bilinear', align_corners=False)
            
            # 4. 构造用于保存的 Full Canvas
            final_full_adv = raw_tensor.clone()
            final_full_clean = raw_tensor.clone()
            
            # 贴回中间位置
            final_full_adv[:, :, top:bottom, left:right] = img_adv_upscaled
            final_full_clean[:, :, top:bottom, left:right] = img_clean_upscaled # 对应 Crop 区域被 resize 过的 clean version

            def save_safe_png(tensor, path):
                if tensor.dim() == 4: tensor = tensor[0]
                ndarr = tensor.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
                Image.fromarray(ndarr).save(path, format='PNG')

            # 1. 保存到 output_dir (给 TransFuser 用的)
            save_safe_png(final_full_adv, os.path.join(output_dir, save_name))
            
            # 2. 保存到 FID 目录
            save_safe_png(final_full_clean, os.path.join(fid_clean_dir, save_name))
            save_safe_png(final_full_adv, os.path.join(fid_adv_dir, save_name))
            
            # 可视化 (Crop 区域)
            self.save_unified_visualization(origin_image, final_img_01, route_clean, route_final, 
                                            gt_route, target_point,
                                            speed_clean, speed_final, output_dir, base_name)

        return final_dist, final_spd_diff, sim_drift, road_lpips, road_ssim, lat_error, lon_error, clean_route_len, is_success

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True, 
                        help="Root directory containing multiple dataset folders")
    parser.add_argument("--output_dir", type=str, default="fgsm_allresults") 
    args = parser.parse_args()

    # 1. 确定本次运行的序号文件夹 (FGSM-gray-N)
    base_dir = args.output_dir
    os.makedirs(base_dir, exist_ok=True)

    existing_folders = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    max_id = 0
    for folder in existing_folders:
        if folder.startswith("FGSM-gray-"):
            try:
                idx = int(folder.split("-")[-1])
                if idx > max_id: max_id = idx
            except ValueError: continue
    
    new_folder_name = f"FGSM-gray-{max_id + 1}"
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
    print("MFA-Style Grey-Box FGSM Config:")
    print(json.dumps(ATTACK_PARAMS, indent=2))
    
    attacker = FGSM_GreyBox_Attacker()
    
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
    total_drift_accumulated = 0.0
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
            total_drift_accumulated += ds_totals['sim_drift'] 
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
        if len(os.listdir(global_fid_clean)) > 0:
            if calculate_metrics is not None:
                metrics = calculate_metrics(input1=global_fid_clean, input2=global_fid_adv, cuda=True, isc=False, fid=True, verbose=False)
                final_fid = metrics['frechet_inception_distance']
                print(f"  > GLOBAL FID SCORE: {final_fid:.4f}")
            else:
                print("  > torch-fidelity not found, skipping FID.")
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
        real_global_drift = total_drift_accumulated / total_samples_processed
        print(f"  > Avg SimLingo Drift:   {real_global_drift:.4f}")
        print(f"  > Avg LPIPS:            {get_avg('road_lpips'):.4f}")
        print(f"  > Avg SSIM:             {get_avg('road_ssim'):.4f}")
        print(f"  > Global FID (All):     {final_fid:.4f}")
        print("="*60)

if __name__ == "__main__":
    main()