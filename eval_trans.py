import os
# 禁用 Tokenizers 并行
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import sys
import json
import argparse
import warnings
from tqdm import tqdm
import time
from datetime import timedelta

warnings.filterwarnings("ignore")

import torch
import numpy as np
from PIL import Image
from torchvision import transforms

# === 绘图依赖 ===
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

# === SimLingo 依赖 ===
import hydra
from omegaconf import OmegaConf
from transformers import AutoProcessor

# === Metrics ===
try:
    import lpips
    USE_LPIPS = True
except ImportError:
    USE_LPIPS = False

# ==========================================
# 1. 可视化保存函数 (保存裁剪后的视角)
# ==========================================
def save_visualization(img_pil, route_clean, route_adv, gt_route, save_path, info_text=""):
    """
    绘制并保存：
    左图：对抗样本 (已经过 Center Crop，是 SimLingo 真正看到的视野)
    右图：BEV 路线对比
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # --- 左图: 对抗样本 ---
    axes[0].imshow(img_pil)
    axes[0].set_title(f"SimLingo Input (Center Cropped)\n{info_text}", fontsize=9)
    axes[0].axis('off')

    # --- 右图: 路线对比 (BEV) ---
    if gt_route is not None and len(gt_route) > 0:
        axes[1].plot(gt_route[:, 1], gt_route[:, 0], 'g-o', label='Ground Truth', markersize=4, linewidth=2)

    if route_clean is not None:
        axes[1].plot(route_clean[:, 1], route_clean[:, 0], 'b-^', label='Clean Pred', markersize=4, linewidth=2)

    if route_adv is not None:
        axes[1].plot(route_adv[:, 1], route_adv[:, 0], 'r-x', label='Adv Pred', markersize=4, linewidth=2)

    axes[1].set_title("Trajectory Prediction (BEV)")
    axes[1].set_xlabel("Lateral (m)")
    axes[1].set_ylabel("Longitudinal (m)")
    axes[1].legend()
    axes[1].grid(True, linestyle='--', alpha=0.5)
    axes[1].set_aspect('equal', adjustable='datalim')
    axes[1].invert_xaxis() # CARLA 坐标系习惯翻转X轴(左转在左)

    plt.tight_layout()
    plt.savefig(save_path, dpi=100)
    plt.close(fig)

# ==========================================
# 辅助函数
# ==========================================
def get_ssim(img1_np, img2_np):
    if img1_np.max() <= 1.0: img1_np = img1_np * 255.0
    if img2_np.max() <= 1.0: img2_np = img2_np * 255.0
    mu1 = img1_np.mean(); mu2 = img2_np.mean()
    sigma1 = np.sqrt(((img1_np - mu1)**2).mean())
    sigma2 = np.sqrt(((img2_np - mu2)**2).mean())
    sigma12 = ((img1_np - mu1)*(img2_np - mu2)).mean()
    k1, k2, L = 0.01, 0.03, 255
    C1 = (k1*L)**2; C2 = (k2*L)**2
    return (2*mu1*mu2 + C1)*(2*sigma12 + C2)/((mu1**2+mu2**2+C1)*(sigma1**2+sigma2**2+C2))

def get_cmd_text(cmd_id):
    cmd_map = {1:"turn left", 2:"turn right", 3:"go straight", 4:"follow lane", 5:"change left", 6:"change right"}
    return cmd_map.get(cmd_id, "unknown")

def load_image_robust(path):
    """
    尝试加载图片，自动处理 .jpg / .png 后缀不匹配的问题
    """
    if os.path.exists(path):
        return Image.open(path).convert('RGB')
    
    # 如果路径是以 .jpg 结尾但找不到，尝试 .png
    if path.endswith(".jpg"):
        png_path = path.replace(".jpg", ".png")
        if os.path.exists(png_path):
            return Image.open(png_path).convert('RGB')
            
    # 如果路径是以 .png 结尾但找不到，尝试 .jpg
    if path.endswith(".png"):
        jpg_path = path.replace(".png", ".jpg")
        if os.path.exists(jpg_path):
            return Image.open(jpg_path).convert('RGB')
            
    raise FileNotFoundError(f"Cannot find image (tried jpg/png): {path}")

# ==========================================
# 核心类
# ==========================================
class DiffAttackerGreyBox:
    def __init__(self, cfg_path, ckpt_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cfg = OmegaConf.load(cfg_path)
        print(f"Loading Model from {ckpt_path}...")
        
        try:
            from transformers import AutoProcessor
            self.processor = AutoProcessor.from_pretrained(self.cfg.model.base_model)
            # === 这里需要您的 SimLingo 模型类 ===
            # self.model = ...
            pass 
        except Exception as e:
            print(f"Model Load Warning: {e}")
        
        if hasattr(self, 'model'):
            self.model.to(self.device).eval()
            self.tokenizer = self.processor.tokenizer
        else:
            from transformers import AutoTokenizer
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.model.base_model)
            except: pass

        self.num_image_token = 12
        if USE_LPIPS:
            self.lpips_loss = lpips.LPIPS(net='vgg').to(self.device).eval()

    def load_measurements(self, clean_img_path):
        # 即使图片是 png，measurement 依然通过原路径规则找
        # 原始数据集通常是 jpg，所以这里的逻辑还是基于 clean_img_path
        json_path = clean_img_path.replace("rgb", "measurements").replace(".jpg", ".json").replace(".png", ".json")
        if not os.path.exists(json_path): 
            json_path = clean_img_path.replace(".jpg", ".json").replace(".png", ".json")
        
        if not os.path.exists(json_path): return None
            
        with open(json_path, 'r') as f: data = json.load(f)
        return (
            data.get('speed', 0), data.get('command', 4), "Green", 
            np.array([0.0, 0.5]), np.array([0.0, 1.0]), np.array(data.get('future_waypoints', []))
        )

    def crop_and_process(self, img_pil, target_size=(448, 448)):
        """
        核心函数：执行中心裁剪 + 缩放
        返回: (PIL_Image_Cropped, Tensor)
        """
        w, h = img_pil.size
        min_dim = min(w, h)
        
        # 1. 定义变换
        # TransFuser 输出可能是宽图，SimLingo 只看中间
        cropper = transforms.Compose([
            transforms.CenterCrop(min_dim),
            transforms.Resize(target_size, interpolation=transforms.InterpolationMode.BICUBIC)
        ])
        
        # 2. 执行变换得到 PIL (用于绘图和 SSIM)
        img_cropped = cropper(img_pil)
        
        # 3. 转 Tensor (用于模型推理和 LPIPS)
        t = torch.from_numpy(np.array(img_cropped)).float() / 255.0
        t = t.permute(2, 0, 1).unsqueeze(0).to(self.device)
        
        return img_cropped, t

    def evaluate_pair(self, clean_path, adv_path):
        """
        加载 PNG/JPG -> Center Crop -> Evaluate -> Visualize
        """
        try:
            clean_pil_full = load_image_robust(clean_path)
            adv_pil_full = load_image_robust(adv_path) # 这里加载 PNG
        except Exception as e:
            print(f"Img Error: {e}"); return None

        # === 核心修改：对 Clean 和 Adv 都进行中心裁剪 ===
        # 这样我们比较的是 SimLingo 真正看到的区域，指标更准确
        clean_crop_pil, clean_crop_tensor = self.crop_and_process(clean_pil_full)
        adv_crop_pil, adv_crop_tensor = self.crop_and_process(adv_pil_full)

        # 1. 计算视觉指标 (基于 Crop 后的图片)
        metrics = {'lpips': 0, 'ssim': 0}
        
        # SSIM
        metrics['ssim'] = get_ssim(np.array(clean_crop_pil), np.array(adv_crop_pil))
        
        # LPIPS
        if USE_LPIPS:
            # 归一化到 [-1, 1]
            t_clean = clean_crop_tensor * 2.0 - 1.0
            t_adv = adv_crop_tensor * 2.0 - 1.0
            metrics['lpips'] = self.lpips_loss(t_clean, t_adv).item()

        # 2. Metadata (JSON)
        meta = self.load_measurements(clean_path)
        if meta is None: return None
        speed, cmd, tl, tp1, tp2, gt_route = meta

        # 3. SimLingo 推理
        prompt = f"Current speed: {speed:.2f} m/s. Command: {get_cmd_text(cmd)}. Traffic light: {tl}. Target waypoint: <TARGET_POINT><TARGET_POINT>. Predict the waypoints."
        full_prompt = '<img>' + '<IMG_CONTEXT>'*self.num_image_token*2 + '</img>\n' + prompt
        
        tokens = self.tokenizer([full_prompt], padding=True, return_tensors="pt")
        base_inputs = {
            "input_ids": tokens.input_ids.to(self.device),
            "attention_mask": tokens.attention_mask.to(self.device),
            "placeholder_dict": {self.tokenizer.convert_tokens_to_ids('<TARGET_POINT>'): np.stack([tp1, tp2])}
        }

        with torch.no_grad():
            # 输入的是 Crop 后的 tensor
            out_adv = self.model(camera_images=adv_crop_tensor, **base_inputs)
            out_clean = self.model(camera_images=clean_crop_tensor, **base_inputs)
            
            r_adv = out_adv[1] if isinstance(out_adv, tuple) else out_adv.logits
            r_clean = out_clean[1] if isinstance(out_clean, tuple) else out_clean.logits

        # 4. 结果打包
        r_adv, r_clean = r_adv.cpu(), r_clean.cpu()
        metrics['shift'] = torch.norm(r_adv.float() - r_clean.float(), p=2).mean().item()
        metrics['lat'] = abs(r_adv[0, -1, 1].item() - r_clean[0, -1, 1].item())
        metrics['lon'] = abs(r_adv[0, -1, 0].item() - r_clean[0, -1, 0].item())
        
        # 返回数据用于绘图 (注意这里返回的是 Crop 后的 Adv 图片)
        metrics['route_clean_np'] = r_clean.numpy()[0]
        metrics['route_adv_np'] = r_adv.numpy()[0]
        metrics['gt_route_np'] = gt_route
        metrics['adv_pil'] = adv_crop_pil # 可视化的是裁剪后的图
        metrics['cmd_text'] = get_cmd_text(cmd)

        return metrics

# ==========================================
# 主程序
# ==========================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adv_dir", type=str, required=True, help="Folder with mapping.json & PNG images")
    parser.add_argument("--simlingo_root", type=str, default="/home/pc/simlingo")
    
    # 可视化开关
    parser.add_argument("--save_vis", action="store_true", help="Save visualization")
    parser.add_argument("--vis_dir", type=str, default="vis_results_png", help="Save folder")
    
    args = parser.parse_args()
    
    ckpt = f"{args.simlingo_root}/outputs/simlingo/checkpoints/epoch=013.ckpt/pytorch_model.pt"
    cfg = f"{args.simlingo_root}/outputs/simlingo/.hydra/config.yaml"
    
    # 读取 Mapping
    map_path = os.path.join(args.adv_dir, "mapping.json")
    if not os.path.exists(map_path):
        print(f"Error: mapping.json not found in {args.adv_dir}"); return

    with open(map_path, 'r') as f: mapping_data = json.load(f)
    print(f"Loaded {len(mapping_data)} samples. (Expecting PNGs or JPGs)")
    
    if args.save_vis:
        os.makedirs(args.vis_dir, exist_ok=True)
        print(f"Vis Dir: {args.vis_dir}")

    attacker = DiffAttackerGreyBox(cfg, ckpt)
    stats = {'shift': [], 'lat': [], 'lon': [], 'lpips': [], 'ssim': []}
    
    for entry in tqdm(mapping_data):
        # 1. 获取 Adv 路径
        adv_fname = entry['adv_filename'] 
        adv_full = os.path.join(args.adv_dir, adv_fname)
        
        # 2. 获取 Clean 路径
        clean_abs = entry['clean_path']
        
        # 3. 运行评估 (函数内部会自动尝试加载 png)
        res = attacker.evaluate_pair(clean_abs, adv_full)
        
        if res:
            for k in stats: stats[k].append(res[k])

            # 4. 保存可视化
            if args.save_vis:
                dataset = entry.get('dataset', 'unknown')
                frame_id = entry.get('frame_id', 0)
                shift_val = res['shift']
                
                # 存为 PNG，并在文件名中标注 Shift 大小
                save_name = f"{dataset}_{frame_id:04d}_S{shift_val:.2f}.png"
                save_path = os.path.join(args.vis_dir, save_name)
                
                info = f"Cmd: {res['cmd_text']} | Shift: {shift_val:.2f}m"
                
                save_visualization(
                    img_pil=res['adv_pil'],       # 这里是裁剪后的图
                    route_clean=res['route_clean_np'],
                    route_adv=res['route_adv_np'],
                    gt_route=res['gt_route_np'],
                    save_path=save_path,
                    info_text=info
                )

    # 打印最终指标
    if len(stats['shift']) > 0:
        def mean(l): return sum(l)/len(l)
        print("\n=== FINAL RESULTS (SimLingo Center-Cropped) ===")
        print(f"Avg Shift: {mean(stats['shift']):.4f} m")
        print(f"Avg LPIPS: {mean(stats['lpips']):.4f}")
        print(f"Avg SSIM:  {mean(stats['ssim']):.4f}")

if __name__ == "__main__":
    main()