import os
import sys
import glob
import json
import gzip
import argparse
import math
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import hydra
from omegaconf import OmegaConf
from transformers import AutoProcessor, AutoConfig
import matplotlib.pyplot as plt

# ================= 配置区域 =================
REPO_ROOT = "/home/pc/simlingo"
CHECKPOINT_PATH = f"{REPO_ROOT}/outputs/simlingo/checkpoints/epoch=013.ckpt/pytorch_model.pt"
CONFIG_PATH = f"{REPO_ROOT}/outputs/simlingo/.hydra/config.yaml"
DEVICE = "cuda:0"

# 添加项目路径
sys.path.append(REPO_ROOT)

try:
    from simlingo_training.utils.custom_types import DrivingInput, LanguageLabel
    from simlingo_training.utils.internvl2_utils import build_transform, dynamic_preprocess
    from team_code.simlingo_utils import get_camera_intrinsics, get_camera_extrinsics
except ImportError as e:
    print(f"环境加载失败: {e}\n请确保 REPO_ROOT 设置正确，且已激活正确的 conda 环境。")
    sys.exit(1)

# ================= 工具函数 =================

def world_to_ego(route_global, ego_x, ego_y, ego_theta_rad):
    """
    将全局坐标转换为车辆(Ego)坐标系
    参数:
      - route_global: (N,2) np.array of global XY
      - ego_x, ego_y: 车辆在全局坐标中的位置
      - ego_theta_rad: 车辆朝向，单位为 弧度 (radians)
    返回:
      - (N,2) 在 Ego 坐标系下的点 (local_x, local_y)
    说明:
      - 使用 R(-theta) * (p - ego_pos)
      - 本函数假定 ego_theta 已经是弧度；调用方负责单位检测/转换
    """
    # 确保数组形状
    route_global = np.asarray(route_global)
    if route_global.ndim != 2 or route_global.shape[1] < 2:
        raise ValueError("route_global must be (N,2)")

    yaw = float(ego_theta_rad)  # already radians

    diff = route_global - np.array([ego_x, ego_y])
    dx = diff[:, 0]
    dy = diff[:, 1]

    c = np.cos(yaw)
    s = np.sin(yaw)

    # R(-theta) * diff  => [cos, sin; -sin, cos] * [dx; dy]
    local_x = dx * c + dy * s
    local_y = -dx * s + dy * c

    return np.stack([local_x, local_y], axis=-1)

# ================= 评估主类 =================

class OpenLoopEvaluator:
    def __init__(self):
        self.device = torch.device(DEVICE)
        
        # 1. 加载配置
        print(f"Loading Config: {CONFIG_PATH}")
        with open(CONFIG_PATH, 'r') as f:
            self.cfg = OmegaConf.load(f)
        self.use_thumbnail = self.cfg.data_module.use_global_img
        
        # 2. 加载 Processor
        variant = self.cfg.model.vision_model.variant
        print(f"Loading Processor: {variant}")
        self.processor = AutoProcessor.from_pretrained(variant, trust_remote_code=True)
        self.tokenizer = getattr(self.processor, 'tokenizer', self.processor)
        self.tokenizer.padding_side = "left"
        # 必须添加特殊 Token
        self.tokenizer.add_special_tokens({'additional_special_tokens': ['<WAYPOINTS>', '<TARGET_POINT>']})

        # 3. 计算 Token 数量
        self.tmp_config = AutoConfig.from_pretrained(variant, trust_remote_code=True)
        image_size = self.tmp_config.force_image_size or self.tmp_config.vision_config.image_size
        patch_size = self.tmp_config.vision_config.patch_size
        downsample_ratio = getattr(self.tmp_config, 'downsample_ratio', 0.5)
        self.num_image_token = int((image_size // patch_size) ** 2 * (downsample_ratio ** 2))

        # 4. 加载模型
        print("Loading Model...")
        torch.set_default_dtype(torch.bfloat16)
        cache_dir = f"pretrained/{variant.split('/')[-1]}"
        self.model = hydra.utils.instantiate(
            self.cfg.model, cfg_data_module=self.cfg.data_module,
            processor=self.processor, cache_dir=cache_dir, _recursive_=False
        ).to(self.device)
        
        # 5. 加载权重
        print(f"Loading Weights: {CHECKPOINT_PATH}")
        state_dict = torch.load(CHECKPOINT_PATH, map_location=self.device)
        if 'state_dict' in state_dict: state_dict = state_dict['state_dict']
        new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        self.model.load_state_dict(new_state_dict, strict=False)
        self.model.eval()

        self.transform = build_transform(input_size=448)
        
        # 预计算相机的内外参
        self.image_sizes = torch.tensor([[448, 448]]).to(self.device)
        self.intrinsics = get_camera_intrinsics(448, 448, fov=110).unsqueeze(0).to(self.device).float().view(1, 3, 3)
        self.extrinsics = get_camera_extrinsics().unsqueeze(0).to(self.device).float().view(1, 4, 4)

    def load_data(self, image_path):
        """加载数据，选取较远的目标点 (Target > 10m & > 20m)
        改进点:
        - 优先从 pos_global 或 ego_matrix 提取 ego pose
        - 对 theta 进行单位检测（若 > 2*pi 则认为是度并转为弧度）
        - 判断 route 是否已经在 ego 坐标系（基于数值量级），仅在需要时做 world->ego 转换
        """
        img_dir = os.path.dirname(image_path)
        root_dir = os.path.dirname(img_dir)
        base_name = os.path.basename(image_path).replace('.jpg', '')

        json_path = os.path.join(root_dir, 'measurements', f"{base_name}.json.gz")
        if not os.path.exists(json_path):
            json_path = os.path.join(root_dir, 'measurements', f"{base_name}.json")
        if not os.path.exists(json_path):
            return None

        try:
            open_func = gzip.open if json_path.endswith('.gz') else open
            with open_func(json_path, 'rt', encoding='utf-8') as f:
                data = json.load(f)

            speed = float(data.get('speed', 0.0))

            # 尝试提取全局位置（pos_global 或 ego_matrix）
            pos_global = data.get('pos_global', None)
            ego_x = ego_y = 0.0
            if pos_global and len(pos_global) >= 2:
                ego_x, ego_y = float(pos_global[0]), float(pos_global[1])
            else:
                # 尝试用 ego_matrix 的平移列（若存在）
                ego_matrix = data.get('ego_matrix', None)
                if ego_matrix and len(ego_matrix) >= 3 and len(ego_matrix[0]) >= 4:
                    ego_x = float(ego_matrix[0][3])
                    ego_y = float(ego_matrix[1][3])

            # theta 单位检测：优先认为数据是弧度；如果数值很大（> 2*pi），则当作度并转换为弧度
            ego_theta = data.get('theta', 0.0)
            try:
                ego_theta = float(ego_theta)
            except:
                ego_theta = 0.0
            if abs(ego_theta) > 2 * math.pi:
                # 很大，可能是度
                ego_theta = math.radians(ego_theta)

            gt_route_local = None
            # 默认给较远的直行点
            target_point_1 = np.array([10.0, 0.0])
            target_point_2 = np.array([20.0, 0.0])

            if 'route' in data:
                route = np.array(data['route'], dtype=float)
                # 确保形状 (N,2)
                if route.ndim == 1 and route.size >= 2:
                    route = route.reshape(-1, 2)
                elif route.ndim >= 2 and route.shape[1] > 2:
                    route = route[:, :2]

                # 判断 route 是 global 还是 local：通过数值量级判断
                # 若 route 的坐标非常大（比如 > 1000），很可能为全局坐标
                max_abs = np.max(np.abs(route))
                route_is_global = max_abs > 1000.0  # 可根据数据集调整阈值

                if route_is_global:
                    # 以 pos_global / ego_matrix 提取的 ego_x, ego_y, ego_theta 做变换
                    gt_route_local = world_to_ego(route, ego_x, ego_y, ego_theta)
                else:
                    # route 很可能就是 Ego/local 坐标，直接使用
                    gt_route_local = route

                if gt_route_local is not None and len(gt_route_local) > 0:
                    dists = np.linalg.norm(gt_route_local, axis=1)

                    idx1 = np.where(dists > 10.0)[0]
                    target_point_1 = gt_route_local[idx1[0]] if len(idx1) > 0 else gt_route_local[-1]

                    idx2 = np.where(dists > 20.0)[0]
                    target_point_2 = gt_route_local[idx2[0]] if len(idx2) > 0 else gt_route_local[-1]

            return {
                "speed": speed,
                "gt_route": gt_route_local,
                "target_point_1": target_point_1,
                "target_point_2": target_point_2
            }
        except Exception as e:
            # 可选：打印异常以便调试
            print(f"load_data error for {image_path}: {e}")
            return None
        
    def visualize(self, img_path, raw_image, gt_route, pred_route, target_point, speed, output_dir):
        base_name = os.path.basename(img_path).replace('.jpg', '')
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        ax1.imshow(raw_image)
        ax1.set_title(f"ID: {base_name}\nSpeed: {speed:.1f} m/s")
        ax1.axis('off')

        ax2.plot(0, 0, 'k^', markersize=12, label='Ego')
        
        if gt_route is not None:
            # 绘制真值
            ax2.plot(gt_route[:, 1], gt_route[:, 0], 'g-o', markersize=4, label='GT')
        
        # 绘制预测
        ax2.plot(pred_route[:, 1], pred_route[:, 0], 'b-x', markersize=4, label='Pred')
        
        # 绘制输入的第一个目标点
        ax2.plot(target_point[1], target_point[0], 'r*', markersize=15, label='Target', zorder=10)

        ax2.set_xlim(-15, 15)
        ax2.set_ylim(-2, 35) # 视野扩大一点，因为Target点变远了
        ax2.set_xlabel("Lateral (m)")
        ax2.set_ylabel("Longitudinal (m)")
        ax2.legend(loc='upper right')
        ax2.grid(True, alpha=0.4, linestyle='--')
        ax2.set_aspect('equal')
        
        os.makedirs(output_dir, exist_ok=True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{base_name}_viz.png"), dpi=80)
        plt.close(fig)

    def run_eval_batch(self, image_dir, output_dir=None):
        image_paths = sorted(glob.glob(os.path.join(image_dir, "*.jpg")))
        print(f"Found {len(image_paths)} images.")
        if output_dir:
            print(f"Visualization will be saved to: {output_dir}")
        
        ade_list, fde_list = [], []
        tp_token_id = self.tokenizer.convert_tokens_to_ids('<TARGET_POINT>')
        
        # 构造 Tokens
        prompt_tp = "Target waypoint: <TARGET_POINT><TARGET_POINT>."
        IMG_START = '<img>'; IMG_END = '</img>'; IMG_CTX = '<IMG_CONTEXT>'
        num_patches = 2 
        img_tokens = IMG_START + IMG_CTX * self.num_image_token * num_patches + IMG_END
        
        pbar = tqdm(image_paths)
        for i, img_path in enumerate(pbar):
            data = self.load_data(img_path)
            if data is None or data['gt_route'] is None: continue
            
            speed = data['speed']
            gt_route = data['gt_route']
            tp1 = data['target_point_1']
            tp2 = data['target_point_2']
            
            # 1. 图像处理
            raw_image = Image.open(img_path).convert('RGB')
            images_pp = dynamic_preprocess(raw_image.resize((448,448)), image_size=448, use_thumbnail=self.use_thumbnail, max_num=2)
            pixel_values = torch.stack([self.transform(img) for img in images_pp])
            if pixel_values.shape[0] == 1: pixel_values = pixel_values.repeat(2, 1, 1, 1)
            clean_cam_imgs = pixel_values.view(1, 1, 2, 3, 448, 448).to(self.device).bfloat16()

            # 2. [核心] 构造包含 Target Point 的输入
            target_points_np = np.stack([tp1, tp2]) # Shape (2, 2)
            placeholder_dict = {tp_token_id: target_points_np}
            
            text_prompt = f"Current speed: {speed:.2f} m/s. {prompt_tp} Predict the waypoints."
            final_prompt = img_tokens + "\n" + text_prompt
            
            tokens = self.tokenizer([final_prompt], padding=True, return_tensors="pt")
            ll = LanguageLabel(
                phrase_ids=tokens["input_ids"].to(self.device), 
                phrase_valid=tokens["attention_mask"].bool().to(self.device),
                phrase_mask=tokens["attention_mask"].bool().to(self.device), 
                placeholder_values=[placeholder_dict], # 注入数值
                language_string=[final_prompt], loss_masking=None
            )

            # 3. 构造参数
            # base_params 中的 target_point 主要是为了接口兼容，实际 LLM 用的是 ll
            tp_tensor = torch.from_numpy(tp1).float().unsqueeze(0).to(self.device)
            
            base_params = {
                "image_sizes": self.image_sizes, "camera_intrinsics": self.intrinsics, "camera_extrinsics": self.extrinsics,
                "vehicle_speed": torch.tensor([[speed]]).to(self.device).float(), 
                "target_point": tp_tensor, 
                "prompt": ll, "prompt_inference": ll
            }

            # 4. 推理
            inputs = DrivingInput(camera_images=clean_cam_imgs, **base_params)
            with torch.no_grad():
                _, pred_route, _ = self.model(inputs)
            
            pred_route_np = pred_route[0].float().cpu().numpy()
            
            # 5. 计算指标
            eval_len = min(len(gt_route), len(pred_route_np))
            if eval_len == 0: continue
            
            diff = gt_route[:eval_len] - pred_route_np[:eval_len]
            l2_errors = np.linalg.norm(diff, axis=1)
            ade, fde = np.mean(l2_errors), l2_errors[-1]
            
            if ade < 50: # 过滤极端异常值
                ade_list.append(ade)
                fde_list.append(fde)
            
            # 6. 可视化 (每5帧存一张)
            if output_dir and i % 5 == 0:
                self.visualize(img_path, raw_image, gt_route, pred_route_np, tp1, speed, output_dir)

            pbar.set_description(f"ADE: {np.mean(ade_list):.3f}m")

        return ade_list, fde_list

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 必需参数：输入路径
    parser.add_argument("--img_dir", type=str, required=True, help="RGB 图像数据的文件夹路径")
    # 可选参数：可视化保存路径 (如果不传，则不保存图片)
    parser.add_argument("--output_dir", type=str, default=None, help="可视化结果保存路径 (可选)")
    args = parser.parse_args()
    
    if not os.path.exists(args.img_dir):
        sys.exit(f"路径不存在: {args.img_dir}")

    evaluator = OpenLoopEvaluator()
    ade_scores, fde_scores = evaluator.run_eval_batch(args.img_dir, output_dir=args.output_dir)
    
    if len(ade_scores) > 0:
        print("\n" + "="*50)
        print(f"评估完成")
        print(f"ADE: {np.mean(ade_scores):.4f} m")
        print(f"FDE: {np.mean(fde_scores):.4f} m")
        if args.output_dir:
            print(f"可视化结果已保存至: {args.output_dir}")
        print("="*50)
    else:
        print("未找到有效数据。")