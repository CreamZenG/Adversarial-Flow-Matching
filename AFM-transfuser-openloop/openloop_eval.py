import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import matplotlib.patches as patches 
import os
import argparse
import torch
import numpy as np
import cv2
from tqdm import tqdm
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# 引入项目中的模块
from config import GlobalConfig
from model import LidarCenterNet
from data import CARLA_Data, lidar_bev_cam_correspondences
from copy import deepcopy

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained .pth model file')
    parser.add_argument('--root_dir', type=str, required=True, help='Root directory of the validation dataset')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    
    # 模型架构参数 (需要与训练时一致)
    parser.add_argument('--backbone', type=str, default='transFuser', help='transFuser, geometric_fusion, late_fusion, latentTF')
    parser.add_argument('--image_architecture', type=str, default='regnety_032')
    parser.add_argument('--lidar_architecture', type=str, default='regnety_032')
    parser.add_argument('--use_velocity', type=int, default=1)
    parser.add_argument('--use_point_pillars', type=int, default=0)
    parser.add_argument('--use_target_point_image', type=int, default=1, help='1 for True, 0 for False')

    # 可视化选项
    parser.add_argument('--save_vis', action='store_true', help='Save visualization images')
    parser.add_argument('--vis_dir', type=str, default='eval_vis_results', help='Directory to save visualizations')
    
    return parser.parse_args()

def visualize_bev(rgb_tensor, pred_wp, gt_wp, target_point, save_path, frame_id):
    """
    绘制左图右表的可视化结果 (最终修正坐标版)
    """
    # --- 1. 处理图像 ---
    img_np = rgb_tensor.permute(1, 2, 0).cpu().numpy()
    img_np = img_np.astype(np.float32) / 255.0
    img_np = np.clip(img_np, 0, 1)

    # (可选) 简单的 Gamma 矫正，让夜景稍微亮一点，方便人眼观察
    # img_np = np.power(img_np, 0.7) 

    # --- 2. 创建画布 ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
    
    # 左图
    ax1.imshow(img_np)
    ax1.set_title(f"Input Crop (Frame {frame_id})")
    ax1.axis('off')

    # 右图：BEV 轨迹
    # 绘制自车
    rect = patches.Rectangle((-1.0, -1.0), 2.0, 4.0, linewidth=1, edgecolor='k', facecolor='gray', alpha=0.3)
    ax2.add_patch(rect)
    ax2.plot(0, 0, 'k^', markersize=10, label='Ego')

    # 绘制 GT (红色) - 这里的坐标是正确的 [纵向, 横向] -> plot(横向, 纵向)
    if gt_wp is not None:
        ax2.plot(gt_wp[:, 1], gt_wp[:, 0], 'r--o', markersize=4, label='GT', alpha=0.6)

    # 绘制 预测 (绿色)
    ax2.plot(pred_wp[:, 1], pred_wp[:, 0], 'g-x', markersize=4, label='Pred', linewidth=2)

    # 绘制 目标点 (蓝色星号)
    # 【最终修正】：
    # 经过分析，data.py 输出的 target_point 格式为：
    # target_point[0]: Lateral (横向，左正右负或反之)
    # target_point[1]: -Longitudinal (负的纵向距离)
    # 
    # 所以为了在图上画对 (X=Lateral, Y=Longitudinal):
    # Plot_X = target_point[0]
    # Plot_Y = -target_point[1]
    
    ax2.plot(target_point[0], -target_point[1], 'b*', markersize=18, label='Target', zorder=10)

    # 设置样式
    ax2.set_title("BEV Trajectory")
    ax2.set_xlabel("Lateral (m) [Y-axis]")
    ax2.set_ylabel("Longitudinal (m) [X-axis]")
    ax2.legend(loc='lower right') # 图例改到右下角，避开前方路径
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.set_aspect('equal')
    

    # 获取目标点的纵向距离 (Longitudinal)
    target_long = -target_point[1]
    
    # 默认视野上限是 30 米
    view_max_y = 30.0
    
    # 如果目标点比 30 米还要远，就把视野上限撑大一点，保证能看到它
    if target_long > 28.0:
        view_max_y = target_long + 5.0 # 多留 5 米余量
    # 视野范围
    ax2.set_xlim(-15, 15)
    ax2.set_ylim(-5, view_max_y)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)

def main():
    args = parse_args()
    
    # 1. 配置与初始化
    config = GlobalConfig(setting='eval') # 使用 eval 模式配置
    config.use_velocity = bool(args.use_velocity)
    config.backbone = args.backbone
    config.use_point_pillars = bool(args.use_point_pillars)
    
    config.use_target_point_image = bool(args.use_target_point_image)

    device = torch.device(args.device)
    
    if args.save_vis:
        os.makedirs(args.vis_dir, exist_ok=True)

    # 2. 加载模型
    print(f"Loading model from {args.model_path}...")
    model = LidarCenterNet(config, device, args.backbone, args.image_architecture, args.lidar_architecture, bool(args.use_velocity))
    
    state_dict = torch.load(args.model_path, map_location=device)
    # 处理 DDP 保存的模型 (去除 'module.' 前缀)
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
            
    model.load_state_dict(new_state_dict, strict=False)
    model.to(device)
    model.eval()

    # 3. 加载数据
    # 直接使用 data.py 中的 CARLA_Data，它处理了复杂的 lidar 对齐和 json 解析
    # 注意：这里我们传入 root_dir 列表，因为 CARLA_Data 期望列表
    print(f"Loading dataset from {args.root_dir}...")
    val_dataset = CARLA_Data(root=[args.root_dir], config=config, shared_dict=None)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    print(f"Start evaluation on {len(val_dataset)} frames.")
    
    total_ade = 0.0
    total_fde = 0.0
    count = 0
    
    # 4. 评估循环
    with torch.no_grad():
        for batch_idx, data in enumerate(tqdm(val_loader)):
            # 准备数据 (参考 train.py 的 load_data_compute_loss)
            rgb = data['rgb'].to(device, dtype=torch.float32)
            bev = data['bev'].to(device, dtype=torch.long)
            label = data['label'].to(device, dtype=torch.float32) # GT 包含 bounding box
            
            # GT Waypoints (在 data.py 中被称为 ego_waypoint)
            # data.py: data['ego_waypoint'] = ego_waypoint (N, 4, 2) ? 需确认维度
            # 检查 data.py: data['ego_waypoint'] 是最后得到的，只有当前帧对应的未来路径
            gt_waypoints = data['ego_waypoint'].to(device, dtype=torch.float32) # (B, 4, 2)
            
            target_point = data['target_point'].to(device, dtype=torch.float32)
            target_point_image = data['target_point_image'].to(device, dtype=torch.float32)
            ego_vel = data['speed'].to(device, dtype=torch.float32).reshape(-1, 1)

            # 处理 LiDAR
            if config.use_point_pillars:
                lidar = data['lidar_raw'].to(device, dtype=torch.float32)
                num_points = data['num_points'].to(device, dtype=torch.int32)
            else:
                lidar = data['lidar'].to(device, dtype=torch.float32)
                num_points = None

            # 处理 Geometric Fusion 特有的输入
            bev_points = None
            cam_points = None
            if args.backbone == 'geometric_fusion':
                bev_points = data['bev_points'].long().to(device)
                cam_points = data['cam_points'].long().to(device)

            # === 模型推理 ===
            # 调用 forward_ego 而不是 forward，因为我们只需要预测，不需要计算 loss
            # 注意：Geometric Fusion 和其他 backbone 参数略有不同
            if args.backbone == 'geometric_fusion':
                pred_wp, _ = model.forward_ego(rgb, lidar, target_point, target_point_image, ego_vel, 
                                            bev_points=bev_points, cam_points=cam_points, num_points=num_points)
            else:
                pred_wp, _ = model.forward_ego(rgb, lidar, target_point, target_point_image, ego_vel, num_points=num_points)
            
            # pred_wp shape: (B, 4, 2) -> (Batch, TimeSteps, XY)
            
            # === 指标计算 ===
            # 计算 L2 距离
            # gt_waypoints shape: (B, 4, 2)
            diff = torch.norm(pred_wp - gt_waypoints, dim=-1) # (B, 4)
            
            ade = diff.mean(dim=-1).sum().item() # 平均位移误差
            fde = diff[:, -1].sum().item()       # 终点位移误差
            
            total_ade += ade
            total_fde += fde
            count += rgb.shape[0]

            if args.save_vis and batch_idx % 10 == 0: 
                save_path = os.path.join(args.vis_dir, f"eval_{batch_idx:04d}.png")
                
                # 获取数据转 numpy
                vis_rgb = rgb[0] # 取 batch 中第一张
                vis_pred = pred_wp[0].cpu().numpy()
                vis_gt = gt_waypoints[0].cpu().numpy()
                vis_target = target_point[0].cpu().numpy()

                # 调用新的 BEV 可视化函数
                visualize_bev(vis_rgb, vis_pred, vis_gt, vis_target, save_path, batch_idx)

    # 5. 输出结果
    avg_ade = total_ade / count
    avg_fde = total_fde / count
    
    print("="*30)
    print(f"Evaluation Results ({count} samples):")
    print(f"ADE (Average Displacement Error): {avg_ade:.4f} meters")
    print(f"FDE (Final Displacement Error)  : {avg_fde:.4f} meters")
    print("="*30)
    if args.save_vis:
        print(f"Visualizations saved to {args.vis_dir}")

if __name__ == "__main__":
    main()