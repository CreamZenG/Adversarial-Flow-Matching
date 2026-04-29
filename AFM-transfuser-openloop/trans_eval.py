import os
import sys
import argparse
import time
import math
import warnings
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 忽略警告
warnings.filterwarnings("ignore")

# ================= TransFuser 依赖检查 =================
try:
    from config import GlobalConfig
    from model import LidarCenterNet
    # 注意：这里优先使用 data，如果你的环境是 data1 请自行修改
    try:
        from data import CARLA_Data
    except ImportError:
        from data1 import CARLA_Data
except ImportError:
    print("【错误】请在 TransFuser 代码根目录下运行此脚本 (需要 config.py, model.py, data.py)")
    sys.exit(1)

# ================= Metrics 依赖 =================
try:
    import lpips
    use_lpips = True
except ImportError:
    print("Warning: lpips not found, LPIPS metric will be skipped.")
    use_lpips = False

try:
    from pytorch_fid import fid_score
    from pytorch_fid.inception import InceptionV3
    use_fid = True
except ImportError:
    try:
        from torchvision.models import inception_v3
        use_fid = True
    except:
        use_fid = False
        print("Warning: pytorch_fid not found, FID metric will be skipped.")

# ================= 辅助函数 (SSIM & Vis & FID) =================
from scipy import linalg

def calculate_fid_features(images, inception_model, device):
    """计算图像的Inception特征"""
    inception_model.eval()
    with torch.no_grad():
        # 确保图像是正确的格式 [N, 3, H, W]，范围 [0, 1]
        if images.dim() == 3:
            images = images.unsqueeze(0)
        # 调整大小到 299x299 (Inception 输入尺寸)
        images = F.interpolate(images, size=(299, 299), mode='bilinear', align_corners=False)
        # 归一化到 [-1, 1]
        images = 2 * images - 1
        features = inception_model(images)
        if isinstance(features, list):
            features = features[0]
        features = features.squeeze()
    return features.cpu().numpy()

def calculate_fid_from_features(features_real, features_fake):
    """根据特征计算FID分数"""
    mu_real = np.mean(features_real, axis=0)
    mu_fake = np.mean(features_fake, axis=0)
    sigma_real = np.cov(features_real, rowvar=False)
    sigma_fake = np.cov(features_fake, rowvar=False)
    
    # 计算 FID
    diff = mu_real - mu_fake
    covmean, _ = linalg.sqrtm(sigma_real @ sigma_fake, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = diff @ diff + np.trace(sigma_real + sigma_fake - 2 * covmean)
    return fid

def gaussian(window_size, sigma):
    gauss = torch.Tensor([math.exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

def ssim_func(img1, img2, window_size=11, size_average=True):
    channel = img1.size(1)
    window = create_window(window_size, channel)
    if img1.is_cuda: window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)
    mu1_sq, mu2_sq, mu1_mu2 = mu1.pow(2), mu2.pow(2), mu1*mu2
    sigma1_sq = F.conv2d(img1*img1, window, padding=window_size//2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding=window_size//2, groups=channel) - mu2_sq
    sigma12   = F.conv2d(img1*img2, window, padding=window_size//2, groups=channel) - mu1_mu2
    C1 = 0.01**2; C2 = 0.03**2
    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
    if size_average: return ssim_map.mean()
    else: return ssim_map.mean(1).mean(1).mean(1)

def save_visualization(rgb_clean, rgb_adv, pred_clean, pred_adv, save_path, drift, gt_wp=None):
    """保存对比可视化结果"""
    fig = plt.figure(figsize=(20, 5))
    
    # 1. Clean Image
    ax1 = fig.add_subplot(1, 4, 1)
    img_clean_disp = rgb_clean.permute(1,2,0).cpu().numpy() / 255.0
    ax1.imshow(np.clip(img_clean_disp, 0, 1))
    ax1.axis('off'); ax1.set_title("Clean Input")

    # 2. Adversarial Image
    ax2 = fig.add_subplot(1, 4, 2)
    img_adv_disp = rgb_adv.permute(1,2,0).cpu().numpy() / 255.0
    ax2.imshow(np.clip(img_adv_disp, 0, 1))
    ax2.axis('off'); ax2.set_title("Adversarial Input")

    # 3. Noise Heatmap
    ax3 = fig.add_subplot(1, 4, 3)
    diff = np.abs(img_adv_disp - img_clean_disp) * 15.0 # 放大噪声以便观察
    ax3.imshow(np.clip(diff, 0, 1))
    ax3.axis('off'); ax3.set_title("Diff (Amplified x15)")

    # 4. Trajectory
    ax4 = fig.add_subplot(1, 4, 4)
    if gt_wp is not None:
        ax4.plot(gt_wp[:,1], gt_wp[:,0], 'g-', label='GT', linewidth=3, alpha=0.5)
    ax4.plot(pred_clean[:,1], pred_clean[:,0], 'b-o', label='Clean Pred', markersize=4)
    ax4.plot(pred_adv[:,1], pred_adv[:,0], 'r--^', label='Adv Pred', markersize=4)
    ax4.set_xlim(-10, 10); ax4.set_ylim(0, 30); ax4.legend(); ax4.grid(True)
    ax4.set_title(f"Shift: {drift:.4f} m {'(FAIL)' if drift >= 1.0 else ''}")

    plt.tight_layout(); plt.savefig(save_path); plt.close(fig)

# ================= Main Evaluation Logic =================
def main():
    parser = argparse.ArgumentParser(description="Evaluate SimLingo/MFA adversarial samples on TransFuser")
    parser.add_argument('--model_path', type=str, required=True, help='Path to TransFuser .pth checkpoint')
    parser.add_argument('--root_dir', type=str, required=True, help='Root of the CLEAN validation/test dataset (CARLA Data)')
    parser.add_argument('--adv_input_dir', type=str, required=True, help='Root directory where SimLingo images are saved')
    parser.add_argument('--backbone', type=str, default='transFuser', help='Model backbone')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--output_dir', type=str, default='eval_simlingo_results', help='Where to save logs and vis')
    args = parser.parse_args()

    # 1. Setup Environment
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"=== TransFuser Transfer Attack Evaluation ===")
    print(f"Device: {device}")
    print(f"Clean Data: {args.root_dir}")
    print(f"Adv Data:   {args.adv_input_dir}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    log_file = os.path.join(args.output_dir, 'evaluation_log.txt')
    log_fp = open(log_file, 'w')

    # 2. Load TransFuser Model
    print(f"Loading TransFuser Model...")
    config = GlobalConfig(setting='eval')
    config.use_velocity = True
    config.backbone = args.backbone
    config.use_target_point_image = True
    
    # 这里的 architecture 默认用 regnety_032，如果你的模型不同请修改
    model = LidarCenterNet(config, device, args.backbone, 'regnety_032', 'regnety_032', True)
    
    state = torch.load(args.model_path, map_location=device)
    # 处理可能的 module. 前缀
    state = {k[7:] if k.startswith('module.') else k: v for k, v in state.items()}
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()

    # 3. Load LPIPS
    lpips_fn = None
    if use_lpips:
        lpips_fn = lpips.LPIPS(net='vgg').to(device).eval()

    # 4. Load Inception Model for FID
    inception_model = None
    if use_fid:
        try:
            from pytorch_fid.inception import InceptionV3
            block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
            inception_model = InceptionV3([block_idx]).to(device).eval()
            print("Loaded InceptionV3 for FID calculation")
        except:
            try:
                from torchvision.models import inception_v3
                inception_model = inception_v3(pretrained=True, transform_input=False).to(device).eval()
                # 移除最后的全连接层，使用pool层输出
                inception_model.fc = torch.nn.Identity()
                print("Loaded torchvision InceptionV3 for FID calculation")
            except Exception as e:
                print(f"Warning: Could not load Inception model for FID: {e}")
                inception_model = None

    # 5. Load Dataset (Clean Ground Truth)
    print(f"Loading clean dataset loader...")
    dataset = CARLA_Data(root=[args.root_dir], config=config, shared_dict=None)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)
    print(f"Total frames in clean dataset: {len(dataset)}")
    
    # Statistics
    global_shift = 0.0
    global_success = 0
    global_count = 0
    global_lpips = 0.0
    global_ssim = 0.0
    
    # FID 特征收集
    clean_features_list = []
    adv_features_list = []
    
    print("\nStarting Evaluation Loop...")
    
    # 先打印一些调试信息，确认数据集结构
    if hasattr(dataset, 'images') and len(dataset.images) > 0:
        sample_path = dataset.images[0][-1]
        if isinstance(sample_path, bytes):
            sample_path = sample_path.decode('utf-8')
        print(f"Sample clean image path: {sample_path}")
        
        # 打印对抗样本目录内容以便调试
        print(f"\nAdv input dir contents:")
        if os.path.exists(args.adv_input_dir):
            for item in os.listdir(args.adv_input_dir)[:10]:  # 只打印前10个
                print(f"  - {item}")
            # 检查 global_fid_adv 子目录
            gfa_dir = os.path.join(args.adv_input_dir, "global_fid_adv")
            if os.path.exists(gfa_dir):
                print(f"\nglobal_fid_adv contents (first 5):")
                for item in os.listdir(gfa_dir)[:5]:
                    print(f"  - {item}")
        else:
            print(f"  WARNING: Directory does not exist!")
    
    with torch.no_grad():
        for batch_idx, data in enumerate(tqdm(loader)):
            # --- A. 获取原始文件路径 ---
            orig_path = None
            if hasattr(dataset, 'images'):
                try:
                    paths_seq = dataset.images[batch_idx] 
                    if len(paths_seq) > 0:
                        path_bytes = paths_seq[-1] 
                        orig_path = path_bytes.decode('utf-8') if isinstance(path_bytes, bytes) else str(path_bytes)
                except: pass

            if orig_path is None:
                if batch_idx < 3: print(f"Warning: Could not determine path for frame {batch_idx}")
                continue
            
            # 1. 解析原始信息
            file_name_raw = os.path.basename(orig_path)          # 0005.png
            dir_rgb = os.path.dirname(orig_path)                 # .../rgb
            dir_route = os.path.dirname(dir_rgb)                 # .../Town05_short
            route_name = os.path.basename(dir_route)             # Town05_short
            
            # 2. 构造可能的文件名变体
            file_name_no_ext = os.path.splitext(file_name_raw)[0]  # 0005
            file_ext = os.path.splitext(file_name_raw)[1]          # .png
            
            fname_variants = [
                file_name_raw,                                      # 0005.png
                f"{route_name}_{file_name_raw}",                    # Town05_short_0005.png
                f"{route_name}_{file_name_no_ext}{file_ext}",       # Town05_short_0005.png
                f"f_{file_name_raw}",                               # f_0005.png
                file_name_raw.replace('.jpg', '.png'),              # 扩展名兼容
                f"{route_name}_{file_name_no_ext}.png",             # 强制 .png 扩展名
            ]

            # 3. 构造可能的文件夹结构
            candidates = []
            for fname in fname_variants:
                candidates.append(os.path.join(args.adv_input_dir, route_name, "rgb", fname))
                candidates.append(os.path.join(args.adv_input_dir, route_name, "global_fid_adv", fname))
                candidates.append(os.path.join(args.adv_input_dir, route_name, fname))
                candidates.append(os.path.join(args.adv_input_dir, fname))
                candidates.append(os.path.join(args.adv_input_dir, "global_fid_adv", fname))
                candidates.append(os.path.join(args.adv_input_dir, "rgb", fname))

            # 4. 搜索文件
            target_path = None
            for p in candidates:
                if os.path.exists(p):
                    target_path = p
                    break
            
            # 调试日志 (打印前5帧)
            if batch_idx < 5:
                print(f"\n--- Frame {batch_idx} Debug ---")
                print(f"Clean Path: {orig_path}")
                print(f"Route Name: {route_name}")
                print(f"File Name: {file_name_raw}")
                print("Tried Paths (first 6):")
                for c in candidates[:6]:
                    exists = "✓" if os.path.exists(c) else "✗"
                    print(f"  [{exists}] {c}")
                if target_path:
                    print(f"✅ Found: {target_path}")
                else:
                    print("❌ No matching adversarial image found.")
            
            if not target_path:
                continue
            
            file_name = file_name_raw

            # --- B. 执行评估 ---
            try:
                # ================= 1. ID 匹配检查 =================
                clean_base = os.path.splitext(os.path.basename(orig_path))[0]
                adv_base = os.path.splitext(os.path.basename(target_path))[0]
                clean_id = clean_base.split('_')[-1]
                adv_id = adv_base.split('_')[-1]

                if clean_id != adv_id:
                    print(f"\n【严重错误】帧ID不匹配！跳过！(Clean:{clean_id} vs Adv:{adv_id})")
                    continue 

                # ================= 2. 加载对抗图像并匹配预处理 =================
                import cv2

                # A. 读取 Adv 图像
                adv_cv2 = cv2.imread(target_path, cv2.IMREAD_COLOR)
                if adv_cv2 is None:
                    print(f"Error loading file: {target_path}")
                    continue
                adv_cv2 = cv2.cvtColor(adv_cv2, cv2.COLOR_BGR2RGB)
                
                # B. 同时读取原始干净图像（用于公平对比）
                clean_cv2 = cv2.imread(orig_path, cv2.IMREAD_COLOR)
                if clean_cv2 is None:
                    print(f"Error loading clean file: {orig_path}")
                    continue
                clean_cv2 = cv2.cvtColor(clean_cv2, cv2.COLOR_BGR2RGB)

                # C. 获取 Config 参数
                try:
                    cfg_scale = config.scale
                    if hasattr(cfg_scale, '__len__'): cfg_scale = float(cfg_scale[0])
                    else: cfg_scale = float(cfg_scale)
                    cfg_res = config.img_resolution  # (H, W) = (160, 704)
                except AttributeError:
                    cfg_scale = 1.0
                    cfg_res = [160, 704]

                crop_h, crop_w = int(cfg_res[0]), int(cfg_res[1])
                
                # D. 对两张图应用相同的预处理
                def preprocess_image(img_cv2, scale, crop_h, crop_w):
                    """复刻 data.py 的预处理：scale + center crop"""
                    h, w = img_cv2.shape[0], img_cv2.shape[1]
                    
                    # 缩放
                    if scale != 1.0:
                        new_w, new_h = int(w // scale), int(h // scale)
                        img_cv2 = cv2.resize(img_cv2, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                        h, w = img_cv2.shape[0], img_cv2.shape[1]
                    
                    # 中心裁剪
                    start_y = h // 2 - crop_h // 2
                    start_x = w // 2 - crop_w // 2
                    
                    if start_y < 0 or start_x < 0 or h < crop_h or w < crop_w:
                        # 图像太小，强制缩放
                        img_cv2 = cv2.resize(img_cv2, (crop_w, crop_h), interpolation=cv2.INTER_LINEAR)
                    else:
                        img_cv2 = img_cv2[start_y:start_y+crop_h, start_x:start_x+crop_w]
                    
                    return img_cv2
                
                # 对两张图应用相同预处理
                adv_processed = preprocess_image(adv_cv2, cfg_scale, crop_h, crop_w)
                clean_processed = preprocess_image(clean_cv2, cfg_scale, crop_h, crop_w)
                
                if batch_idx < 3:
                    print(f"  Original sizes - Clean: {clean_cv2.shape[1]}x{clean_cv2.shape[0]}, Adv: {adv_cv2.shape[1]}x{adv_cv2.shape[0]}")
                    print(f"  After preprocess: {adv_processed.shape[1]}x{adv_processed.shape[0]}")

                # E. 转为 Tensor
                adv_tensor = torch.from_numpy(adv_processed.transpose(2, 0, 1)).unsqueeze(0).to(device).float()
                clean_tensor = torch.from_numpy(clean_processed.transpose(2, 0, 1)).unsqueeze(0).to(device).float()

                # ================= 3. 准备模型其他输入 =================
                lidar = data['lidar'].to(device, dtype=torch.float32)
                tp = data['target_point'].to(device, dtype=torch.float32)
                tp_img = data['target_point_image'].to(device, dtype=torch.float32)
                if tp_img.dim() != 4: 
                    tp_img = torch.zeros(1, 1, 256, 256).to(device)
                speed = data['speed'].to(device, dtype=torch.float32).reshape(-1,1)
                gt_wp = data['ego_waypoint'].to(device, dtype=torch.float32)
                
                # 使用我们自己预处理的 clean_tensor，确保与 adv_tensor 处理方式一致
                rgb_clean = clean_tensor

                # ================= 4. 尺寸对齐 =================
                if adv_tensor.shape[2:] != rgb_clean.shape[2:]:
                    if batch_idx < 3:
                        print(f"  Size mismatch! Adv: {adv_tensor.shape[2:]}, Clean: {rgb_clean.shape[2:]}, resizing...")
                    adv_tensor = F.interpolate(adv_tensor, size=rgb_clean.shape[2:], mode='bilinear')

                # 5. 模型推理
                # Forward Clean
                pred_c, _ = model.forward_ego(rgb_clean, lidar, tp, tp_img, speed)
                # Forward Adv
                pred_a, _ = model.forward_ego(adv_tensor, lidar, tp, tp_img, speed)

                # 5. 计算指标
                clean_wp_np = pred_c[0].cpu().numpy()
                adv_wp_np = pred_a[0].cpu().numpy()
                
                # Shift: L2 distance of trajectories
                shift = np.linalg.norm(clean_wp_np - adv_wp_np)
                
                # Metrics: SSIM & LPIPS
                im_c_norm = rgb_clean / 255.0
                im_a_norm = adv_tensor / 255.0
                
                curr_ssim = ssim_func(im_c_norm, im_a_norm).item()
                curr_lpips = 0.0
                if lpips_fn: 
                    curr_lpips = lpips_fn(im_a_norm*2-1, im_c_norm*2-1).mean().item()

                # 6. 统计
                global_shift += shift
                if shift >= 1.0: # TransFuser 默认 1.0m 偏离视为攻击成功
                    global_success += 1
                global_ssim += curr_ssim
                global_lpips += curr_lpips
                global_count += 1
                
                # 7. 收集 FID 特征
                if inception_model is not None:
                    try:
                        clean_feat = calculate_fid_features(im_c_norm, inception_model, device)
                        adv_feat = calculate_fid_features(im_a_norm, inception_model, device)
                        clean_features_list.append(clean_feat)
                        adv_features_list.append(adv_feat)
                    except Exception as e:
                        if batch_idx < 3:
                            print(f"FID feature extraction error: {e}")

                # 7. 可视化 (每 100 帧保存一次，或攻击成功幅度很大时)
                if batch_idx % 100 == 0 or shift > 3.0:
                    vis_name = os.path.join(args.output_dir, f"eval_{route_name}_{file_name}")
                    save_visualization(
                        rgb_clean[0], adv_tensor[0], 
                        clean_wp_np, adv_wp_np, 
                        vis_name, shift, 
                        gt_wp=gt_wp[0].cpu().numpy()
                    )

            except Exception as e:
                print(f"Error evaluating frame {batch_idx}: {e}")
                import traceback; traceback.print_exc()
                continue

    # Final Report
    print("\n" + "="*60)
    if global_count > 0:
        avg_shift = global_shift / global_count
        asr = (global_success / global_count) * 100.0
        avg_lpips = global_lpips / global_count
        avg_ssim = global_ssim / global_count
        
        # 计算 FID
        fid_value = -1.0
        if len(clean_features_list) > 1 and len(adv_features_list) > 1:
            try:
                clean_features = np.array(clean_features_list)
                adv_features = np.array(adv_features_list)
                # 确保特征是2D的
                if clean_features.ndim == 1:
                    clean_features = clean_features.reshape(1, -1)
                if adv_features.ndim == 1:
                    adv_features = adv_features.reshape(1, -1)
                fid_value = calculate_fid_from_features(clean_features, adv_features)
                print(f"FID calculated from {len(clean_features_list)} samples")
            except Exception as e:
                print(f"FID calculation error: {e}")
                fid_value = -1.0
        
        results = [
            "FINAL EVALUATION RESULTS (TransFuser Transfer)",
            f"  > Clean Data:   {args.root_dir}",
            f"  > Adv Samples:  {args.adv_input_dir}",
            f"  > Total Frames: {global_count}",
            "-" * 40,
            f"  > Avg Route Shift: {avg_shift:.4f} m",
            f"  > Success Rate:    {asr:.2f} % (Threshold >= 1.0m)",
            f"  > Avg LPIPS:       {avg_lpips:.4f}",
            f"  > Avg SSIM:        {avg_ssim:.4f}",
            f"  > FID:             {fid_value:.4f}" if fid_value >= 0 else "  > FID:             N/A",
        ]
        
        for line in results:
            print(line)
            log_fp.write(line + "\n")
    else:
        err_msg = "FAILED: No valid adversarial pairs found. Check directory structure."
        print(err_msg)
        log_fp.write(err_msg + "\n")
        
    log_fp.close()
    print("="*60)

if __name__ == "__main__":
    main()