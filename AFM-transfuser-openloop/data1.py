import os
import ujson
import json  # 引入标准json作为备用
from skimage.transform import rotate
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm
import sys
from pathlib import Path
import cv2
import random
from copy import deepcopy
import io

from utils import get_vehicle_to_virtual_lidar_transform, get_vehicle_to_lidar_transform, get_lidar_to_vehicle_transform, get_lidar_to_bevimage_transform

class CARLA_Data(Dataset):

    def __init__(self, root, config, shared_dict=None):
        self.seq_len = np.array(config.seq_len)
        self.pred_len = np.array(config.pred_len)
        self.img_resolution = np.array(config.img_resolution)
        self.img_width = np.array(config.img_width)
        self.scale = np.array(config.scale)
        self.multitask = np.array(config.multitask)
        self.data_cache = shared_dict
        self.augment = np.array(config.augment)
        self.aug_max_rotation = np.array(config.aug_max_rotation)
        self.use_point_pillars = np.array(config.use_point_pillars)
        self.max_lidar_points = np.array(config.max_lidar_points)
        self.backbone = np.array(config.backbone).astype(np.string_)
        self.inv_augment_prob = np.array(config.inv_augment_prob)
        self.converter = np.uint8(config.converter)

        self.images = []
        self.bevs = []
        self.depths = []
        self.semantics = []
        self.lidars = []
        self.labels = []
        self.measurements = []

        for sub_root in tqdm(root, file=sys.stdout):
            sub_root = Path(sub_root)
            root_files = os.listdir(sub_root)
            
            # [MFA修复] 判断 sub_root 本身是否就是一个有效的 Route 目录
            # 如果 sub_root 直接包含 'rgb' 或 'lidar' 文件夹，则它本身就是 route
            if os.path.exists(sub_root / "rgb") or os.path.exists(sub_root / "lidar"):
                routes = [""]  # 空字符串表示 sub_root 本身就是 route
            else:
                # 否则，查找子目录作为 routes
                routes = [folder for folder in root_files if not os.path.isfile(os.path.join(sub_root, folder))]
            
            for route in routes:
                route_dir = sub_root / route if route else sub_root
                
                # 确定序列总长度
                if os.path.exists(route_dir / "rgb"):
                    files = sorted(os.listdir(route_dir / "rgb"))
                    num_seq = len(files)
                elif os.path.exists(route_dir / "lidar"):
                    files = sorted(os.listdir(route_dir / "lidar"))
                    num_seq = len(files)
                else:
                    continue

                # [MFA关键修改] 处理所有帧，不丢弃头尾
                # 原始代码: for seq in range(2, num_seq - self.pred_len - self.seq_len - 2):
                for seq in range(0, num_seq):
                    
                    image = []
                    bev = []
                    depth = []
                    semantic = []
                    lidar = []
                    label = []
                    measurement= []
                    
                    # 加载当前和过去帧 (Seq Len)
                    for idx in range(self.seq_len):
                        # [修正] 防止索引越界：如果小于0，取0；如果大于最大值，取最大值
                        # 正常情况 seq_len=1, idx=0, 所以 target_idx = seq
                        target_idx = np.clip(seq + idx, 0, num_seq - 1)
                        
                        image.append(route_dir / "rgb" / ("%04d.png" % target_idx))
                        bev.append(route_dir / "topdown" / ("encoded_%04d.png" % target_idx))
                        depth.append(route_dir / "depth" / ("%04d.png" % target_idx))
                        semantic.append(route_dir / "semantics" / ("%04d.png" % target_idx))
                        lidar.append(route_dir / "lidar" / ("%04d.npy" % target_idx))
                        measurement.append(route_dir / "measurements" / ("%04d.json"% target_idx))

                    # 加载未来标签 (Pred Len)
                    for idx in range(self.seq_len + self.pred_len):
                        # [修正] 防止未来帧越界
                        target_idx = np.clip(seq + idx, 0, num_seq - 1)
                        label.append(route_dir / "label_raw" / ("%04d.json" % target_idx))

                    self.images.append(image)
                    self.bevs.append(bev)
                    self.depths.append(depth)
                    self.semantics.append(semantic)
                    self.lidars.append(lidar)
                    self.labels.append(label)
                    self.measurements.append(measurement)

        self.images       = np.array(self.images      ).astype(np.string_)
        self.bevs         = np.array(self.bevs        ).astype(np.string_)
        self.depths       = np.array(self.depths      ).astype(np.string_)
        self.semantics    = np.array(self.semantics   ).astype(np.string_)
        self.lidars       = np.array(self.lidars      ).astype(np.string_)
        self.labels       = np.array(self.labels      ).astype(np.string_)
        self.measurements = np.array(self.measurements).astype(np.string_)
        print("Loading %d sequences from %d folders"%(len(self.lidars), len(root)))

    def __len__(self):
        """Returns the length of the dataset. """
        return self.lidars.shape[0]

    def __getitem__(self, index):
        """Returns the item at index idx. """
        cv2.setNumThreads(0) 

        data = dict()
        backbone = str(self.backbone, encoding='utf-8')

        images = self.images[index]
        bevs = self.bevs[index]
        depths = self.depths[index]
        semantics = self.semantics[index]
        lidars = self.lidars[index]
        labels = self.labels[index]
        measurements = self.measurements[index]

        loaded_images = []
        loaded_bevs = []
        loaded_depths = []
        loaded_semantics = []
        loaded_lidars = []
        loaded_labels = []
        loaded_measurements = []

        if(backbone == 'geometric_fusion'):
            loaded_lidars_raw = []

        # [MFA修改] 安全读取 Labels (Future Waypoints)
        # 如果文件不存在，生成伪造的标签数据
        for i in range(self.seq_len+self.pred_len):
            label_path = str(labels[i], encoding='utf-8')
            
            if ((not (self.data_cache is None)) and (label_path in self.data_cache)):
                labels_i = self.data_cache[label_path]
            elif os.path.exists(label_path):
                with open(label_path, 'r') as f2:
                    labels_i = ujson.load(f2)
                if not self.data_cache is None:
                    self.data_cache[label_path] = labels_i
            else:
                # [MFA] 伪造 Label 数据
                # id=0 通常留给 ego vehicle, position 设为 0
                labels_i = [{
                    'id': 0, 'num_points': 100, 'distance': 0, 
                    'position': [0.0, 0.0, 0.0], 'extent': [0.0, 0.0, 0.0], 
                    'yaw': 0.0, 'speed': 0.0, 'brake': 0.0, 'ego_matrix': np.eye(4).tolist()
                }]

            loaded_labels.append(labels_i)


        for i in range(self.seq_len):
            meas_path = str(measurements[i], encoding='utf-8')
            
            # 尝试从 Cache 读取
            if not self.data_cache is None and meas_path in self.data_cache:
                    measurements_i, images_i, lidars_i, lidars_raw_i, bevs_i, depths_i, semantics_i = self.data_cache[meas_path]
                    images_i = cv2.imdecode(images_i, cv2.IMREAD_UNCHANGED)
                    depths_i = cv2.imdecode(depths_i, cv2.IMREAD_UNCHANGED)
                    semantics_i = cv2.imdecode(semantics_i, cv2.IMREAD_UNCHANGED)
                    bevs_i.seek(0)
                    bevs_i = np.load(bevs_i)['arr_0']
            else:
                # [MFA修改] 1. 安全读取 Measurements
                if os.path.exists(meas_path):
                    with open(meas_path, 'r') as f1:
                        measurements_i = ujson.load(f1)
                else:
                    # 伪造 Measurements
                    measurements_i = {
                        'speed': 4.0, 'steer': 0.0, 'throttle': 0.5, 'brake': 0.0,
                        'theta': 0.0, 'x': 0.0, 'y': 0.0, 'z': 0.0,
                        'x_command': 20.0, 'y_command': 0.0, # 假设目标在前方20米
                        'command': 4, # Lane Follow
                        'light_hazard': 0,
                        'ego_matrix': np.eye(4).tolist() # 单位矩阵
                    }

                # [MFA修改] 2. 安全读取 LiDAR
                lidar_path = str(lidars[i], encoding='utf-8')
                if os.path.exists(lidar_path):
                    lidars_i = np.load(lidar_path, allow_pickle=True)[1]
                    if (backbone == 'geometric_fusion'):
                        lidars_raw_i = np.load(lidar_path, allow_pickle=True)[1][..., :3]
                    else:
                        lidars_raw_i = None
                    lidars_i[:, 1] *= -1 # 注意：这里修改了 array，如果是伪造的需要小心
                else:
                    # 伪造 LiDAR (N, 4)
                    lidars_i = np.zeros((1, 4), dtype=np.float32)
                    lidars_raw_i = np.zeros((1, 3), dtype=np.float32) if (backbone == 'geometric_fusion') else None
                    # 不需要乘 -1，因为是 0

                # [MFA修改] 3. 安全读取 RGB
                img_path = str(images[i], encoding='utf-8')
                # 尝试 jpg 和 png
                if not os.path.exists(img_path) and img_path.endswith('.png'):
                    img_path = img_path.replace('.png', '.jpg')
                
                if os.path.exists(img_path):
                    images_i = cv2.imread(img_path, cv2.IMREAD_COLOR)
                    if images_i is None:
                        # 文件坏了
                        images_i = np.zeros((160, 704, 3), dtype=np.uint8)
                    else:
                        images_i = scale_image_cv2(cv2.cvtColor(images_i, cv2.COLOR_BGR2RGB), self.scale)
                else:
                    # 文件不存在
                    images_i = np.zeros((160, 704, 3), dtype=np.uint8)

                # [MFA修改] 4. 安全读取 BEV
                bev_path = str(bevs[i], encoding='utf-8')
                if os.path.exists(bev_path):
                    bev_array = cv2.imread(bev_path, cv2.IMREAD_UNCHANGED)
                    if bev_array is None:
                         bevs_i = np.zeros((2, 256, 256), dtype=np.uint8)
                    else:
                        bev_array = cv2.cvtColor(bev_array, cv2.COLOR_BGR2RGB)
                        bev_array = np.moveaxis(bev_array, -1, 0)
                        bevs_i = decode_pil_to_npy(bev_array).astype(np.uint8)
                else:
                    bevs_i = np.zeros((2, 256, 256), dtype=np.uint8)

                # [MFA修改] 5. 安全读取 Depth / Semantic
                if self.multitask:
                    depth_path = str(depths[i], encoding='utf-8')
                    if os.path.exists(depth_path):
                        depths_i = cv2.imread(depth_path, cv2.IMREAD_COLOR)
                        if depths_i is not None:
                            depths_i = scale_image_cv2(cv2.cvtColor(depths_i, cv2.COLOR_BGR2RGB), self.scale)
                        else:
                            depths_i = np.zeros((160, 704, 3), dtype=np.uint8)
                    else:
                        depths_i = np.zeros((160, 704, 3), dtype=np.uint8)

                    sem_path = str(semantics[i], encoding='utf-8')
                    if os.path.exists(sem_path):
                        semantics_i = cv2.imread(sem_path, cv2.IMREAD_UNCHANGED)
                        if semantics_i is not None:
                            semantics_i = scale_seg(semantics_i, self.scale)
                        else:
                             semantics_i = np.zeros((160, 704), dtype=np.uint8)
                    else:
                        semantics_i = np.zeros((160, 704), dtype=np.uint8)
                else:
                    depths_i = None
                    semantics_i = None

                # 存入 Cache (如果有的话)
                if not self.data_cache is None:
                    result, compressed_imgage = cv2.imencode('.png', images_i)
                    if self.multitask:
                        result, compressed_depths = cv2.imencode('.png', depths_i)
                        result, compressed_semantics = cv2.imencode('.png', semantics_i)
                    else:
                        compressed_depths = None
                        compressed_semantics = None
                    compressed_bevs = io.BytesIO()
                    np.savez_compressed(compressed_bevs, bevs_i)
                    self.data_cache[meas_path] = (measurements_i, compressed_imgage, lidars_i, lidars_raw_i, compressed_bevs, compressed_depths, compressed_semantics)

            loaded_images.append(images_i)
            loaded_bevs.append(bevs_i)
            loaded_depths.append(depths_i)
            loaded_semantics.append(semantics_i)
            loaded_lidars.append(lidars_i)
            loaded_measurements.append(measurements_i)
            if (backbone == 'geometric_fusion'):
                loaded_lidars_raw.append(lidars_raw_i)

        labels = loaded_labels
        measurements = loaded_measurements

        # load image, only use current frame
        # augment here
        crop_shift = 0
        degree = 0
        rad = np.deg2rad(degree)
        do_augment = self.augment and random.random() > self.inv_augment_prob
        if do_augment:
            degree = (random.random() * 2. - 1.) * self.aug_max_rotation
            rad = np.deg2rad(degree)
            crop_shift = degree / 60 * self.img_width / self.scale # we scale first

        # === [MFA修改] 强制全图缩放，禁止裁剪 ===
        images_i = loaded_images[self.seq_len-1]
        
        # 获取目标尺寸 (H, W) -> TransFuser Config 通常是 [160, 704]
        target_h, target_w = self.img_resolution[0], self.img_resolution[1]
        
        # 使用 cv2.resize 直接将任意尺寸 (如 512x512) 压扁成 (704x160)
        # 这样画面会变形，但内容 100% 保留
        images_i = cv2.resize(images_i, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        
        # TransFuser 需要 Channel First [C, H, W]
        if len(images_i.shape) == 3:
            images_i = np.transpose(images_i, (2, 0, 1)) # [H, W, C] -> [C, H, W]
        
        # ----------------------------------------
        
        # 处理 BEV (BEV 还是保持裁剪比较好，或者你也想 resize?)
        # 通常 BEV 标签也是正方形的，为了对应，这里不动 BEV 逻辑，或者也改成 resize
        # 但考虑到你是做攻击，BEV 只是用来防止报错的 dummy 数据，所以这里不动也没关系
        bevs_i = load_crop_bev_npy(loaded_bevs[self.seq_len-1], degree)
        
        data['rgb'] = images_i
        data['bev'] = bevs_i

        if self.multitask:
            depths_i = loaded_depths[self.seq_len-1]
            depths_i = get_depth(crop_image_cv2(depths_i, crop=self.img_resolution, crop_shift=crop_shift))

            semantics_i = loaded_semantics[self.seq_len-1]
            semantics_i = self.converter[crop_seg(semantics_i, crop=self.img_resolution, crop_shift=crop_shift)]

            data['depth'] = depths_i
            data['semantic'] = semantics_i

        # need to concatenate seq data here and align to the same coordinate
        lidars = []
        if (backbone == 'geometric_fusion'):
            lidars_raw = []
        if (self.use_point_pillars == True):
            lidars_pillar = []

        for i in range(self.seq_len):
            lidar = loaded_lidars[i]
            # [MFA修改] 确保 align 不会因为 dummy data 报错
            # dummy measurements 已经包含了 ego_matrix，align 函数应该能正常工作
            lidar = align(lidar, measurements[i], measurements[self.seq_len-1], degree=degree)
            lidar_bev = lidar_to_histogram_features(lidar)
            lidars.append(lidar_bev)

            if (backbone == 'geometric_fusion'):
                lidar_raw = loaded_lidars_raw[i]
                lidars_raw.append(lidar_raw)

            if (self.use_point_pillars == True):
                lidar_pillar = deepcopy(loaded_lidars[i])
                lidar_pillar = align(lidar_pillar, measurements[i], measurements[self.seq_len-1], degree=degree)
                lidars_pillar.append(lidar_pillar)

        lidar_bev = np.concatenate(lidars[::-1], axis=0)
        if (backbone == 'geometric_fusion'):
            lidars_raw = np.concatenate(lidars_raw[::-1], axis=0)
        if (self.use_point_pillars == True):
            lidars_pillar = np.concatenate(lidars_pillar[::-1], axis=0)

        if (backbone == 'geometric_fusion'):
            curr_bev_points, curr_cam_points = lidar_bev_cam_correspondences(deepcopy(lidars_raw), debug=False)

        # [MFA修改] 处理 labels 为空的情况 (虽然我们之前伪造了)
        if len(labels) > 0:
             # ego car is always the first one in label file
            try:
                ego_id = labels[self.seq_len-1][0]['id']
                bboxes = parse_labels(labels[self.seq_len-1], rad=-rad)
                waypoints = get_waypoints(labels[self.seq_len-1:], self.pred_len+1)
                waypoints = transform_waypoints(waypoints)
            except (KeyError, IndexError):
                # Fallback if dummy label structure is slightly off
                ego_id = 0
                bboxes = {}
                waypoints = {0: [[np.eye(4).tolist(), True] for _ in range(self.pred_len+1)]}

        else:
            ego_id = 0
            bboxes = {}
            waypoints = {}

        # save waypoints in meters
        filtered_waypoints = []
        # [MFA] 防御 waypoints 不存在 key
        ids_to_check = list(bboxes.keys()) + [ego_id]
        for id in ids_to_check:
            if id in waypoints:
                waypoint = []
                for matrix, flag in waypoints[id][1:]:
                    waypoint.append(np.array(matrix)[:2, 3])
                filtered_waypoints.append(waypoint)
            else:
                # 针对 ego 补全
                if id == ego_id:
                     filtered_waypoints.append([np.zeros(2) for _ in range(self.pred_len)])

        waypoints = np.array(filtered_waypoints)

        label = []
        for id in bboxes.keys():
            label.append(bboxes[id])
        label = np.array(label)
        
        # padding
        label_pad = np.zeros((20, 7), dtype=np.float32)
        
        # [MFA] 确保 waypoints 不为空
        if len(waypoints) > 0:
            ego_waypoint = waypoints[-1]
        else:
            ego_waypoint = np.zeros((self.pred_len, 2))

        # for the augmentation we only need to transform the waypoints for ego car
        degree_matrix = np.array([[np.cos(rad), np.sin(rad)],
                              [-np.sin(rad), np.cos(rad)]])
        ego_waypoint = (degree_matrix @ ego_waypoint.T).T

        if label.shape[0] > 0:
            label_pad[:label.shape[0], :] = label

        if(self.use_point_pillars == True):
            fixed_lidar_raw = np.empty((self.max_lidar_points, 4), dtype=np.float32)
            num_points = min(self.max_lidar_points, lidars_pillar.shape[0])
            fixed_lidar_raw[:num_points, :4] = lidars_pillar
            data['lidar_raw'] = fixed_lidar_raw
            data['num_points'] = num_points

        if (backbone == 'geometric_fusion'):
            data['bev_points'] = curr_bev_points
            data['cam_points'] = curr_cam_points

        data['lidar'] = lidar_bev
        data['label'] = label_pad
        data['ego_waypoint'] = ego_waypoint

        # other measurement
        data['steer'] = measurements[self.seq_len-1]['steer']
        data['throttle'] = measurements[self.seq_len-1]['throttle']
        data['brake'] = measurements[self.seq_len-1]['brake']
        data['light'] = measurements[self.seq_len-1]['light_hazard']
        data['speed'] = measurements[self.seq_len-1]['speed']
        data['theta'] = measurements[self.seq_len-1]['theta']
        data['x_command'] = measurements[self.seq_len-1]['x_command']
        data['y_command'] = measurements[self.seq_len-1]['y_command']

        ego_theta = measurements[self.seq_len-1]['theta'] + rad 
        ego_x = measurements[self.seq_len-1]['x']
        ego_y = measurements[self.seq_len-1]['y']
        x_command = measurements[self.seq_len-1]['x_command']
        y_command = measurements[self.seq_len-1]['y_command']
        
        R = np.array([
            [np.cos(np.pi/2+ego_theta), -np.sin(np.pi/2+ego_theta)],
            [np.sin(np.pi/2+ego_theta),  np.cos(np.pi/2+ego_theta)]
            ])
        local_command_point = np.array([x_command-ego_x, y_command-ego_y])
        local_command_point = R.T.dot(local_command_point)

        data['target_point'] = local_command_point
        
        data['target_point_image'] = draw_target_point(local_command_point)
        return data

# ... (后面的辅助函数 get_depth, get_waypoints 等保持不变，可以直接复制原来的) ...
def get_depth(data):
    data = np.transpose(data, (1,2,0))
    data = data.astype(np.float32)

    normalized = np.dot(data, [65536.0, 256.0, 1.0]) 
    normalized /=  (256 * 256 * 256 - 1)
    normalized = np.clip(normalized, a_min=0.0, a_max=0.05)
    normalized = normalized * 20.0 
    return normalized

def get_waypoints(labels, len_labels):
    assert(len(labels) == len_labels)
    num = len_labels
    waypoints = {}
    
    for result in labels[0]:
        car_id = result["id"]
        waypoints[car_id] = [[result['ego_matrix'], True]]
        for i in range(1, num):
            for to_match in labels[i]:
                if to_match["id"] == car_id:
                    waypoints[car_id].append([to_match["ego_matrix"], True])

    Identity = list(list(row) for row in np.eye(4))
    for k in waypoints.keys():
        while len(waypoints[k]) < num:
            waypoints[k].append([Identity, False])
    return waypoints

def transform_waypoints(waypoints):
    T = get_vehicle_to_virtual_lidar_transform()
    for k in waypoints.keys():
        vehicle_matrix = np.array(waypoints[k][0][0])
        try:
            vehicle_matrix_inv = np.linalg.inv(vehicle_matrix)
        except np.linalg.LinAlgError:
            vehicle_matrix_inv = np.eye(4)
            
        for i in range(1, len(waypoints[k])):
            matrix = np.array(waypoints[k][i][0])
            waypoints[k][i][0] = T @ vehicle_matrix_inv @ matrix
            
    return waypoints

def align(lidar_0, measurements_0, measurements_1, degree=0):
    matrix_0 = measurements_0['ego_matrix']
    matrix_1 = measurements_1['ego_matrix']
    matrix_0 = np.array(matrix_0)
    matrix_1 = np.array(matrix_1)
    Tr_lidar_to_vehicle = get_lidar_to_vehicle_transform()
    Tr_vehicle_to_lidar = get_vehicle_to_lidar_transform()
    
    try:
        transform_0_to_1 = Tr_vehicle_to_lidar @ np.linalg.inv(matrix_1) @ matrix_0 @ Tr_lidar_to_vehicle
    except np.linalg.LinAlgError:
        transform_0_to_1 = np.eye(4)

    rad = np.deg2rad(degree)
    degree_matrix = np.array([[np.cos(rad), np.sin(rad), 0, 0],
                              [-np.sin(rad), np.cos(rad), 0, 0],
                              [0, 0, 1, 0],
                              [0, 0, 0, 1]])
    transform_0_to_1 = degree_matrix @ transform_0_to_1
                            
    lidar = lidar_0.copy()
    if lidar.shape[0] > 0:
        lidar[:, -1] = 1.
        lidar[:, 1] *= -1.
        lidar = transform_0_to_1 @ lidar.T
        lidar = lidar.T
        lidar[:, -1] = lidar_0[:, -1]
        lidar[:, 1] *= -1.
    return lidar

def lidar_to_histogram_features(lidar):
    def splat_points(point_cloud):
        pixels_per_meter = 8
        hist_max_per_pixel = 5
        x_meters_max = 16
        y_meters_max = 32
        xbins = np.linspace(-x_meters_max, x_meters_max, 32*pixels_per_meter+1)
        ybins = np.linspace(-y_meters_max, 0, 32*pixels_per_meter+1)
        hist = np.histogramdd(point_cloud[..., :2], bins=(xbins, ybins))[0]
        hist[hist>hist_max_per_pixel] = hist_max_per_pixel
        overhead_splat = hist/hist_max_per_pixel
        return overhead_splat

    if lidar.shape[0] == 0:
        # Return empty features if no lidar points
        return np.zeros((2, 256, 256), dtype=np.float32)

    below = lidar[lidar[...,2]<=-2.3]
    above = lidar[lidar[...,2]>-2.3]
    below_features = splat_points(below)
    above_features = splat_points(above)
    features = np.stack([above_features, below_features], axis=-1)
    features = np.transpose(features, (2, 0, 1)).astype(np.float32)
    features = np.rot90(features, -1, axes=(1,2)).copy()
    return features

def get_bbox_label(bbox, rad=0):
    dz, dx, dy, x, y, z, yaw, speed, brake =  bbox
    pixels_per_meter = 8
    degree_matrix = np.array([[np.cos(rad), np.sin(rad), 0],
                              [-np.sin(rad), np.cos(rad), 0],
                              [0, 0, 1]])
    T = get_lidar_to_bevimage_transform() @ degree_matrix
    position = np.array([x, y, 1.0]).reshape([3, 1])
    position = T @ position
    position = np.clip(position, 0., 255.)
    x, y = position[:2, 0]
    bbox = np.array([x, y, dy*pixels_per_meter, dx*pixels_per_meter, 0, 0, 0])
    bbox[4] = yaw + rad
    bbox[5] = speed
    bbox[6] = brake
    return bbox

def parse_labels(labels, rad=0):
    bboxes = {}
    if not isinstance(labels, list): return bboxes
    for result in labels:
        if 'num_points' not in result: continue # Skip invalid dummy labels
        num_points = result['num_points']
        
        # Guard against missing keys in dummy data
        if 'position' not in result or 'extent' not in result: continue

        bbox = result['extent'] + result['position'] + [result['yaw'], result['speed'], result['brake']]
        bbox = get_bbox_label(bbox, rad)
        if num_points <= 1 or bbox[0] <= 0.0 or bbox[0] >= 255.0 or bbox[1] <= 0.0 or bbox[1] >=255.0:
            continue
        bboxes[result['id']] = bbox
    return bboxes

# ... (scale_image, scale_image_cv2, crop_image, crop_image_cv2, scale_seg, crop_seg, load_crop_bev_npy, draw_target_point, correspondences_at_one_scale, lidar_bev_cam_correspondences, decode_pil_to_npy 保持原样即可) ...
# 为了完整性，建议保留你原来文件底部的这些辅助函数
# 只要上面的 __getitem__ 和 __init__ 替换了，下面的辅助函数通常不会报错
# 唯一需要注意的是 parse_labels 我稍微加了一点防御性代码
# 其余函数请直接使用你原来的代码内容补充完整
def scale_image(image, scale):
    (width, height) = (int(image.width // scale), int(image.height // scale))
    im_resized = image.resize((width, height))
    return im_resized

def scale_image_cv2(image, scale):
    (width, height) = (int(image.shape[1] // scale), int(image.shape[0] // scale))
    im_resized = cv2.resize(image, (width, height))
    return im_resized

def crop_image(image, crop=(128, 640), crop_shift=0):
    width = image.width
    height = image.height
    crop_h, crop_w = crop
    start_y = height//2 - crop_h//2
    start_x = width//2 - crop_w//2
    start_x += int(crop_shift)
    image = np.asarray(image)
    cropped_image = image[start_y:start_y+crop_h, start_x:start_x+crop_w]
    cropped_image = np.transpose(cropped_image, (2,0,1))
    return cropped_image

def crop_image_cv2(image, crop=(128, 640), crop_shift=0):
    width = image.shape[1]
    height = image.shape[0]
    crop_h, crop_w = crop
    start_y = height // 2 - crop_h // 2
    start_x = width // 2 - crop_w // 2
    start_x += int(crop_shift)
    cropped_image = image[start_y:start_y + crop_h, start_x:start_x + crop_w]
    cropped_image = np.transpose(cropped_image, (2, 0, 1))
    return cropped_image

def scale_seg(image, scale):
    (width, height) = (int(image.shape[1] / scale), int(image.shape[0] / scale))
    if scale != 1:
        im_resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_NEAREST)
    else:
        im_resized = image
    return im_resized

def crop_seg(image, crop=(128, 640), crop_shift=0):
    width = image.shape[1]
    height = image.shape[0]
    crop_h, crop_w = crop
    start_y = height//2 - crop_h//2
    start_x = width//2 - crop_w//2
    start_x += int(crop_shift)
    cropped_image = image[start_y:start_y+crop_h, start_x:start_x+crop_w]
    return cropped_image

def load_crop_bev_npy(bev_array, degree):
    PIXELS_PER_METER_FOR_BEV = 5
    PIXLES = 32 * PIXELS_PER_METER_FOR_BEV
    start_x = 250 - PIXLES // 2
    start_y = 250 - PIXLES
    bev_array = np.moveaxis(bev_array, 0, -1).astype(np.float32)
    bev_shift = np.zeros_like(bev_array)
    bev_shift[7:] = bev_array[:-7]
    bev_shift = rotate(bev_shift, degree)
    cropped_image = bev_shift[start_y:start_y+PIXLES, start_x:start_x+PIXLES]
    cropped_image = np.moveaxis(cropped_image, -1, 0)
    cropped_image = np.concatenate((np.zeros_like(cropped_image[:1]), 
                                    cropped_image[:1],
                                    cropped_image[:1] + cropped_image[1:2]), axis=0)
    cropped_image = np.argmax(cropped_image, axis=0)
    return cropped_image

def draw_target_point(target_point, color = (255, 255, 255)):
    image = np.zeros((256, 256), dtype=np.uint8)
    target_point = target_point.copy()
    target_point[1] += 1.3
    point = target_point * 8.
    point[1] *= -1
    point[1] = 256 - point[1] 
    point[0] += 128 
    point = point.astype(np.int32)
    point = np.clip(point, 0, 256)
    cv2.circle(image, tuple(point), radius=5, color=color, thickness=3)
    image = image.reshape(1, 256, 256)
    return image.astype(np.float64) / 255.

def correspondences_at_one_scale(valid_bev_points, valid_cam_points, lidar_x, lidar_y, camera_x, camera_y, scale):
    cam_to_bev_proj_locs = np.zeros((lidar_x, lidar_y, 5, 2))
    bev_to_cam_proj_locs = np.zeros((camera_x, camera_y, 5, 2))
    tmp_bev = np.empty((lidar_x, lidar_y, ), dtype=object)
    tmp_cam = np.empty((camera_x, camera_y, ), dtype=object)
    for i in range(lidar_x):
        for j in range(lidar_y):
            tmp_bev[i,j] = []
    for i in range(camera_x):
        for j in range(camera_y):
            tmp_cam[i, j] = []
    for i in range(valid_bev_points.shape[0]):
        tmp_bev[valid_bev_points[i][0]//scale, valid_bev_points[i][1]//scale].append(valid_cam_points[i]//scale)
        tmp_cam[valid_cam_points[i][0]//scale, valid_cam_points[i][1]//scale].append(valid_bev_points[i]//scale)
    for i in range(lidar_x):
        for j in range(lidar_y):
            cam_to_bev_points = tmp_bev[i,j]
            if len(cam_to_bev_points) > 5:
                cam_to_bev_proj_locs[i,j] = np.array(random.sample(cam_to_bev_points, 5))
            elif len(cam_to_bev_points) > 0:
                num_points = len(cam_to_bev_points)
                cam_to_bev_proj_locs[i,j,:num_points] = np.array(cam_to_bev_points)
    for i in range(camera_x):
        for j in range(camera_y):
            bev_to_cam_points = tmp_cam[i,j]
            if len(bev_to_cam_points) > 5:
                bev_to_cam_proj_locs[i,j] = np.array(random.sample(bev_to_cam_points, 5))
            elif len(bev_to_cam_points) > 0:
                num_points = len(bev_to_cam_points)
                bev_to_cam_proj_locs[i,j,:num_points] = np.array(bev_to_cam_points)
    return cam_to_bev_proj_locs, bev_to_cam_proj_locs

def lidar_bev_cam_correspondences(world, lidar_vis=None, image_vis=None, step=None, debug=False):
    pixels_per_meter = 8
    lidar_width      = 256
    lidar_height     = 256
    lidar_meters_x   = (lidar_width  / pixels_per_meter) / 2
    lidar_meters_y   =  lidar_height / pixels_per_meter
    downscale_factor = 32
    img_width  = 352
    img_height = 160
    fov_width  = 60
    left_camera_rotation  = -60.0
    right_camera_rotation =  60.0
    fov_height = 2.0 * np.arctan((img_height / img_width) * np.tan(0.5 * np.radians(fov_width)))
    fov_height = np.rad2deg(fov_height)
    focal_x = img_width  / (2.0 * np.tan(np.deg2rad(fov_width)  / 2.0))
    focal_y = img_height / (2.0 * np.tan(np.deg2rad(fov_height) / 2.0))
    cam_z   = 2.3
    lidar_z = 2.5
    world[:, 0] *= -1 
    lidar = world[abs(world[:,0])<lidar_meters_x] 
    lidar = lidar[lidar[:,1]<lidar_meters_y] 
    lidar = lidar[lidar[:,1]>0] 
    lidar[..., 2] = lidar[..., 2] + (lidar_z - cam_z)
    lidar_for_left_camera  = deepcopy(lidar)
    lidar_for_right_camera = deepcopy(lidar)
    lidar_indices = np.arange(0, lidar.shape[0], 1)
    z = lidar[..., 1]
    if z.shape[0] > 0:
        x = ((focal_x * lidar[..., 0]) / z) + (img_width  / 2.0)
        y = ((focal_y * lidar[..., 2]) / z) + (img_height / 2.0)
        result_center = np.stack([x, y, lidar_indices], 1)
        result_center = result_center[np.logical_and(result_center[...,0] > 0, result_center[...,0] < img_width)]
        result_center = result_center[np.logical_and(result_center[...,1] > 0, result_center[...,1] < img_height)]
        result_center_shifted = result_center
        result_center_shifted[..., 0] = result_center_shifted[..., 0] + (img_width / 2.0)
    else:
        result_center_shifted = np.zeros((0, 3))

    theta = np.radians(left_camera_rotation)
    R = np.array([[np.cos(theta), -np.sin(theta), 0.0],[np.sin(theta),  np.cos(theta), 0.0],[0.0, 0.0, 1.0]])
    if lidar_for_left_camera.shape[0] > 0:
        lidar_for_left_camera = R.dot(lidar_for_left_camera.T).T
        z = lidar_for_left_camera[..., 1]
        x = ((focal_x * lidar_for_left_camera[..., 0]) / z) + (img_width  / 2.0)
        y = ((focal_y * lidar_for_left_camera[..., 2]) / z) + (img_height / 2.0)
        result_left = np.stack([x, y, lidar_indices], 1)
        result_left = result_left[np.logical_and(result_left[...,0] > 0, result_left[...,0] < img_width)]
        result_left = result_left[np.logical_and(result_left[...,1] > 0, result_left[...,1] < img_height)]
        result_left_shifted        = result_left[result_left[...,0] >= (img_width/2.0)]
        if result_left_shifted.shape[0] > 0:
            result_left_shifted[...,0] = result_left_shifted[...,0] - (img_width/2.0)
    else:
        result_left_shifted = np.zeros((0, 3))

    theta = np.radians(right_camera_rotation)
    R = np.array([[np.cos(theta), -np.sin(theta), 0.0],[np.sin(theta),  np.cos(theta), 0.0],[0.0, 0.0, 1.0]])
    if lidar_for_right_camera.shape[0] > 0:
        lidar_for_right_camera = R.dot(lidar_for_right_camera.T).T
        z = lidar_for_right_camera[..., 1]
        x = ((focal_x * lidar_for_right_camera[..., 0]) / z) + (img_width / 2.0)
        y = ((focal_y * lidar_for_right_camera[..., 2]) / z) + (img_height / 2.0)
        result_right = np.stack([x, y, lidar_indices], 1)
        result_right = result_right[np.logical_and(result_right[..., 0] > 0, result_right[..., 0] < img_width)]
        result_right = result_right[np.logical_and(result_right[..., 1] > 0, result_right[..., 1] < img_height)]
        result_right_shifted = result_right[result_right[...,0] < (img_width/2.0)] 
        if result_right_shifted.shape[0] > 0:
            result_right_shifted[...,0] = result_right_shifted[...,0] + (img_width/2.0) + img_width
    else:
        result_right_shifted = np.zeros((0, 3))

    results_total = np.concatenate((result_left_shifted, result_center_shifted, result_right_shifted), axis=0)

    valid_bev_points = []
    valid_cam_points = []
    for i in range(results_total.shape[0]):
        lidar_index = int(results_total[i, 2])
        bev_x = int((lidar[lidar_index][0] + lidar_meters_x) * pixels_per_meter)
        bev_y = (int(lidar[lidar_index][1] * pixels_per_meter) - (lidar_height-1)) * -1
        valid_bev_points.append([bev_x, bev_y])
        img_x = int(results_total[i][0])
        img_y = (int(results_total[i][1]) - (img_height - 1)) * -1
        valid_cam_points.append([img_x, img_y])

    valid_bev_points = np.array(valid_bev_points)
    valid_cam_points = np.array(valid_cam_points)
    if valid_bev_points.shape[0] == 0:
        return np.zeros((256//downscale_factor, 256//downscale_factor, 5, 2)), np.zeros((160//downscale_factor, 704//downscale_factor, 5, 2))

    bev_points, cam_points = correspondences_at_one_scale(valid_bev_points, valid_cam_points,  (lidar_width // downscale_factor),
                                                          (lidar_height // downscale_factor), (img_width // downscale_factor) * 2,
                                                          (img_height // downscale_factor), downscale_factor)
    return bev_points, cam_points

def decode_pil_to_npy(img):
    (channels, width, height) = (15, img.shape[1], img.shape[2])
    bev_array = np.zeros([channels, width, height])
    for ix in range(5):
        bit_pos = 8-ix-1
        bev_array[[ix, ix+5, ix+5+5]] = (img & (1<<bit_pos)) >> bit_pos
    return bev_array[10:12]