#点云处理类的实现

import numpy as np
import cv2
import os
import json # <-- 导入 json 库
from types import SimpleNamespace # 用于创建一个简单的对象来存储内参
import open3d as o3d

class OfflinePointCloudProcessor:
    
    def __init__(self, T_world_cam, intrinsics_json_path, depth_scale):
        """
        构造函数 - 适用于离线数据，从 JSON 文件加载内参。

        参数:
        T_world_cam (np.array, 4x4): 
            从相机坐标系到世界坐标系的变换矩阵。
        intrinsics_json_path (str): 
            包含相机内参的 .json 文件路径。
        depth_scale (float): 
            深度缩放因子。(例如 0.001)
            (注意: 这个值通常不在内参json中，所以我们单独传入)
        """
        print("Offline PointCloudProcessor 初始化中...")
        
        # --- 1. 从 JSON 加载相机内参 ---
        try:
            with open(intrinsics_json_path, 'r') as f:
                intr_data = json.load(f)
            
            # 从 json 数据中提取参数
            H = intr_data['height']
            W = intr_data['width']
            fx = intr_data['fx']
            fy = intr_data['fy']
            ppx = intr_data['ppx']
            ppy = intr_data['ppy']
            
            print(f"成功从 {intrinsics_json_path} 加载内参。")

        except FileNotFoundError:
            print(f"错误: 内参文件未找到! {intrinsics_json_path}")
            raise # 抛出异常，因为没有内参无法继续
        except KeyError as e:
            print(f"错误: 内参文件 {intrinsics_json_path} 中缺少键: {e}")
            raise
        except Exception as e:
            print(f"加载内参时出错: {e}")
            raise
            
        # 我们使用 SimpleNamespace 来模拟原始 `intrinsics` 对象的结构
        self.depth_intrinsics = SimpleNamespace(
            height=H, width=W,
            fx=fx, fy=fy,
            ppx=ppx, ppy=ppy
        )
        # "width": color_intrinsics.width,
        # "height": color_intrinsics.height,
        # "fx": color_intrinsics.fx,
        # "fy": color_intrinsics.fy,
        # "ppx": color_intrinsics.ppx,
        # "ppy": color_intrinsics.ppy,
        # "model": str(color_intrinsics.model),
        # "coeffs": color_intrinsics.coeffs,
        # "K": [
        #     [color_intrinsics.fx, 0, color_intrinsics.ppx],
        #     [0, color_intrinsics.fy, color_intrinsics.ppy],
        #     [0, 0, 1]
        # ]
        self.depth_scale = depth_scale
        
        # 2. 存储变换矩阵
        self.T_world_cam = T_world_cam 
        
        # 3. 预计算像素网格 (优化)
        self.uu, self.vv = np.meshgrid(np.arange(W), np.arange(H))
        
        # 4. 空返回 (用于出错时)
        self.empty_return = (
            np.array([]), 
            np.array([]), 
            np.zeros((H, W, 3), dtype=np.uint8), 
            np.zeros((H, W), dtype=bool)
        )
        
        print(f"处理器已准备就绪，期望分辨率: {H}x{W}。")

    
    def apply_segmentation_mask(self, depth_path, mask_path, output_path, target_id=None):
        """
        将语义/实例分割掩码应用到深度图上。

        参数:
        depth_path (str): 深度数据 .npy 文件的路径。
        mask_path (str): 掩码数据 .npy 文件的路径 (背景为0，物体为 1, 2, 3...)。
        output_path (str): 保存结果 .npy 文件的路径。
        target_id (int 或 None): 
            - 如果为 None (默认): 应用所有非零的掩码区域 (即所有物体)。
            - 如果为 int (例如 1): 仅应用掩码中等于该 ID 的区域。
        """
        
        print(f"--- 开始处理 ---")
        try:
            # 1. 加载数据
            depth_data = np.load(depth_path)
            mask_data = np.load(mask_path)
            
            print(f"成功加载: {depth_path} (形状: {depth_data.shape}, 类型: {depth_data.dtype})")
            print(f"成功加载: {mask_path} (形状: {mask_data.shape}, 类型: {mask_data.dtype})")
            
            # 可以在这里查看掩码中有哪些唯一的ID
            unique_ids = np.unique(mask_data)
            print(f"掩码中找到的唯一ID: {unique_ids}")

        except FileNotFoundError as e:
            print(f"错误: 文件未找到 - {e}")
            return
        except Exception as e:
            print(f"加载文件时出错: {e}")
            return

        # 2. 检查形状是否匹配
        if depth_data.shape != mask_data.shape:
            print(f"错误: 深度图和掩码的形状不匹配! {depth_data.shape} vs {mask_data.shape}")
            # 尝试处理常见的 (H, W, 1) vs (H, W) 的情况
            if depth_data.shape[:2] == mask_data.shape[:2] and depth_data.ndim == 3:
                print("检测到深度图可能有多余的通道维度，尝试将其压缩...")
                try:
                    depth_data = np.squeeze(depth_data, axis=-1)
                    print(f"压缩后深度图形状: {depth_data.shape}")
                    if depth_data.shape != mask_data.shape:
                        raise ValueError("压缩后形状仍不匹配")
                except ValueError as ve:
                    print(f"自动压缩失败: {ve}。请手动检查你的 .npy 文件。")
                    return
            else:
                print("形状不兼容，停止处理。")
                return


        # 3. 创建布尔掩码 (Boolean Mask)
        if target_id is None:
            # --- 这是你当前的需求 ---
            # "暂时将所有编号的都应用到深度上"
            # 只要掩码值不是 0 (背景)，就为 True
            print("模式: 应用所有非零 (非背景) 的掩码区域...")
            boolean_mask = (mask_data != 0)
        else:
            # --- 这是你未来可能的需求 ---
            # "只看 ID 为 1 的物体"
            # 只有掩码值等于 target_id 时才为 True
            print(f"模式: 仅应用 ID 为 {target_id} 的掩码区域...")
            if target_id not in unique_ids:
                print(f"警告: ID {target_id} 不在掩码的唯一ID列表中 {unique_ids}。输出文件可能全为0。")
            boolean_mask = (mask_data == target_id)

        # 4. 应用掩码
        # NumPy 在进行乘法操作时，会把 True 当作 1，False 当作 0
        # 所以 `depth_data * boolean_mask` 会实现：
        # - 掩码为 True 的地方: depth_value * 1 = depth_value (保留)
        # - 掩码为 False 的地方: depth_value * 0 = 0         (清除)
        
        # (或者, 你也可以用 np.where: masked_depth = np.where(boolean_mask, depth_data, 0))
        
        masked_depth = depth_data * boolean_mask
        
        # 确保输出的数据类型与输入深度图一致
        masked_depth = masked_depth.astype(depth_data.dtype)

        # 5. 保存结果
        try:
            np.save(output_path, masked_depth)
            print(f"成功! 已将掩码后的深度图保存到: {output_path}")
        except Exception as e:
            print(f"保存文件时出错: {e}")
            
        print(f"--- 处理完成 ---")


    def process_frame_from_files(self, depth_npy_path, color_jpg_path):
        """
        从 .npy 深度文件和 .jpg 颜色文件加载帧并处理。
        
        *** 此函数与上一版本完全相同，无需修改 ***

        返回:
        - points_world_np (N, 3): 世界坐标系中的 3D 点
        - colors_rgb_flat (N, 3): 每个点的颜色 (0-1)
        - frame_rgb (H, W, 3): 原始 RGB 图像 (0-255)
        - valid (H, W): 有效深度像素的布尔掩码
        """
        try:
            # --- 1. 加载数据 ---
            depth_image = np.load(depth_npy_path)       # (H, W)
            color_image = cv2.imread(color_jpg_path)    # (H, W, 3) BGR
            
            if color_image is None:
                raise FileNotFoundError(f"无法加载 RGB 图像: {color_jpg_path}")
                
        except FileNotFoundError as e:
            print(f"错误: 文件未找到 - {e}")
            return self.empty_return
        except Exception as e:
            print(f"加载文件时出错: {e}")
            return self.empty_return
        
        # --- 2. 验证尺寸 ---
        expected_shape = (self.depth_intrinsics.height, self.depth_intrinsics.width)
        
        if depth_image.shape != expected_shape:
            print(f"错误: 深度图 {depth_npy_path} 形状 {depth_image.shape} 与内参 {expected_shape} 不符。")
            return self.empty_return
            
        if color_image.shape[:2] != expected_shape:
            print(f"错误: RGB图 {color_jpg_path} 形状 {color_image.shape[:2]} 与内参 {expected_shape} 不符。")
            return self.empty_return

        # --- 3. 反投影 (2D -> 3D 相机坐标系) ---
        z_cam = depth_image * self.depth_scale # (H, W)
        x_cam = (self.uu - self.depth_intrinsics.ppx) * z_cam / self.depth_intrinsics.fx
        y_cam = (self.vv - self.depth_intrinsics.ppy) * z_cam / self.depth_intrinsics.fy
        points_cam_np = np.stack([x_cam, y_cam, z_cam], axis=-1)
        
        # --- 4. 过滤和扁平化 ---
        valid = (z_cam > 0) & (z_cam < 2.0) # (H, W) bool
        
        points_cam_flat = points_cam_np[valid]  # (N, 3)
        colors_bgr_flat = color_image[valid]    # (N, 3)
        
        frame_rgb = color_image[..., ::-1].copy() # BGR to RGB
        colors_rgb_flat = colors_bgr_flat[..., ::-1] / 255.0
        
        # --- 5. 坐标变换 (相机 -> 世界) ---
        points_cam_homo = np.hstack((points_cam_flat, np.ones((points_cam_flat.shape[0], 1))))
        points_world_homo = (self.T_world_cam @ points_cam_homo.T).T
        points_world_np = points_world_homo[:, :3]
        
        return points_world_np, colors_rgb_flat, frame_rgb, valid
    
if __name__  == "__main__":
    json_path = "/home/shui/sam2_pointclouds/processed/test1_processed/intrinsics.json"
    try: 
        with open(json_path, 'r', encoding='utf-8') as f:
        # 使用 json.load() 从文件对象中加载数据
            data = json.load(f)

        print("加载json成功")

    except FileNotFoundError:
        print("找不到文件")
    except json.JSONDecodeError:
        print("json格式无效")
    except Exception as e:
        print("发生未知错误")
    
    DEPTH_SCALE = 0.001
    R_world_cam = np.array([
    [ 0.03139845, -0.94181956,  0.33464915],
    [-0.99726785, -0.00712148,  0.07352629],
    [-0.0668653,  -0.33604345, -0.93946998]
    ])
    t_world_cam = np.array([-0.68908522, -0.07112856, 0.54733009]) 
    T_world_cam = np.eye(4)
    T_world_cam[:3, :3] = R_world_cam
    T_world_cam[:3, 3] = t_world_cam

    try:
        processor = OfflinePointCloudProcessor(
                T_world_cam, 
                json_path, 
                DEPTH_SCALE
            )
    except Exception as e:
        print("初始化失败")

    # vis = o3d.visualization.Visualizer()
    # vis.create_window()
    # pcd = o3d.geometry.PointCloud()
    # is_first_frame = True

    points, colors, rgb_image, valid_mask = processor.process_frame_from_files(
            "masked_depth_all_objects.npy", "image_00020.jpg"
        )
    # pcd.points = o3d.utility.Vector3dVector(points)
    # pcd.colors = o3d.utility.Vector3dVector(colors)
    # if is_first_frame:
    #         vis.add_geometry(pcd)
    #         vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5))
    #         is_first_frame = False
    # else:
    #     vis.update_geometry(pcd)
    
    # vis.poll_events()
    # vis.update_renderer()
    # while True:
    #     cv2.imshow("RGB Image", rgb_image)

    # e. 设置 Open3D
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # f. (新!) 将几何体添加到查看器
    vis.add_geometry(pcd)
    # 添加一个世界坐标系原点，以便观察
    vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5))

    print("\n--- 启动显示 ---")
    print("点云已加载。请查看弹出的 Open3D 和 CV2 窗口。")
    print("在 CV2 (图像) 窗口中按 'q' 键退出。")

    # g. (新!) 启动持续显示循环
    try:
        while True:
            # 1. 刷新 Open3D 窗口
            vis.poll_events()
            vis.update_renderer()
            
            # 2. 显示 2D 图像和掩码
            cv2.imshow("RGB Image", rgb_image)
            cv2.imshow("Valid Mask", valid_mask.astype(np.uint8) * 255)

            # 3. 检查退出 (每 10 毫秒检查一次)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    finally:
        # h. 清理
        vis.destroy_window()
        cv2.destroyAllWindows()
        print("清理完成，退出。")