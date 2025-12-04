import sys
import hydra
import time
from omegaconf import OmegaConf
from hydra.core.global_hydra import GlobalHydra
import pathlib
from diffusion_policy_3d.workspace.base_workspace import BaseWorkspace
import torch
import os 
import numpy as np
from termcolor import cprint
import threading
import IPython
import pyrealsense2 as rs 
import open3d as o3d
from process_data_offline_for_all import IncrementalObjectTracker
from depth_workflow_for_all import resample_point_cloud
from scipy.spatial.transform import Rotation as R
from realman65.my_robot.realman_65_interface_dual import Realman65Interface
from processor.KinHelper_Dual import KinHelper
import copy
import IPython

os.environ['WANDB_SILENT'] = "True"
OmegaConf.register_new_resolver("eval", eval, replace=True)
e = IPython.embed

def matrix_to_rotation_6d_numpy(matrix: np.ndarray) -> np.ndarray:
    return np.concatenate([matrix[:, 0], matrix[:, 1]], axis=0)

def rotation_6d_to_matrix_numpy(rot_6d: np.ndarray) -> np.ndarray:
    if rot_6d.shape != (6,):
        raise ValueError(f"输入 shape 必须是 (6,)，但得到的是 {rot_6d.shape}")
    a1 = rot_6d[0:3]
    a2 = rot_6d[3:6]
    b1 = a1 / (np.linalg.norm(a1) + 1e-6)
    dot_b1_a2 = np.dot(b1, a2)
    v2 = a2 - dot_b1_a2 * b1
    b2 = v2 / (np.linalg.norm(v2) + 1e-6)
    b3 = np.cross(b1, b2)
    matrix = np.stack([b1, b2, b3], axis=1)
    return matrix


class RealSenseCamera:
    def __init__(self, width=640, height=480, fps=30):
        self.width = width
        self.height = height
        self.fps = fps

        self.pipeline = None
        self.profile = None
        self.depth_scale = None
        self.color_intrinsics = None
        self.align = None
        self.T_world_cam = None
        self.uu = None
        self.vv = None

        self.rgb_frame = None
        self.depth_frame = None
        self.filtered_depth = None

        self.frame_lock = threading.Lock()
        self.running = False
        self.thread = None

        try:
            self._init_camera(self.width, self.height, self.fps)
            cprint(f"RealSense camera initialized at {width}x{height}, {fps} FPS.", "green")
        except Exception as e:
            cprint(f"Failed to initialize RealSense camera: {e}", "red")
            raise

    def _init_camera(self, width, height, fps):
            # 1. 创建 Pipeline 和 Config
            self.pipeline = rs.pipeline()
            config = rs.config()
            config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
            config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)

            # ================= [修改开始] =================
            # 2. 在启动流之前，先通过 Context 获取设备并设置 Preset
            # 这样可以避免 "Device or resource busy" 错误
            ctx = rs.context()
            if len(ctx.devices) > 0:
                # 默认获取第一个设备 (如果你有多个相机，这里需要用 serial number 筛选)
                dev = ctx.devices[0] 
                depth_sensor = dev.first_depth_sensor()
                
                # 设置 Visual Preset (High Density)
                if depth_sensor.supports(rs.option.visual_preset):
                    try:
                        # 4 = High Density
                        depth_sensor.set_option(rs.option.visual_preset, 4) 
                        cprint("Visual Preset set to High Density (4)", "green")
                    except Exception as e:
                        cprint(f"Warning: Failed to set Visual Preset: {e}", "yellow")
            else:
                cprint("No RealSense device connected!", "red")
                raise RuntimeError("No RealSense device found")
            # ================= [修改结束] =================

            # 3. 启动 Pipeline (现在设备已经配置好了)
            try:
                self.profile = self.pipeline.start(config)
            except Exception as e:
                cprint(f"RealSense Pipeline Start Error: {e}", "red")
                raise

            # 4. 获取一些必要的参数 (Intrinsics, Scale)
            depth_sensor = self.profile.get_device().first_depth_sensor()
            self.depth_scale = depth_sensor.get_depth_scale()

            color_profile = self.profile.get_stream(rs.stream.color)
            self.color_intrinsics = color_profile.as_video_stream_profile().get_intrinsics()

            # 5. 初始化后处理过滤器
            # (A) 空间过滤器
            self.spatial = rs.spatial_filter()
            self.spatial.set_option(rs.option.filter_magnitude, 2)
            self.spatial.set_option(rs.option.filter_smooth_alpha, 0.5)
            self.spatial.set_option(rs.option.filter_smooth_delta, 20)
            self.spatial.set_option(rs.option.holes_fill, 3) 
            
            # (B) 时间过滤器
            self.temporal = rs.temporal_filter()
            self.temporal.set_option(rs.option.filter_smooth_alpha, 0.4)
            self.temporal.set_option(rs.option.filter_smooth_delta, 20)
            
            # (C) 空洞填充
            self.hole_filling = rs.hole_filling_filter(1)

            align_to = rs.stream.color
            self.align = rs.align(align_to)

            # 外参设置 (保持你原有的逻辑)
            R_world_cam = np.array([[-0.04975599, -0.86800262,  0.49406051],
                                    [-0.99875799,  0.0445353,  -0.02234025],
                                    [-0.00261174, -0.49455845, -0.86914045]])
            t_world_cam = np.array([-0.79518668,0.2869315,0.7312764])
            T_world_cam = np.eye(4)
            T_world_cam[:3, :3] = R_world_cam
            T_world_cam[:3, 3] = t_world_cam
            self.T_world_cam = T_world_cam

            self.uu, self.vv = np.meshgrid(np.arange(self.color_intrinsics.width),
                                            np.arange(self.color_intrinsics.height),indexing='xy')
            self.init_status = True

    def _receive_image_thread(self):
        time.sleep(1.0)
        cprint("Image receiving thread started.", "cyan")
        
        while self.running:
            try:
                frames = self.pipeline.wait_for_frames(5000) 
                aligned_frames = self.align.process(frames)
                
                with self.frame_lock:
                    self.rgb_frame = aligned_frames.get_color_frame()
                    self.depth_frame = aligned_frames.get_depth_frame()
                    filtered_depth = self.depth_frame
            
                    # 应用空间滤波 (平滑)
                    filtered_depth = self.spatial.process(filtered_depth)
                    
                    # 应用时间滤波 (稳定)
                    # 只有当深度图分辨率没有改变时，时间滤波才有效
                    # filtered_depth = self.temporal.process(filtered_depth)
                    
                    # 应用空洞填充 (修补)
                    self.depth_frame = self.hole_filling.process(filtered_depth)

                # cprint("frames receivede","red")

            except RuntimeError as e:
                if "Frame didn't arrive" in str(e):
                    cprint("Warning: RealSense frame timeout. Re-trying...", "yellow")
                    time.sleep(0.1)
                    continue
                else:
                    cprint(f"Unknown RealSense error: {e}", "red")
                    self.running = False  # 发生未知错误时停止线程
                    break
            except Exception as e:
                cprint(f"Error in image thread: {e}", "red")
                self.running = False
                break

            time.sleep(0.01)
        
        cprint("Image receiving thread stopped.", "cyan")

    def start(self):
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self._receive_image_thread, daemon=True)
            self.thread.start()
            cprint("Camera thread started.", "green")
        else:
            cprint("Camera thread is already running.", "yellow")

    def stop(self):
        if self.running:
            self.running = False
            if self.thread is not None:
                self.thread.join()  # 等待线程完全退出
            
            if self.pipeline:
                self.pipeline.stop()
            cprint("Camera thread and pipeline stopped.", "green")
        else:
            cprint("Camera is not running.", "yellow")

    def get_frames(self):
        with self.frame_lock:
            rgb = self.rgb_frame
            depth = self.depth_frame
        
        if rgb and depth:
            rgb_image = copy.copy(np.asanyarray(rgb.get_data()))
            depth_image = copy.copy(np.asanyarray(depth.get_data()))
            return True, rgb_image, depth_image
        else:
            return False, None, None

    def get_intrinsics(self):
        return self.color_intrinsics

    def get_depth_scale(self):
        return self.depth_scale

    def get_transform(self):
        return self.T_world_cam

    def __enter__(self):
        self.start()
        cprint("Waiting for first camera frame...", "cyan")
        while True:
            success, _, _ = self.get_frames()
            if success:
                cprint("First frame received. Camera is ready.", "green")
                break
            if not self.running:
                cprint("Camera failed to start during __enter__.", "red")
                raise RuntimeError("Camera failed to start")
            time.sleep(0.1)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

class ActionAgent:
    def __init__(self, gripper_threshold=0.5):
        self.rm_interface = Realman65Interface(auto_setup=False)

        try:
            self.rm_interface.set_up()
        except Exception as exc:
            cprint(f"Failed to set up RM65 interface: {exc}", "red")
        try:
            self.rm_interface.reset()
            cprint("RM65 robot reset completed.", "green")
            # 启动双臂连续控制线程
            self.rm_interface.start_control()
        except Exception as exc:
            cprint(f"Failed to reset RM65 robot: {exc}", "yellow")

        self._gripper_threshold = gripper_threshold
        self._last_gripper_state = None

        # 双臂关节状态缓存：[LJ1..LJ6, LGrip, RJ1..RJ6, RGrip]
        self.joint_qpos = np.zeros(14, dtype=np.float32)
        self.pose = np.zeros(10, dtype=np.float32)
        
        # --- 1. 夹爪线程控制变量初始化 ---
        # Target: 策略网络希望夹爪处于的状态 (0或1, None表示未收到指令)
        self._target_left_grip = None
        self._target_right_grip = None
        
        # Actual: 上一次成功发送给硬件的状态
        self._actual_left_grip = None
        self._actual_right_grip = None

        self._run_state_thread = True
        self._state_thread = threading.Thread(target=self._receive_state_thread, daemon=True)
        self._state_thread.start()
        
        # --- 3. 启动夹爪控制专用线程 (关键修改) ---
        self._gripper_thread = threading.Thread(target=self._gripper_control_thread, daemon=True)
        self._gripper_thread.start()
        
        
        

    def execute(self, action, mode="angle"):
        if mode == "jointangle":
            return self._execute_action_angle(action)
        elif mode == "pose":
            return self._execute_action_pose(action)
        elif mode == "pose10d":
            return self._execute_action_pose_10d(action)
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def _execute_action_angle(self, action):
        # 6+1+6+1=14 6个关节+1个夹爪+6个关节+1个夹爪=14
        action_np = np.asarray(action, dtype=np.float64).flatten()
        if action_np.size < 14:
            raise ValueError("Action dim must be >= 14 for angle control.")

        # 约定 14D: [LJ1..LJ6, LGrip, RJ1..RJ6, RGrip]
        left_rad = None
        left_grip = None
        right_rad = None
        right_grip = None

        if action_np.size >= 14:
            left_rad = action_np[:6]
            left_grip = action_np[6]
            right_rad = action_np[7:13]
            right_grip = action_np[13]
        
        left_rad[5] += np.radians(180)
        try:
            # 下发双臂关节角（弧度）
            if left_rad is not None:
                self.rm_interface.target_joint_angles_left = left_rad
            if right_rad is not None:
                self.rm_interface.target_joint_angles_right = right_rad
            if not self.rm_interface.init_ik:
                self.rm_interface.init_ik = True
        except Exception as exc:
            cprint(f"[Angle] Failed to apply joint command: {exc}", "red")

        # 2. 夹爪控制 (Gripper Control) - 只更新目标变量，不调用API
        # 这里的操作是微秒级的，绝对不会阻塞
        if left_grip is not None:
            self._target_left_grip = 1 if left_grip >= self._gripper_threshold else 0
            
        if right_grip is not None:
            self._target_right_grip = 1 if right_grip >= self._gripper_threshold else 0
    
    def _gripper_control_thread(self):
        """
        后台线程：专门负责处理耗时的夹爪 IO 操作。
        只有当目标状态发生改变时，才调用硬件接口。
        """
        cprint("Gripper control thread started.", "cyan")
        while self._run_state_thread:
            # --- 左臂处理 ---
            # 如果有目标指令，且目标指令与上次发送的不一致
            if self._target_left_grip is not None and self._target_left_grip != self._actual_left_grip:
                target = self._target_left_grip
                try:
                    # 这里的阻塞只会卡住这个后台线程，不会卡主线程
                    # cprint(f"Setting Left Gripper to {target}...", "cyan")
                    self.rm_interface.set_gripper("left_arm", target)
                    
                    # 只有发送成功才更新状态，避免失败后不再重试
                    self._actual_left_grip = target 
                except Exception as exc:
                    cprint(f"Failed to set left gripper: {exc}", "yellow")
                    time.sleep(0.1) # 失败了歇一会再试

            # --- 右臂处理 ---
            if self._target_right_grip is not None and self._target_right_grip != self._actual_right_grip:
                target = self._target_right_grip
                try:
                    # cprint(f"Setting Right Gripper to {target}...", "cyan")
                    self.rm_interface.set_gripper("right_arm", target)
                    self._actual_right_grip = target
                except Exception as exc:
                    cprint(f"Failed to set right gripper: {exc}", "yellow")
                    time.sleep(0.1)

            # 防止 CPU 空转，给予 10ms 休眠
            # 这个频率对于夹爪来说足够了
            time.sleep(0.01) 
    
    
    def _execute_action_pose(self, action):
        action_np = np.asarray(action, dtype=np.float64).flatten()
        if action_np.size < 10:
            raise ValueError("Pose control requires 10D action.")

        try:
            target_pos = action_np[:3]
            rot6d = action_np[3:9]
            rot_mat = rotation_6d_to_matrix_numpy(rot6d)
            rot = R.from_matrix(rot_mat)
            quat = rot.as_quat(scalar_first=False)

            target_rad_angle = self.rm_interface.ik_solver.move_to_pose_and_get_joints(
                target_pos, quat
            )
            if target_rad_angle is None:
                return

            self.rm_interface.target_joint_angles = target_rad_angle
            if not self.rm_interface.init_ik:
                self.rm_interface.init_ik = True

        except Exception as exc:
            cprint(f"[Pose] Failed to send pose command: {exc}", "red")

        self._apply_gripper(action_np[9])

    def _execute_action_pose_10d(self, action):
        return self._execute_action_pose(action)

    def _apply_gripper(self, gripper_val):
        if np.isnan(gripper_val):
            return
        gripper_cmd = 1 if gripper_val >= self._gripper_threshold else 0

        if self._last_gripper_state is None or gripper_cmd != self._last_gripper_state:
            try:
                self.rm_interface.set_gripper("left_arm", gripper_cmd)
                self._last_gripper_state = gripper_cmd
            except Exception as exc:
                cprint(f"Failed to apply gripper command: {exc}", "yellow")

    def _receive_state_thread(self):
        while self._run_state_thread:
            try:
                joint_dict = self.rm_interface.get_joint_angles()
                eepose_dict = self.rm_interface.get_end_effector_pose()
                gripper_dict = self.rm_interface.get_gripper_state()
                gripper_state_left = None
                gripper_state_right = None

                # ---- Joint angles ----
                if joint_dict:
                    left_arm_angles = joint_dict.get("left_arm")
                    if left_arm_angles is not None:
                        # self.joint_qpos[:6] = np.radians(left_arm_angles).astype(np.float32)
                        self.joint_qpos[:6] = np.radians(left_arm_angles).astype(np.float32)
                        # self.joint_qpos[5] -= np.radians(180)
                    right_arm_angles = joint_dict.get("right_arm")
                    if right_arm_angles is not None:
                        self.joint_qpos[7:13] = np.radians(right_arm_angles).astype(np.float32)

                # ---- Gripper ----
                if gripper_dict:
                    gripper_state_left = gripper_dict.get("left_arm")
                    if gripper_state_left is not None:
                        self.joint_qpos[6] = gripper_state_left
                    gripper_state_right = gripper_dict.get("right_arm")
                    if gripper_state_right is not None:
                        self.joint_qpos[13] = gripper_state_right

                # ---- Pose ----
                if eepose_dict:
                    eepose = eepose_dict.get("left_arm")
                    if eepose is not None:
                        xyz = np.array(eepose[0:3], dtype=np.float32)
                        rpy = np.array(eepose[3:6], dtype=np.float32)

                        rot = R.from_euler('xyz', rpy, degrees=False)
                        rotation_6d = matrix_to_rotation_6d_numpy(rot.as_matrix())

                        self.pose[0:3] = xyz
                        self.pose[3:9] = rotation_6d

                if gripper_state_left is not None:
                    self.pose[9] = float(gripper_state_left)

            except Exception as exc:
                cprint(f"Failed to read sensor data: {exc}", "yellow")

            time.sleep(0.01)   # 10ms

    # 停止线程接口
    def stop(self):
        self._run_state_thread = False
        self._state_thread.join()



import open3d as o3d
import numpy as np

def visualize_single_frame(obs_pcd_list, control_pcd_list, frame_idx=0):
    """
    可视化指定那一帧的数据
    """
    print(f"Visualizing frame {frame_idx}")
    
    # 1. 处理带颜色的点云 (obs_pcd_list)
    # 你的数据格式应该是 (N, 6) -> [x, y, z, r, g, b]
    raw_target = obs_pcd_list[frame_idx]
    pcd_target = o3d.geometry.PointCloud()
    pcd_target.points = o3d.utility.Vector3dVector(raw_target[:, :3])
    pcd_target.colors = o3d.utility.Vector3dVector(raw_target[:, 3:6])
    
    # 为了方便看，稍微平移一下第二个点云，防止重叠
    pcd_target.translate([0, 0, 0]) 

    # 2. 处理纯几何点云 (control_pcd_list)
    # 你的数据格式应该是 (N, 3) -> [x, y, z]
    raw_control = control_pcd_list[frame_idx]
    pcd_control = o3d.geometry.PointCloud()
    pcd_control.points = o3d.utility.Vector3dVector(raw_control[:, :3])
    
    # 因为 control_pcd_list 没有颜色，Open3D 默认会显示黑色或灰色。
    # 这里我们手动给它涂成红色，方便和上面的蓝色夹爪/彩色物体区分
    pcd_control.paint_uniform_color([1, 0, 0]) 
    
    # 将其平移到侧面，方便同时观察对比
    pcd_control.translate([0, 0, 0]) # 向 X 轴平移 0.5 米

    # 3. 创建坐标轴辅助观察 (红色X, 绿色Y, 蓝色Z)
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0])

    print("左边: Target (彩色) + 蓝夹爪")
    print("右边: Obs (染红示意) + 蓝夹爪几何")
    
    # 4. 启动可视化窗口
    o3d.visualization.draw_geometries([pcd_target, pcd_control, coord_frame], 
                                      window_name="Point Cloud Check",
                                      width=1200, height=800)

# === 使用方法 ===
# 假设你已经运行了上面的代码得到了 list
# visualize_single_frame(obs_pcd_list, control_pcd_list, frame_idx=0)


class MaskPointCloudExtractor:
    """
    将 tracker 初始化、mask 生成、点云提取统一封装成一个类。
    """
    def __init__(self, camera: RealSenseCamera, kin_helper):
        self.camera = camera
        self.kin_helper = kin_helper
        self.tracker = None
        self.CONTROL_OBJECT_NAMES = ["blue box"]
        self.TARGET_OBJECT_NAMES = ["white cup", "yellow cup"]

    def init_tracker(self, api_token, prompt_text, detection_interval=1000):
        self.tracker = IncrementalObjectTracker(
            api_token=api_token,
            grounding_model_name="GroundingDino-1.5-Pro",
            sam2_model_cfg="configs/sam2.1/sam2.1_hiera_t.yaml",
            sam2_ckpt_path="/home/shui/idp3_test/Improved-3D-Diffusion-Policy/Improved-3D-Diffusion-Policy/checkpoints/sam2.1_hiera_tiny.pt",
            device="cuda",
            prompt_text=prompt_text,
            detection_interval=detection_interval,
        )
        self.tracker.set_prompt(prompt_text)
        print("[Tracker] 初始化完成。")

    def gen_mask(self, rgb):
        if self.tracker is None:
            raise RuntimeError("Tracker 尚未初始化，请先调用 init_tracker().")

        process_img, mask_array,json_metadata = self.tracker.add_image(rgb)
        if process_img is None or not isinstance(process_img, np.ndarray):
            cprint("[Warning] Tracker返回空 mask。","yellow")
            return np.zeros((self.camera.height, self.camera.width), dtype=np.uint8)
        return mask_array,json_metadata


    def extract_pcs_from_frame(self, rgb_array, depth_array, qpos_array):
            if self.tracker is None:
                raise RuntimeError("Tracker 尚未初始化，请先调用 init_tracker().")

            obs_pcd_list = [] 
            control_pcd_list = []    
            
            size = len(rgb_array)
            if size != len(depth_array):
                print(f"错误: RGB({size}) 和 Depth({len(depth_array)}) 长度不一致。")
                return [], []

            # === 内部辅助函数：提取点云 ===
            # 增加 use_texture 参数：True则使用图片颜色，False则填充黑色(0,0,0)
            def get_world_points_from_mask(mask_ids, current_mask_array, current_depth, current_rgb, use_texture=True):
                if not mask_ids:
                    return np.empty((0, 6)) 
                
                bool_mask = np.isin(current_mask_array, mask_ids)
                
                # 深度处理
                masked_depth = (current_depth * bool_mask).astype(current_depth.dtype)
                
                # 投影到 3D
                z_cam = masked_depth * self.camera.get_depth_scale()
                intr = self.camera.get_intrinsics()
                x_cam = (self.camera.uu - intr.ppx) * z_cam / intr.fx
                y_cam = (self.camera.vv - intr.ppy) * z_cam / intr.fy
                points_cam = np.stack([x_cam, y_cam, z_cam], axis=-1)
                
                # 过滤有效点
                valid = (z_cam > 0) & (z_cam < 2.0)
                points_cam_flat = points_cam[valid]
                
                if points_cam_flat.shape[0] == 0:
                    return np.empty((0, 6))

                # 颜色处理逻辑
                if use_texture:
                    # 使用原图纹理 (BGR -> RGB)
                    colors_bgr_flat = current_rgb[valid]
                    colors_rgb_flat = colors_bgr_flat[..., ::-1] / 255.0
                else:
                    # 不使用纹理，填充全黑 (0, 0, 0)
                    # 如果想填充白色改成 np.ones_like...
                    colors_rgb_flat = np.zeros((points_cam_flat.shape[0], 3), dtype=np.float32)

                # 转世界坐标
                points_cam_homo = np.concatenate(
                    [points_cam_flat, np.ones((points_cam_flat.shape[0], 1))], axis=1
                )
                world_homo = (self.camera.get_transform() @ points_cam_homo.T).T
                world_xyz = world_homo[:, :3]
                
                return np.hstack([world_xyz, colors_rgb_flat])

            # === 主循环 ===
            for rgb_np, depth_np, qpos in zip(rgb_array, depth_array, qpos_array):
                mask_array, json_metadata = self.gen_mask(rgb_np)
                
                obs_object_ids = [] 
                target_object_ids = []
                
                if json_metadata:
                    try:
                        labels = json_metadata.get("labels", {})
                        for key, info in labels.items():
                            cls_name = info.get("class_name")
                            instance_id = int(info.get("instance_id"))
                            
                            if cls_name in self.CONTROL_OBJECT_NAMES:
                                obs_object_ids.append(instance_id)
                            if cls_name in self.TARGET_OBJECT_NAMES:
                                target_object_ids.append(instance_id)
                    except Exception as e:
                        print(f"Metadata Warning: {e}")

                # 1. 提取物体点云
                # Target: 保留纹理 (use_texture=True)
                target_xyzrgb = get_world_points_from_mask(
                    target_object_ids, mask_array, depth_np, rgb_np, use_texture=True
                )
                
                # Obs: 去除颜色 (use_texture=False)，点云颜色将变成纯黑
                obs_xyzrgb = get_world_points_from_mask(
                    obs_object_ids, mask_array, depth_np, rgb_np, use_texture=False
                )

                # 2. 生成机械臂(夹爪)点云
                qpos_indicies = [i for i in range(6)] + [j for j in range(7, 12)]
                robot_joint_name = ['left_joint1', 'left_joint2', 'left_joint3', 'left_joint4', 'left_joint5', 'left_joint6','right_joint1', 'right_joint2', 'right_joint3', 'right_joint4', 'right_joint5', 'right_joint6']
                q_rad = qpos[qpos_indicies]
                gripper_pts = self.kin_helper.get_ee_pcs_through_joint(
                    robot_joint_name, q_rad, num_claw_pts=[500, 500]
                )
                gripper_xyz = np.array(gripper_pts)
                
                # === 设置夹爪颜色为纯蓝 ===
                # RGB: R=0, G=0, B=1
                gripper_color = np.array([0.0, 0.0, 1.0]) 
                gripper_colors = np.tile(gripper_color, (gripper_xyz.shape[0], 1))
                
                gripper_xyzrgb = np.hstack([gripper_xyz, gripper_colors]) \
                    if gripper_xyz.shape[0] > 0 else np.empty((0, 6))

                # 3. 组合与重采样

                # --- Target List (Target纹理 + 蓝色夹爪) ---
                combined_target = np.vstack([target_xyzrgb, gripper_xyzrgb])
                resampled_target = resample_point_cloud(combined_target, 4096) 
                obs_pcd_list.append(resampled_target.astype(np.float32))

                # --- Obs List (黑色Obs + 蓝色夹爪) ---
                # 这样网络只能看到障碍物的几何形状，看不到纹理，但能看到蓝色的夹爪位置
                combined_obs = np.vstack([obs_xyzrgb, gripper_xyzrgb])
                resampled_obs = resample_point_cloud(combined_obs, 1024)[:,:3] 
                control_pcd_list.append(resampled_obs.astype(np.float32))

            # visualize_single_frame(obs_pcd_list, control_pcd_list, frame_idx=0)
            # if str(input("input something:")) == "0":
            #     IPython.embed()
            return obs_pcd_list, control_pcd_list

class RM65Inference:
    def __init__(
        self,
        camera: RealSenseCamera,
        agent: ActionAgent,
        mode="jointangle",
        obs_horizon=2,
        action_horizon=8,
        device="gpu",
        use_point_cloud=True,
        use_image=False,
    ):
        self.camera = camera
        self.action_agent = agent
        self.mode = mode
        self.use_point_cloud = use_point_cloud
        self.use_image = use_image
        self.obs_horizon = obs_horizon
        self.action_horizon = action_horizon

        # ------- point cloud extractor -------
        self.kin_helper = KinHelper(ee_names=['left_hand_gripper', 'right_hand_gripper'])
        self.extractor = MaskPointCloudExtractor(
            camera=self.camera,
            kin_helper=self.kin_helper
        )

        # ------- device -------
        if device == "gpu":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device("cpu")

        self.freq = 1 / 15

        self.frame_lock = threading.Lock()

        self.extractor.init_tracker(
            api_token="841bcd0d691170479c7f759523be12f2",
            prompt_text="white cup.yellow cup. blue box.",
            detection_interval=1000
        )
        self.fake_gripper_state = 0.0

    def step(self, action_list):
        
        step_start_t = time.time()
        
        # --- 修正点 1: time_start 必须在循环外定义，循环内绝对不能重置！ ---
        # loop_start_time = time.time() 

        for action_id in range(self.action_horizon):
            # iter_start = time.time()
            act = action_list[action_id]
            self.action_array.append(act)
            # --- 执行动作 ---
            # t_exec_0  = time.time()
            self.action_agent.execute(act, mode=self.mode)
            # self.fake_gripper_state = act[6]     
            # t_exec_1  = time.time()
            # cprint(f"delta_time:{t_exec_1-t_exec_0}","red")
            
            
            # --- 控制频率 ---
            elapsed = time.time() - step_start_t
            sleep_time = self.freq - elapsed
            # time.sleep(max(sleep_time,0))
            step_start_t = time.time()  
            time.sleep(self.freq*3)
            # t_sleep_end = time.time()
            # --- 获取相机数据 ---
            _, rgb_frame, depth_frame = self.camera.get_frames()
            self.rgb_array.append(rgb_frame)
            self.depth_array.append(depth_frame)
            
            # --- 从 ActionAgent 获取机械臂状态 ---
            env_qpos = np.copy(self.action_agent.joint_qpos)
            # env_qpos[6] = 0.0 if self.fake_gripper_state < 0.5 else 1.0
            env_qpos[5] -= np.radians(180)
            env_pose = np.copy(self.action_agent.pose)

            self.pose_array.append(env_pose)
            self.env_qpos_array.append(env_qpos)
            
            # print(f"Iter {action_id}: Exec={t_exec_1 - t_exec_0:.4f}s | Sleep={t_sleep_end - current_time:.4f}s | Cam={t_cam_end - t_sleep_end:.4f}s")

        
        # loop_end_t = time.time()
        # cprint(f"[Loop Total] Time: {loop_end_t - step_start_t:.4f}s (Should be ~0.8s)", "yellow")
        
        # t_extract_0 = time.time()
        
        # --- recent frames ---
        recent_rgb = self.rgb_array[-self.action_horizon:]
        recent_depth = self.depth_array[-self.action_horizon:]
        recent_qpos = self.env_qpos_array[-self.action_horizon:]

        obs_pcs, control_pcs = self.extractor.extract_pcs_from_frame(
            recent_rgb, recent_depth, recent_qpos
        )
        
        # t_extract_1 = time.time()
        # cprint(f"[Extract PC] Time: {t_extract_1 - t_extract_0:.4f}s (Processing {self.obs_horizon} frames)", "yellow")
        
        # cprint(f"delta_time1:{time1-time_start},delta_time2:{time2-time1}","red")
        self.cloud_array.extend(obs_pcs)
        self.control_cloud_array.extend(control_pcs)
        qpose_stack = np.stack(self.env_qpos_array[-self.obs_horizon:], axis=0)
        pose_stack = np.stack(self.pose_array[-self.obs_horizon:], axis=0)
        obs_cloud = np.stack(self.cloud_array[-self.obs_horizon:], axis=0)
        control_obs_cloud = np.stack(self.control_cloud_array[-self.obs_horizon:], axis=0) 

        if self.mode == 'jointangle':
            obs_dict = {
                'agent_pos': torch.from_numpy(qpose_stack).unsqueeze(0).to(self.device),
                'point_cloud': torch.from_numpy(obs_cloud).unsqueeze(0).to(self.device),
                # 'control_point_cloud': torch.from_numpy(control_obs_cloud).unsqueeze(0).to(self.device),
            }
        if self.mode == 'pose10d':
            obs_dict = {
                'agent_pos': torch.from_numpy(pose_stack).unsqueeze(0).to(self.device),
                'point_cloud': torch.from_numpy(obs_cloud).unsqueeze(0).to(self.device),
                'control_point_cloud': torch.from_numpy(control_obs_cloud).unsqueeze(0).to(self.device),
            }

        cprint(f"[STEP] agent_pos shape: {obs_dict['agent_pos']}", "red")
        
        # step_end_t = time.time()
        # cprint(f"[STEP Total] {step_end_t - step_start_t:.4f}s", "red")
        return obs_dict

    def reset(self, first_init=True):
        self.rgb_array = []
        self.depth_array = []
        self.cloud_array = []
        self.control_cloud_array = []
        self.env_qpos_array = []
        self.action_array = []
        self.pose_array = []

        # --- read first frame ---
        _, rgb_frame, depth_frame = self.camera.get_frames()
        self.rgb_array.append(rgb_frame)
        self.depth_array.append(depth_frame)

        # --- 状态来自 ActionAgent ---
        env_qpos = np.copy(self.action_agent.joint_qpos)
        # env_qpos[6] = self.fake_gripper_state
        env_qpos[5] -= np.radians(180)
        env_pose = np.copy(self.action_agent.pose)

        self.env_qpos_array.append(env_qpos)
        self.pose_array.append(env_pose)

        pose_stack = np.stack([env_pose] * self.obs_horizon, axis=0)
        rgb_stack = np.stack([rgb_frame] * self.obs_horizon, axis=0)
        depth_stack = np.stack([depth_frame] * self.obs_horizon, axis=0)
        qpos_stack = np.stack([env_qpos] * self.obs_horizon, axis=0)

        while self.extractor.tracker is None:
            time.sleep(0.1)

        obs_clouds,control_obs_clouds = self.extractor.extract_pcs_from_frame(
            rgb_stack, depth_stack, qpos_stack
        )
        self.cloud_array.append(obs_clouds[-1])
        self.control_cloud_array.append(control_obs_clouds[-1])
        if self.mode == 'jointangle':
            obs_dict = {
                'agent_pos': torch.from_numpy(qpos_stack).unsqueeze(0).to(self.device),
                'point_cloud': torch.from_numpy(np.array(obs_clouds)).unsqueeze(0).to(self.device),
                # 'control_point_cloud': torch.from_numpy(np.array(control_obs_clouds)).unsqueeze(0).to(self.device),
            }
        if self.mode == 'pose10d':
            obs_dict = {
                'agent_pos': torch.from_numpy(pose_stack).unsqueeze(0).to(self.device),
                'point_cloud': torch.from_numpy(np.array(obs_clouds)).unsqueeze(0).to(self.device),
                'control_point_cloud': torch.from_numpy(np.array(control_obs_clouds)).unsqueeze(0).to(self.device),
            }

        cprint(f"[RESET] agent_pos: {obs_dict['agent_pos']}", "red")
        return obs_dict
    
    def stop(self):
        self.action_agent.stop()
        self.camera.stop()

    
GlobalHydra.instance().clear()
@hydra.main(
    config_path=str(pathlib.Path(__file__).parent.joinpath(
        'diffusion_policy_3d','config'))
)
def main(cfg: OmegaConf):
    cam = RealSenseCamera(width=640, height=480, fps=15)
    cam.start()
    # rm_interface = Realman65Interface(auto_setup=True)
    # rm_interface.reset()
    action_agent = ActionAgent()
    
    torch.manual_seed(42)
    OmegaConf.resolve(cfg)
    cls = hydra.utils.get_class(cfg._target_)
    workspace: BaseWorkspace = cls(cfg)
    use_point_cloud = True

    policy = workspace.get_model()
    # action_horizon = policy.horizon - policy.n_obs_steps + 1
    action_horizon = 12
    roll_out_length = 1e6
    first_init = True
    record_data = True

    env = RM65Inference(camera=cam, agent=action_agent, mode="jointangle",obs_horizon=2, 
                        action_horizon=action_horizon, device="cpu",
                        use_point_cloud=use_point_cloud, use_image=False)
    
    obs_dict = env.reset(first_init=first_init)
    step_count = 0
    try:
        while step_count < roll_out_length:
            time0 = time.time()
            with torch.no_grad():
                # IPython.embed()
                action = policy(obs_dict)[0]
                action_list = [act.numpy() for act in action[3:]]
            time1 = time.time()
            obs_dict = env.step(action_list)
            time2 = time.time()
            step_count += action_horizon
            print(f"step: {obs_dict['agent_pos'][0]}")
            # cprint(f"policy_time:{time1-time0},step_time:{time2-time1}","red")
    except KeyboardInterrupt:
        env.stop()
        cprint("main thread stop successfully", "green")


if __name__ == "__main__":
    main()