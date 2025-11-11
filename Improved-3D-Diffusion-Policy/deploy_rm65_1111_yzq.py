import sys

from realman65.utils.data_handler import debug_print
# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import hydra
import time
from omegaconf import OmegaConf
from hydra.core.global_hydra import GlobalHydra
import pathlib
from diffusion_policy_3d.workspace.base_workspace import BaseWorkspace
import diffusion_policy_3d.common.gr1_action_util as action_util
import diffusion_policy_3d.common.rotation_util as rotation_util
import tqdm
import torch
import os 
os.environ['WANDB_SILENT'] = "True"
# allows arbitrary python code execution in configs using the ${eval:''} resolver
OmegaConf.register_new_resolver("eval", eval, replace=True)
import numpy as np
from PIL import Image
from termcolor import cprint
import threading
import socket
import IPython
e = IPython.embed
import struct
from collections import defaultdict
import cv2
import pyrealsense2 as rs 
import numpy as np
import open3d as o3d
import urchin
import sapien.core as sapien
import copy
import time
from process_data_offline_for_all import IncrementalObjectTracker
from depth_workflow_for_all import KinHelper, resample_point_cloud

from realman65.my_robot.realman_65_interface import Realman65Interface

from scipy.spatial.transform import Rotation as R


def matrix_to_rotation_6d_numpy(matrix: np.ndarray) -> np.ndarray:
    """
    将一个 (3, 3) 旋转矩阵转换为 6D 表示。
    
    参数:
    matrix: shape (3, 3) 的旋转矩阵
    
    返回:
    shape (6,) 的 6D 旋转向量 [r11, r21, r31, r12, r22, r32]
    """
    # 提取前两列并展平
    # (3, 3) -> (3, 2) -> (6,)
    return matrix[:, :2].T.flatten()


def rotation_6d_to_matrix_numpy(rot_6d: np.ndarray) -> np.ndarray:
    """
    将 (6,) 的 6D 旋转向量转换为 (3, 3) 的旋转矩阵。
    
    参数:
    rot_6d: shape (6,) 的 numpy 数组
    
    返回:
    shape (3, 3) 的旋转矩阵
    """
    if rot_6d.shape != (6,):
        raise ValueError(f"输入 shape 必须是 (6,)，但得到的是 {rot_6d.shape}")

    # 1. 提取前两列 a1 和 a2
    a1 = rot_6d[0:3]
    a2 = rot_6d[3:6]

    # 2. 格拉姆-施密特正交化
    
    # b1 = a1 / ||a1||
    # (添加 1e-6 是为了防止除以零)
    b1 = a1 / (np.linalg.norm(a1) + 1e-6)
    
    # v2 = a2 - (b1·a2) * b1
    # (计算 a2 在 b1 上的投影并减去)
    dot_b1_a2 = np.dot(b1, a2)
    v2 = a2 - dot_b1_a2 * b1
    
    # b2 = v2 / ||v2||
    b2 = v2 / (np.linalg.norm(v2) + 1e-6)

    # 3. 计算第三列 (叉积)
    # b3 = b1 x b2
    b3 = np.cross(b1, b2)
    
    # 4. 堆叠成 3x3 矩阵
    # np.stack(..., axis=1) 将 [b1, b2, b3] 作为列向量堆叠
    matrix = np.stack([b1, b2, b3], axis=1)
    
    return matrix


global rgb_frame,depth_frame

class RealSenseCamera:
    def __init__(self, width=640, height=480, fps=30):
        """
        初始化相机实例
        """
        self.width = width
        self.height = height
        self.fps = fps

        # 相机硬件相关属性
        self.pipeline = None
        self.profile = None
        self.depth_scale = None
        self.color_intrinsics = None
        self.align = None
        self.T_world_cam = None
        self.uu = None
        self.vv = None

        # 帧数据
        self.rgb_frame = None
        self.depth_frame = None
        
        # 线程相关
        self.frame_lock = threading.Lock()
        self.running = False
        self.thread = None

        # 调用您提供的初始化方法
        try:
            self._init_camera(self.width, self.height, self.fps)
            cprint(f"RealSense camera initialized at {width}x{height}, {fps} FPS.", "green")
        except Exception as e:
            cprint(f"Failed to initialize RealSense camera: {e}", "red")
            raise

    def _init_camera(self, width, height, fps):
        """
        [您提供的函数]
        初始化相机流、对齐和坐标变换
        """
        # (注意：我将您函数签名中的 'weight' 修改为了 'width' 以保持一致)
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        
        try:
            self.profile = self.pipeline.start(config)
        except Exception as e:
            cprint(f"RealSense Error: {e}", "red")
            # 重新抛出异常，因为如果相机无法启动，__init__ 应该失败
            raise

        depth_sensor = self.profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()

        color_profile = self.profile.get_stream(rs.stream.color)
        self.color_intrinsics = color_profile.as_video_stream_profile().get_intrinsics()

        align_to = rs.stream.color
        self.align = rs.align(align_to)

        # 相机坐标转换
        R_world_cam = np.array([
            [-0.07826698, -0.92756758, 0.36536649],
            [-0.99663678, 0.06387478, -0.05133362],
            [0.0242777, -0.36815541, -0.92944725]
        ])
        t_world_cam = np.array([-0.62249788, -0.08463483, 0.67800801])
        T_world_cam = np.eye(4)
        T_world_cam[:3, :3] = R_world_cam
        T_world_cam[:3, 3] = t_world_cam
        self.T_world_cam = T_world_cam

        self.uu, self.vv = np.meshgrid(np.arange(self.color_intrinsics.width),
                                        np.arange(self.color_intrinsics.height))

    def _receive_image_thread(self):
        """
        [您提供的函数，稍作修改]
        在后台线程中运行，持续获取和对齐帧。
        """
        # 稍微等待，确保相机稳定
        global rgb_frame,depth_frame
        time.sleep(1.0)
        cprint("Image receiving thread started.", "cyan")
        
        # *** 关键修改：使用 self.running 作为循环条件 ***
        while self.running:
            try:
                # 等待帧，设置一个合理的超时时间（例如5000毫秒）
                frames = self.pipeline.wait_for_frames(5000) 
                aligned_frames = self.align.process(frames)
                
                with self.frame_lock:
                    self.rgb_frame = aligned_frames.get_color_frame()
                    self.depth_frame = aligned_frames.get_depth_frame()
                    rgb_frame = np.asanyarray(self.rgb_frame.get_data())
                    depth_frame = np.asanyarray(self.depth_frame.get_data())
                # cprint("received frames","red")

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
            
            # 您原来的 sleep，用于控制循环频率（如果需要的话）
            time.sleep(0.01)
        
        cprint("Image receiving thread stopped.", "cyan")

    def start(self):
        """
        启动图像接收线程
        """
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self._receive_image_thread, daemon=True)
            self.thread.start()
            cprint("Camera thread started.", "green")
        else:
            cprint("Camera thread is already running.", "yellow")

    def stop(self):
        """
        停止图像接收线程并释放相机
        """
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
        """
        从类中安全地获取最新的帧数据（作为 NumPy 数组）
        
        :return: (bool, np.ndarray, np.ndarray) 
                 (success, rgb_image, depth_image)
        """
        with self.frame_lock:
            rgb = self.rgb_frame
            depth = self.depth_frame
        
        if rgb and depth:
            rgb_image = np.asanyarray(rgb.get_data())
            depth_image = np.asanyarray(depth.get_data())
            return True, rgb_image, depth_image
        else:
            # 帧还没有准备好
            return False, None, None

    def get_intrinsics(self):
        """
        获取相机内参
        """
        return self.color_intrinsics

    def get_depth_scale(self):
        """
        获取深度尺度
        """
        return self.depth_scale

    def get_transform(self):
        """
        获取T_world_cam变换矩阵
        """
        return self.T_world_cam

    def __enter__(self):
        """
        上下文管理器：进入
        """
        self.start()
        # 等待第一帧数据
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
        """
        上下文管理器：退出
        """
        self.stop()


class RM65Inference:
    def __init__(self,obs_horizon=2, action_horizon=8, device="gpu",
                use_point_cloud=True, use_image=False, img_size=224,
                num_points=4096, camera=None) -> None:
        self.use_point_cloud = use_point_cloud
        self.use_image = use_image
        self.obs_horizon = obs_horizon
        self.action_horizon = action_horizon
        self.kin_helper = KinHelper(ee_names=['left_hand_gripper'])
        if device == "gpu":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device("cpu")
        self.camera = camera
        # self.controller = RobotController(robot_ip=ROBOT_IP)
        self._init_camera(weight=640, height=480, fps=30)
        self.freq = 1/15
        
        self.rm_interface = Realman65Interface(auto_setup=False)
        try:
            self.rm_interface.set_up()
        except Exception as exc:
            cprint(f"Failed to set up RM65 interface: {exc}", "red")
        try:
            self.rm_interface.reset()
        except Exception as exc:
            cprint(f"Failed to reset RM65 robot: {exc}", "yellow")
            
        self.pose = np.zeros(10, dtype=np.float32)
        self.joint_qpos = np.zeros(7, dtype=np.float32)
        self._last_gripper_state = None
        self._gripper_threshold = 0.5
        
        self.frame_lock = threading.Lock()
        
        # threadings
        self._joint_qpos_thread = threading.Thread(
            target=self._receive_joint_qpos_thread, daemon=True)
        self._joint_qpos_thread.start()
        
        self._pose_thread = threading.Thread(
            target=self._receive_pose_thread, daemon=True)
        self._pose_thread.start()
        
        self._camerea_thread = threading.Thread(
            target= self._receive_image_thread,daemon=True
        )
        self._camerea_thread.start()
        self._init_tracker(api_token= "a787ccd8e3c2dc7d8e19e046ca026384",
                            prompt_text="brown-paper-cup. white-basket.",
                            detection_interval= 1000)
        time.sleep(3)


        
    def step(self,action_list):
        time_start = time.time()
        # cprint(f"length {self.action_horizon}, actionlist length {len(action_list)}","green")
        for action_id in range(self.action_horizon):

            act = action_list[action_id]
            self.action_array.append(act)
            self._execute_action_pose(act)
            elapsed = time.time() - time_start
            sleep_time = (action_id + 1) * self.freq - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

            self.rgb_array.append(self.rgb_frame)
            self.depth_array.append(self.depth_frame)
            env_qpos = np.copy(self.joint_qpos)
            self.pose_array.append(np.copy(self.pose))

            # print(env_qpos)
            self.env_qpos_array.append(env_qpos)
            
        
        self.cloud_array.extend(self.extract_pcs_from_frame(self.rgb_array[-self.action_horizon:],self.depth_array[-self.action_horizon:], self.env_qpos_array[-self.action_horizon:]))

        agent_pos = np.stack(self.pose_array[-self.obs_horizon:], axis=0)

        # cprint(f"cloud_array:{self.cloud_array}","green")
        obs_cloud = np.stack(self.cloud_array[-self.obs_horizon:], axis=0)

        obs_dict = {
            'agent_pos': torch.from_numpy(agent_pos).unsqueeze(0).to(self.device),
        }
        obs_dict['point_cloud'] = torch.from_numpy(obs_cloud).unsqueeze(0).to(self.device)
        cprint(f"obs dict:{obs_dict['agent_pos'].shape}",'red')
        return obs_dict
    
    def reset(self,first_init=True):
        self.rgb_array = []
        self.depth_array = []
        self.cloud_array = []
        self.env_qpos_array = []
        self.action_array = []
        self.pose_array = []

        self.rgb_array.append(self.rgb_frame)
        self.depth_array.append(self.depth_frame)
        env_qpos = np.copy(self.joint_qpos)
        env_pose = np.copy(self.pose)
        self.env_qpos_array.append(env_qpos)
        self.pose_array.append(env_pose)

        agent_pos = np.stack([self.env_qpos_array[-1]]*self.obs_horizon, axis=0)
        rgb = np.stack([self.rgb_array[-1]]*self.obs_horizon, axis=0)
        depth = np.stack([self.depth_array[-1]]*self.obs_horizon, axis=0)
        pose_stack = np.stack([self.pose_array[-1]]*self.obs_horizon, axis=0)

        obs_cloud = self.extract_pcs_from_frame(rgb,depth,agent_pos)
        self.cloud_array.append(obs_cloud[-1])

        obs_dict = {
            'agent_pos': torch.from_numpy(pose_stack).unsqueeze(0).to(self.device),
        }
        obs_dict['point_cloud'] = torch.from_numpy(np.array(obs_cloud)).unsqueeze(0).to(self.device)
        cprint(f"obs_dict: {obs_dict['agent_pos']}",color="red")

        return obs_dict
 
    
    def _execute_action_angle(self, action):
        action_np = np.asarray(action, dtype=np.float64).flatten()
        if action_np.size < 6:
            raise ValueError(
                f"Action dim {action_np.size} is insufficient for pose control.")
        
        # predict rad angle
        target_rad_angle = action_np[:6]
        try:
            self.rm_interface.target_joint_angles= target_rad_angle
            # 首次赋值时设置 init_ik 为 True，启动线程执行
            if not self.rm_interface.init_ik:
                self.rm_interface.init_ik = True
        except Exception as exc:
            cprint(f"Failed to apply pose command: {exc}", "red")

        if action_np.size >= 7:
            gripper_val = action_np[6]
            if not np.isnan(gripper_val):
                gripper_cmd = 1 if gripper_val >= self._gripper_threshold else 0
                if self._last_gripper_state is None or gripper_cmd != self._last_gripper_state:
                    try:
                        self.rm_interface.set_gripper("left_arm", gripper_cmd)
                        self._last_gripper_state = gripper_cmd
                    except Exception as exc:
                        cprint(f"Failed to apply gripper command: {exc}", "yellow")

    def _execute_action_pose(self, action):
        action_np = np.asarray(action, dtype=np.float64).flatten()
        # 期望 10 维: [x, y, z] + [6D 旋转] + [gripper]
        if action_np.size < 10:
            raise ValueError(
                f"Action dim {action_np.size} is insufficient for pose(need 10).")

        try:
            # xyz rad
            target_pos = action_np[:3]
            rot6d = action_np[3:9]
            rot_mat = rotation_6d_to_matrix_numpy(rot6d)
            # rotation object
            rot = R.from_matrix(rot_mat)
            xyzw_quat = rot.as_quat(scalar_first=False)
            # use ik solve
            target_rad_angle = self.rm_interface.ik_solver.move_to_pose_and_get_joints(target_pos, xyzw_quat)
            # cprint(f"target_rad_angle: {target_rad_angle}",color="blue")
            if target_rad_angle is None:
                return
            self.rm_interface.target_joint_angles = target_rad_angle
            self.rm_interface.init_ik = True

        except Exception as exc:
            cprint(f"Failed to send pose command: {exc}", "red")

        # 夹爪（第 10 维）
        gripper_val = action_np[9]
        if not np.isnan(gripper_val):
            gripper_cmd = 1 if gripper_val >= self._gripper_threshold else 0
            if self._last_gripper_state is None or gripper_cmd != self._last_gripper_state:
                try:
                    self.rm_interface.set_gripper("left_arm", gripper_cmd)
                    self._last_gripper_state = gripper_cmd
                except Exception as exc:
                    cprint(f"Failed to apply gripper command: {exc}", "yellow")
    def _init_camera(self, weight=640, height=480, fps=30):
        # init camera frame stream and align it
        # self.pipeline = rs.pipeline()
        # config = rs.config()
        # config.enable_stream(rs.stream.depth, weight, height, rs.format.z16, fps)
        # config.enable_stream(rs.stream.color, weight, height, rs.format.bgr8, fps)
        # try:
        #     self.profile = self.pipeline.start(config)
        # except Exception as e:
        #     print("RealSense Error:", e)

        # depth_sensor = self.profile.get_device().first_depth_sensor()
        # self.depth_scale = depth_sensor.get_depth_scale()
        self.depth_scale = self.camera.depth_scale
        # color_profile = self.profile.get_stream(rs.stream.color)
        # self.color_intrinsics = color_profile.as_video_stream_profile().get_intrinsics()
        self.color_intrinsics = self.camera.color_intrinsics
        # align_to = rs.stream.color
        # self.align = rs.align(align_to)

        # 相机坐标转换
        R_world_cam = np.array([
        [-0.07826698, -0.92756758,  0.36536649],
        [-0.99663678,  0.06387478, -0.05133362],
        [ 0.0242777,  -0.36815541, -0.92944725]
        ])
        t_world_cam = np.array([-0.62249788, -0.08463483, 0.67800801]) 
        T_world_cam = np.eye(4)
        T_world_cam[:3, :3] = R_world_cam
        T_world_cam[:3, 3] = t_world_cam 
        self.T_world_cam = T_world_cam

        self.uu, self.vv = np.meshgrid(np.arange(self.color_intrinsics.width), np.arange(self.color_intrinsics.height))

    def _init_tracker(self, api_token, prompt_text, detection_interval = 1000):
        self.tracker = IncrementalObjectTracker(
        api_token=api_token, # <--- 使用传入的 token
        grounding_model_name="GroundingDino-1.5-Pro",
        sam2_model_cfg="configs/sam2.1/sam2.1_hiera_t.yaml",
        sam2_ckpt_path="/home/shui/idp3_test/Improved-3D-Diffusion-Policy/Improved-3D-Diffusion-Policy/checkpoints/sam2.1_hiera_tiny.pt",
        device="cuda",
        prompt_text=prompt_text,
        detection_interval=detection_interval,
        )
        self.tracker.set_prompt(prompt_text)

    def gen_mask(self, rgb):
        process_image_rgb, mask_array = self.tracker.add_image(rgb)
        if process_image_rgb is None or not isinstance(process_image_rgb, np.ndarray):
            print(f"[Warning] (Tracker 返回空结果).")
        return mask_array
        

    def _receive_image_thread(self):
        global rgb_frame,depth_frame
        # use realsense to get latest depth image and rgb image
        # update class variable: rgb_frame and depth_frame
        # while True:
        #     try:
        #         frames = self.pipeline.wait_for_frames()
        #         aligned_frames = self.align.process(frames)
        #         with self.frame_lock:
        #             self.rgb_frame = aligned_frames.get_color_frame()
        #             self.depth_frame = aligned_frames.get_depth_frame()

        #     except RuntimeError as e:
        #         # 2. 如果发生超时 (或其他运行时错误)，捕获它
        #         if "Frame didn't arrive" in str(e):
        #             # 打印警告而不是崩溃，然后继续循环
        #             cprint("Warning: RealSense frame timeout. Re-trying...", "yellow")
        #             # 发生错误时，短暂 sleep 防止刷屏
        #             time.sleep(0.1) 
        #             continue # 继续下一次循环
        #         else:
        #             # 如果是其他未知的 RuntimeError
        #             cprint(f"Unknown RealSense error: {e}", "red")
        #             break # 退出循环
        #     except Exception as e:
        #         cprint(f"Error in image thread: {e}", "red")
        #         break
        self.rgb_frame = rgb_frame
        self.depth_frame = depth_frame
        time.sleep(0.01)


    def _receive_joint_qpos_thread(self):
        while True:
            try:
                joint_dict = self.rm_interface.get_joint_angles()
                gripper_dict = self.rm_interface.get_gripper_state()
                if joint_dict:
                    left_arm_angles = joint_dict['left_arm']
                    if left_arm_angles is not None:
                        # [1.4429999589920044, 8.208999633789062, 120.58399963378906, 0.6740000247955322, 46.41699981689453, -135.7220001220703]
                        # 转换为弧度后，赋值给前6个维度
                        self.joint_qpos[:6] = np.radians(left_arm_angles).tolist()
                if gripper_dict:
                    gripper_state = gripper_dict['left_arm']
                    # 更新第7维（夹爪状态）
                    self.joint_qpos[6] = gripper_state
            except Exception as exc:
                cprint(f"Failed to read joint state: {exc}", "yellow")
            time.sleep(0.01)
            
            
    def _receive_pose_thread(self):
        while True:
            try:
                eepose_dict = self.rm_interface.get_end_effector_pose()
                gripper_dict = self.rm_interface.get_gripper_state()
                if eepose_dict:
                    eepose = eepose_dict['left_arm']
                    # cprint(f"eepose: {eepose}",color="red")
                    if eepose is not None:
                        # cprint(f"eepose: {eepose}",color="blue")
                        # eepose: [x, y, z, roll, pitch, yaw]（rpy 为弧度）
                        # 1) 拆分 xyz 与 rpy
                        xyz = np.array(eepose[0:3], dtype=np.float32)
                        rpy = np.array(eepose[3:6], dtype=np.float32)

                        # 2) rpy -> quat -> 6D rot（参考 depth_workflow_for_all.py）
                        # 注意：eulerToQuat 期望 numpy 数组
                        quat_np = rotation_util.eulerToQuat(rpy)  # 返回 [x, y, z, w]
                        quat_t = torch.from_numpy(quat_np).float()
                        rot6d_t = rotation_util.quaternion_to_rotation_6d(quat_t)
                        rot6d = rot6d_t.numpy().reshape(-1)
                        # cprint(f"pose: {self.pose}",color="blue")
                
                        # 3) 更新 10 维 pose: [xyz(3), rot6d(6), gripper(1)]
                        self.pose[0:3] = xyz.tolist()
                        self.pose[3:9] = rot6d.tolist()
                        # cprint(f"pose: {self.pose}",color="blue")
                if gripper_dict:
                    gripper_state = gripper_dict['left_arm']
                    # 更新第10维（夹爪状态，0/1 或连续值）
                    if gripper_state is not None:
                        self.pose[9] = float(gripper_state)
            except Exception as exc:
                cprint(f"Failed to read eepose state: {exc}", "yellow")
            time.sleep(0.01)

    def extract_pcs_from_frame(self, rgb_array, depth_array, qpos_array):
        # return point cloud with input depth frame array and rgb frame array
        # rgb array; a list of rgb frames, ...
        # return pcd with length(rgb_array)
        pcd = []
        size = len(rgb_array)
        if size != len(depth_array):
            print(f"错误: RGB ({size}) 和 Depth ({len(depth_array)}) 列表的长度不匹配。")
            return []
        
        for rgb_frame, depth_frame,qpos in zip(rgb_array, depth_array,qpos_array):
            # 生成掩码
            with self.frame_lock:
                rgb_np = rgb_frame.copy()
            mask_array = self.gen_mask(rgb_np)
            boolean_mask = (mask_array != 0)
            with self.frame_lock:
                depth_image = depth_frame.copy()
            masked_depth = depth_image * boolean_mask
            masked_depth = masked_depth.astype(depth_image.dtype)
            # mask_display = (boolean_mask * 255).astype(np.uint8)
            # cv2.imshow("Generated Mask", mask_display)
            # key = cv2.waitKey(1) & 0xFF
            # if key == ord('q'):  # 按 'q' 键退出循环
            #     print("Visualization stopped by user.")
            #     break
            z_cam = masked_depth * self.depth_scale # (H, W)
            x_cam = (self.uu - self.color_intrinsics.ppx) * z_cam / self.color_intrinsics.fx
            y_cam = (self.vv - self.color_intrinsics.ppy) * z_cam / self.color_intrinsics.fy
            points_cam_np = np.stack([x_cam, y_cam, z_cam], axis=-1)

            
            valid = (z_cam > 0) & (z_cam < 2.0) # (H, W) bool
            
            points_cam_flat = points_cam_np[valid]  # (N, 3)
            colors_bgr_flat = rgb_np[valid]      # (N, 3)

            frame_rgb = rgb_np[..., ::-1].copy() # BGR to RGB
            colors_rgb_flat = colors_bgr_flat[..., ::-1] / 255.0

            # --- 坐标变换 (相机 -> 世界) ---
            points_cam_homo = np.hstack((points_cam_flat, np.ones((points_cam_flat.shape[0], 1))))
            points_world_homo = (self.T_world_cam @ points_cam_homo.T).T
            points_world_np = points_world_homo[:, :3]
            xyz_rgb_data = np.hstack((points_world_np, colors_rgb_flat)) if points_world_np.shape[0] > 0 else np.empty((0, 6))
            
            robot_joint_name = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']
        
            joint_states_rad = qpos[:6] # 假设前6个是FK所需的
            # current_joint_state_rad = np.radians(joint_states_degrees)

            gripper_pcd_list = self.kin_helper.get_ee_pcs_through_joint(
                robot_joint_name=robot_joint_name,
                robot_joint_state=joint_states_rad,
                num_claw_pts=[500])
            
            # 合并点云并重采样
            gripper_pcd_xyz = np.array(gripper_pcd_list)
            gripper_color = np.array([0.0, 0.0, 1.0]) # 蓝色夹爪
            gripper_colors = np.tile(gripper_color, (gripper_pcd_xyz.shape[0], 1))
            xyz_rgb_data_gripper = np.hstack((gripper_pcd_xyz, gripper_colors)) if gripper_pcd_xyz.shape[0] > 0 else np.empty((0, 6))
            combined_pcd_data = np.vstack((xyz_rgb_data, xyz_rgb_data_gripper))
            resampled_pcd_combined = resample_point_cloud(
                                    combined_pcd_data, 
                                    4096 
                                    )
            pcd.append(resampled_pcd_combined.astype(np.float32))
        return pcd





GlobalHydra.instance().clear()
@hydra.main(
    config_path=str(pathlib.Path(__file__).parent.joinpath(
        'diffusion_policy_3d','config'))
)
def main(cfg: OmegaConf):

    cam = RealSenseCamera(width=640, height=480, fps=30)
    cam.start()


    torch.manual_seed(42)
    OmegaConf.resolve(cfg)
    cls = hydra.utils.get_class(cfg._target_)
    workspace: BaseWorkspace = cls(cfg)
    use_point_cloud = True

    policy = workspace.get_model()
    action_horizon = policy.horizon - policy.n_obs_steps + 1
    # action_horizon = 6

    # rollout length
    roll_out_length = 1e6
    img_size = 224
    num_points = 4096
    first_init = True
    record_data = True

    env = RM65Inference(obs_horizon=2, action_horizon=action_horizon, device="cpu",
                             use_point_cloud=use_point_cloud,
                             use_image=False,
                             img_size=img_size,
                             num_points=num_points,camera=cam)
    
    # while not hasattr(env, "cloud") or not hasattr(env, "joint_qpos"):
    #     time.sleep(0.3)
    
    obs_dict = env.reset(first_init=first_init)
    step_count = 0

    while step_count < roll_out_length:
        with torch.no_grad():
            action = policy(obs_dict)[0]
            action_list = [act.numpy() for act in action]
        
        obs_dict = env.step(action_list)
        step_count += action_horizon
        print(f"step: {obs_dict['agent_pos'][0]}")

if __name__ == "__main__":
    main()