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
                cprint("frames receivede","red")

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
            self.thread = threading.Thread(target=self._receive_image_thread)
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
            rgb_image = np.asanyarray(rgb.get_data())
            depth_image = np.asanyarray(depth.get_data())
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


if __name__ == "__main__":
    cam = RealSenseCamera()
    cam.start()
