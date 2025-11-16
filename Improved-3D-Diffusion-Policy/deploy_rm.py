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
from depth_workflow_for_all import KinHelper, resample_point_cloud
from scipy.spatial.transform import Rotation as R
from realman65.my_robot.realman_65_interface import Realman65Interface
import copy

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
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        
        try:
            self.profile = self.pipeline.start(config)
        except Exception as e:
            cprint(f"RealSense Error: {e}", "red")
            raise

        depth_sensor = self.profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()

        color_profile = self.profile.get_stream(rs.stream.color)
        self.color_intrinsics = color_profile.as_video_stream_profile().get_intrinsics()

        align_to = rs.stream.color
        self.align = rs.align(align_to)

        R_world_cam = np.array([[-0.03236645, -0.86951084,  0.49285222],
                                [-0.99945595,  0.02502858, -0.02147949],
                                [ 0.00634126, -0.4932793,  -0.86984787]])
        t_world_cam = np.array([-0.68853796,-0.1209404,0.77337429])
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
        # 原 Realman65Interface 迁移到这里
        self.rm_interface = Realman65Interface(auto_setup=False)

        try:
            self.rm_interface.set_up()
        except Exception as exc:
            cprint(f"Failed to set up RM65 interface: {exc}", "red")
        try:
            self.rm_interface.reset()
        except Exception as exc:
            cprint(f"Failed to reset RM65 robot: {exc}", "yellow")

        self._gripper_threshold = gripper_threshold
        self._last_gripper_state = None

        self.joint_qpos = np.zeros(7, dtype=np.float32)
        self.pose = np.zeros(10, dtype=np.float32)

        self._run_state_thread = True
        self._state_thread = threading.Thread(target=self._receive_state_thread, daemon=True)
        self._state_thread.start()

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
        action_np = np.asarray(action, dtype=np.float64).flatten()
        if action_np.size < 6:
            raise ValueError("Action dim must be >= 6 for angle control.")

        target_rad_angle = action_np[:6]
        try:
            self.rm_interface.target_joint_angles = target_rad_angle
            if not self.rm_interface.init_ik:
                self.rm_interface.init_ik = True
        except Exception as exc:
            cprint(f"[Angle] Failed to apply pose command: {exc}", "red")

        # ---- Gripper ----
        if action_np.size >= 7:
            gripper_val = action_np[6]
            self._apply_gripper(gripper_val)

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
                gripper_state = None

                # ---- Joint angles ----
                if joint_dict:
                    left_arm_angles = joint_dict.get("left_arm")
                    if left_arm_angles is not None:
                        self.joint_qpos[:6] = np.radians(left_arm_angles).astype(np.float32)

                # ---- Gripper ----
                if gripper_dict:
                    gripper_state = gripper_dict.get("left_arm")
                    if gripper_state is not None:
                        self.joint_qpos[6] = gripper_state

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

                if gripper_state is not None:
                    self.pose[9] = float(gripper_state)

            except Exception as exc:
                cprint(f"Failed to read sensor data: {exc}", "yellow")

            time.sleep(0.01)   # 10ms

    # 停止线程接口
    def stop(self):
        self._run_state_thread = False
        self._state_thread.join()

class MaskPointCloudExtractor:
    """
    将 tracker 初始化、mask 生成、点云提取统一封装成一个类。
    """
    def __init__(self, camera: RealSenseCamera, kin_helper):
        self.camera = camera
        self.kin_helper = kin_helper
        self.tracker = None

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

        process_img, mask_array = self.tracker.add_image(rgb)
        if process_img is None or not isinstance(process_img, np.ndarray):
            cprint("[Warning] Tracker返回空 mask。","yellow")
            return np.zeros((self.camera.height, self.camera.width), dtype=np.uint8)
        return mask_array


    def extract_pcs_from_frame(self, rgb_array, depth_array, qpos_array):

        if self.tracker is None:
            raise RuntimeError("Tracker 尚未初始化，请先调用 init_tracker().")

        pcd_list = []
        size = len(rgb_array)
        if size != len(depth_array):
            print(f"错误: RGB({size}) 和 Depth({len(depth_array)}) 长度不一致。")
            return []

        for rgb_np, depth_np, qpos in zip(rgb_array, depth_array, qpos_array):
            mask_array = self.gen_mask(rgb_np)
            boolean_mask = (mask_array != 0)

            masked_depth = (depth_np * boolean_mask).astype(depth_np.dtype)
            z_cam = masked_depth * self.camera.get_depth_scale()

            intr = self.camera.get_intrinsics()
            x_cam = (self.camera.uu - intr.ppx) * z_cam / intr.fx
            y_cam = (self.camera.vv - intr.ppy) * z_cam / intr.fy
            points_cam = np.stack([x_cam, y_cam, z_cam], axis=-1)

            valid = (z_cam > 0) & (z_cam < 2.0)
            points_cam_flat = points_cam[valid]
            colors_bgr_flat = rgb_np[valid]

            # BGR → RGB
            colors_rgb_flat = colors_bgr_flat[..., ::-1] / 255.0

            points_cam_homo = np.concatenate(
                [points_cam_flat, np.ones((points_cam_flat.shape[0], 1))],
                axis=1
            )
            world_homo = (self.camera.get_transform() @ points_cam_homo.T).T
            world_xyz = world_homo[:, :3]

            xyz_rgb = np.hstack([world_xyz, colors_rgb_flat]) \
                if world_xyz.shape[0] > 0 else np.empty((0, 6))

            robot_joint_name = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']
            q_rad = qpos[:6]
            gripper_pts = self.kin_helper.get_ee_pcs_through_joint(
                robot_joint_name,
                q_rad,
                num_claw_pts=[500]
            )
            gripper_xyz = np.array(gripper_pts)

            gripper_color = np.array([0.0, 0.0, 1.0])
            gripper_colors = np.tile(gripper_color, (gripper_xyz.shape[0], 1))

            gripper_xyz_rgb = np.hstack([gripper_xyz, gripper_colors]) \
                if gripper_xyz.shape[0] > 0 else np.empty((0, 6))

            combined = np.vstack([xyz_rgb, gripper_xyz_rgb])
            resampled_pcd = resample_point_cloud(combined, 4096)
            pcd_list.append(resampled_pcd.astype(np.float32))

        return pcd_list

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
        self.kin_helper = KinHelper(ee_names=['left_hand_gripper'])
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
            api_token="a06fd6f7e5c5cb3586bad2ac1b40597f",
            prompt_text="blue-paper-cup. white-basket.",
            detection_interval=1000
        )

    def step(self, action_list):
        time_start = time.time()

        for action_id in range(self.action_horizon):
            act = action_list[action_id]
            self.action_array.append(act)
            # --- 执行动作 ---
            self.action_agent.execute(act, mode=self.mode)
            # --- 控制频率 ---
            elapsed = time.time() - time_start
            sleep_time = (action_id + 1) * self.freq - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

            # --- 获取相机数据 ---
            _, rgb_frame, depth_frame = self.camera.get_frames()
            self.rgb_array.append(rgb_frame)
            self.depth_array.append(depth_frame)
            # --- 从 ActionAgent 获取机械臂状态 ---
            env_qpos = np.copy(self.action_agent.joint_qpos)
            env_pose = np.copy(self.action_agent.pose)

            self.pose_array.append(env_pose)
            self.env_qpos_array.append(env_qpos)

        # --- recent frames ---
        recent_rgb = self.rgb_array[-self.action_horizon:]
        recent_depth = self.depth_array[-self.action_horizon:]
        recent_qpos = self.env_qpos_array[-self.action_horizon:]

        pcs = self.extractor.extract_pcs_from_frame(
            recent_rgb, recent_depth, recent_qpos
        )
        self.cloud_array.extend(pcs)
        qpose_stack = np.stack(self.env_qpos_array[-self.obs_horizon:], axis=0)
        pose_stack = np.stack(self.pose_array[-self.obs_horizon:], axis=0)
        obs_cloud = np.stack(self.cloud_array[-self.obs_horizon:], axis=0)

        if self.mode == 'jointangle':
            obs_dict = {
                'agent_pos': torch.from_numpy(qpose_stack).unsqueeze(0).to(self.device),
                'point_cloud': torch.from_numpy(obs_cloud).unsqueeze(0).to(self.device)
            }
        if self.mode == 'pose10d':
            obs_dict = {
                'agent_pos': torch.from_numpy(pose_stack).unsqueeze(0).to(self.device),
                'point_cloud': torch.from_numpy(obs_cloud).unsqueeze(0).to(self.device)
            }

        cprint(f"[STEP] agent_pos shape: {obs_dict['agent_pos'].shape}", "red")

        return obs_dict

    def reset(self, first_init=True):
        self.rgb_array = []
        self.depth_array = []
        self.cloud_array = []
        self.env_qpos_array = []
        self.action_array = []
        self.pose_array = []

        # --- read first frame ---
        _, rgb_frame, depth_frame = self.camera.get_frames()
        self.rgb_array.append(rgb_frame)
        self.depth_array.append(depth_frame)

        # --- 状态来自 ActionAgent ---
        env_qpos = np.copy(self.action_agent.joint_qpos)
        env_pose = np.copy(self.action_agent.pose)

        self.env_qpos_array.append(env_qpos)
        self.pose_array.append(env_pose)

        pose_stack = np.stack([env_pose] * self.obs_horizon, axis=0)
        rgb_stack = np.stack([rgb_frame] * self.obs_horizon, axis=0)
        depth_stack = np.stack([depth_frame] * self.obs_horizon, axis=0)
        qpos_stack = np.stack([env_qpos] * self.obs_horizon, axis=0)

        while self.extractor.tracker is None:
            time.sleep(0.1)

        obs_clouds = self.extractor.extract_pcs_from_frame(
            rgb_stack, depth_stack, qpos_stack
        )
        self.cloud_array.append(obs_clouds[-1])

        if self.mode == 'jointangle':
            obs_dict = {
                'agent_pos': torch.from_numpy(qpos_stack).unsqueeze(0).to(self.device),
                'point_cloud': torch.from_numpy(np.array(obs_clouds)).unsqueeze(0).to(self.device),
            }
        if self.mode == 'pose10d':
            obs_dict = {
                'agent_pos': torch.from_numpy(pose_stack).unsqueeze(0).to(self.device),
                'point_cloud': torch.from_numpy(np.array(obs_clouds)).unsqueeze(0).to(self.device),
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
    action_horizon = policy.horizon - policy.n_obs_steps + 1
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
            with torch.no_grad():
                action = policy(obs_dict)[0]
                action_list = [act.numpy() for act in action]
            
            obs_dict = env.step(action_list)
            step_count += action_horizon
            print(f"step: {obs_dict['agent_pos'][0]}")
    except KeyboardInterrupt:
        env.stop()
        cprint("main thread stop successfully", "green")


if __name__ == "__main__":
    main()