import pyrealsense2 as rs
import os
import copy
import sapien.core as sapien
import urchin
import open3d as o3d
from collections import defaultdict
import struct
import IPython
import socket
import threading
from termcolor import cprint
import numpy as np
import torch
import tqdm
import diffusion_policy_3d.common.rotation_util as rotation_util
import diffusion_policy_3d.common.gr1_action_util as action_util
from diffusion_policy_3d.workspace.base_workspace import BaseWorkspace
import pathlib
from omegaconf import OmegaConf
import time
import hydra
import sys
from realman65.my_robot.realman_65_interface import Realman65Interface
# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

os.environ['WANDB_SILENT'] = "True"
# allows arbitrary python code execution in configs using the ${eval:''} resolver
OmegaConf.register_new_resolver("eval", eval, replace=True)
e = IPython.embed


def resample_point_cloud(pcd_data, num_points):
    """
    将 (N, 6) 的点云数据重采样到 (num_points, 6)。
    ... (保持不变) ...
    """
    current_num_points = pcd_data.shape[0]

    if current_num_points == num_points:
        return pcd_data

    if current_num_points == 0:
        return np.zeros((num_points, 6), dtype=np.float32)

    if current_num_points > num_points:
        indices = np.random.choice(
            current_num_points, num_points, replace=False
        )
        return pcd_data[indices, :]

    if current_num_points < num_points:
        indices = np.random.choice(
            current_num_points, num_points, replace=True
        )
        return pcd_data[indices, :]


def rearrange_array(robot_joint_name, input_qpos):
    # ... (保持不变) ...
    func_joint_sequence = ['joint1', 'joint2',
                           'joint3', 'joint4', 'joint5', 'joint6']

    sequence = [0]*len(robot_joint_name)
    for i in range(len(robot_joint_name)):
        for j in range(len(func_joint_sequence)):
            if robot_joint_name[i] == func_joint_sequence[j]:
                sequence[i] = j

    sequence_dict = dict(zip(robot_joint_name, sequence))
    joint_angles = [0.0]*len(func_joint_sequence)
    for i in range(len(robot_joint_name)):
        joint_angles[sequence_dict[robot_joint_name[i]]] = input_qpos[i]
    return joint_angles


def np2o3d(pcd, color=None):
    # ... (保持不变) ...
    pcd = pcd.reshape(-1, 3)
    pcd_o3d = o3d.geometry.PointCloud()
    if pcd.shape[0] == 0:
        return pcd_o3d
    pcd_o3d.points = o3d.utility.Vector3dVector(pcd)
    if color is not None:
        color = color.reshape(-1, 3)
        assert pcd.shape[0] == color.shape[0]
        assert color.max() <= 1

        assert color.min() >= 0
        pcd_o3d.colors = o3d.utility.Vector3dVector(color)
    return pcd_o3d


class KinHelper():
    # [!! 修改 !!] 将 urdf_path 作为参数传入
    def __init__(self, urdf_path="/home/shui/cloudfusion/DA_D03_description/urdf/rm_65_with_gripper.urdf", ee_names=None):
        # urdf_path = "/home/shui/cloudfusion/DA_D03_description/urdf/rm_65_with_gripper.urdf" # <- 不再硬编码
        self.robot_name = 'rm_65_with_gripper'
        self.urdf_robot = urchin.URDF.load(urdf_path)  # <- 使用传入的路径

        self.engine = sapien.Engine()
        # ... (KinHelper 的其余部分保持不变) ...
        self.scene = self.engine.create_scene()
        loader = self.scene.create_urdf_loader()
        self.sapien_robot = loader.load(urdf_path)

        self.robot_model = self.sapien_robot.create_pinocchio_model()
        self.meshes = {}
        self.scales = {}
        self.offsets = {}
        print_link = []
        for link in self.urdf_robot.links:
            if len(link.collisions) > 0:
                collision = link.collisions[0]
                if len(collision.geometry.mesh.meshes) > 0:
                    mesh = collision.geometry.mesh.meshes[0]
                    self.meshes[link.name] = mesh.as_open3d
                    self.meshes[link.name].compute_vertex_normals()
                    self.meshes[link.name].paint_uniform_color([0.2, 0.2, 0.2])
                    self.scales[link.name] = collision.geometry.mesh.scale[0] if collision.geometry.mesh.scale is not None else 1.0
                    self.offsets[link.name] = collision.origin
            # print_link.append(link.name)
        # print(print_link)
        self.pcd_dict = {}
        self.tool_meshes = {}

        if ee_names == None:
            self.ee_names = ["left_hand_gripper"]
        else:
            self.ee_names = ee_names
        self.ee_ids = []
        for link_name in self.ee_names:
            for link_idx, link in enumerate(self.sapien_robot.get_links()):
                if link.name == link_name:
                    self.ee_ids.append(link_idx)
                    break
        self.nof_ee = len(self.ee_names)

        self.left_hand_pcd = None
        # self.right_hand_pcd = None
        self.init_qpos = None
        self.init_pose = None
        self.init_flag = False

    def compute_fk_sapien_links(self, qpos, link_idx):
        # ... (保持不变) ...
        fk = self.robot_model.compute_forward_kinematics(qpos)
        link_pose_ls = []
        for i in link_idx:
            link_pose_ls.append(self.robot_model.get_link_pose(
                i).to_transformation_matrix())
        return link_pose_ls

    def _mesh_poses_to_pc(self, poses, meshes, offsets, num_pts, scales, pcd_name=None):
        # ... (保持不变) ...
        try:
            assert poses.shape[0] == len(meshes)
            assert poses.shape[0] == len(offsets)
            assert poses.shape[0] == len(num_pts)
            assert poses.shape[0] == len(scales)
        except:
            raise RuntimeError('poses and meshes must have the same length')

        N = poses.shape[0]
        all_pc = []
        for index in range(N):
            mat = poses[index]
            if pcd_name is None or pcd_name not in self.pcd_dict or len(self.pcd_dict[pcd_name]) <= index:
                mesh = copy.deepcopy(meshes[index])
                mesh.scale(scales[index], center=np.array([0, 0, 0]))
                sampled_cloud = mesh.sample_points_poisson_disk(
                    number_of_points=num_pts[index])
                cloud_points = np.asarray(sampled_cloud.points)
                if pcd_name not in self.pcd_dict:
                    self.pcd_dict[pcd_name] = []
                self.pcd_dict[pcd_name].append(cloud_points)
            else:
                cloud_points = self.pcd_dict[pcd_name][index]
            tf_obj_to_link = offsets[index]

            mat = mat @ tf_obj_to_link
            transformed_points = cloud_points @ mat[:3, :3].T + mat[:3, 3]
            all_pc.append(transformed_points)
        all_pc = np.concatenate(all_pc, axis=0)
        return all_pc

    def compute_robot_pcd(self, qpos, link_names=None, num_pts=None, pcd_name=None):
        # ... (保持不变) ...
        fk = self.robot_model.compute_forward_kinematics(qpos)
        joint_names = []
        for i, joint in enumerate(self.sapien_robot.get_active_joints()):
            joint_names.append(joint.get_name())
        if link_names is None:
            link_names = self.meshes.keys()
        # print(link_names)
        if num_pts is None:
            num_pts = [500] * len(link_names)
        link_idx_ls = []
        for link_name in link_names:
            for link_idx, link in enumerate(self.sapien_robot.get_links()):

                if link.name == link_name:
                    link_idx_ls.append(link_idx)
                    break
        link_pose_ls = np.stack([self.robot_model.get_link_pose(
            link_idx).to_transformation_matrix() for link_idx in link_idx_ls])
        meshes_ls = [self.meshes[link_name] for link_name in link_names]
        offsets_ls = [self.offsets[link_name] for link_name in link_names]
        scales_ls = [self.scales[link_name] for link_name in link_names]
        pcd = self._mesh_poses_to_pc(poses=link_pose_ls, meshes=meshes_ls,
                                     offsets=offsets_ls, num_pts=num_pts, scales=scales_ls, pcd_name=pcd_name)
        return pcd

    def gen_robot_meshes(self, qpos, link_names=None):
        # ... (保持不变) ...
        fk = self.robot_model.compute_forward_kinematics(qpos)
        if link_names is None:
            link_names = self.meshes.keys()
        link_idx_ls = []
        for link_name in link_names:
            for link_idx, link in enumerate(self.sapien_robot.get_links()):
                if link.name == link_name:
                    link_idx_ls.append(link_idx)
                    break

        link_pose_ls = np.stack([self.robot_model.get_link_pose(
            link_idx).to_transformation_matrix() for link_idx in link_idx_ls])
        offsets_ls = [self.offsets[link_name] for link_name in link_names]
        meshes_ls = []
        for link_idx, link_name in enumerate(link_names):
            import copy
            mesh = copy.deepcopy(self.meshes[link_name])
            mesh.scale(0.001, center=np.array([0, 0, 0]))
            tf = link_pose_ls[link_idx] @ offsets_ls[link_idx]
            mesh.transform(tf)
            meshes_ls.append(mesh)
        return meshes_ls

    def get_ee_pcs_through_joint(self, robot_joint_name, robot_joint_state, num_claw_pts=[150], return_o3d=False):
        # ... (保持不变) ...
        curr_qpos = np.array(robot_joint_state[:])
        robot_joint_name = robot_joint_name
        curr_qpos = rearrange_array(robot_joint_name, curr_qpos)

        if not self.init_flag:
            # left_claw_pcd, right_claw_pcd = self.init_robot_pcd(curr_qpos,num_claw_pts)
            left_claw_pcd = self.init_robot_pcd(curr_qpos, num_claw_pts)
        else:
            # left_claw_pcd, right_claw_pcd = self.update_robot_pcd(curr_qpos)
            left_claw_pcd = self.update_robot_pcd(curr_qpos)
        left_claw_pcd = left_claw_pcd.tolist()
        # right_claw_pcd = right_claw_pcd.tolist()
        # claw_pcd = left_claw_pcd + right_claw_pcd
        # claw_pcd_o3d = np2o3d(np.array(claw_pcd))
        claw_pcd_o3d = np2o3d(np.array(left_claw_pcd))
        curr_claw_pcd = copy.deepcopy(claw_pcd_o3d)

        if return_o3d:
            return curr_claw_pcd
        # return claw_pcd
        return left_claw_pcd

    def init_robot_pcd(self, curr_qpos, num_claw_pts):
        # ... (保持不变) ...
        left_claw_pcd = self.compute_robot_pcd(curr_qpos, link_names=[
                                               self.ee_names[0]], num_pts=[num_claw_pts[0]], pcd_name=None)
        # right_claw_pcd = self.compute_robot_pcd(curr_qpos, link_names=[self.ee_names[1]], num_pts=[num_claw_pts[1]], pcd_name=None)
        self.left_hand_pcd = left_claw_pcd
        # self.right_hand_pcd = right_claw_pcd
        self.init_qpos = curr_qpos
        self.init_pose = self.compute_fk_sapien_links(curr_qpos, self.ee_ids)
        self.init_flag = True

        # return left_claw_pcd, right_claw_pcd
        return left_claw_pcd

    def update_robot_pcd(self, curr_qpos):
        # ... (保持不变) ...
        curr_pose = self.compute_fk_sapien_links(curr_qpos, self.ee_ids)
        left_relative_transform = np.array(
            curr_pose[0]) @ np.linalg.pinv(self.init_pose[0])
        # right_relative_transform = np.array(curr_pose[1]) @ np.linalg.pinv(self.init_pose[1])

        augment_left_pcd = np.concatenate(
            [np.array(self.left_hand_pcd), np.ones((self.left_hand_pcd.shape[0], 1))], axis=1
        )
        # augment_right_pcd = np.concatenate(
        #     [np.array(self.right_hand_pcd), np.ones((self.right_hand_pcd.shape[0], 1))], axis=1
        # )

        left_transformed_pcd = (left_relative_transform @
                                augment_left_pcd.T).T[:, :3]
        # right_transformed_pcd = (right_relative_transform @ augment_right_pcd.T).T[:, :3]
        # print("update") # 减少打印
        # return left_transformed_pcd, right_transformed_pcd
        return left_transformed_pcd


class RM65Inference:
    def __init__(self, obs_horizon=2, action_horizon=8, device="gpu",
                 use_point_cloud=True, use_image=False, img_size=224,
                 num_points=4096) -> None:
        self.use_point_cloud = use_point_cloud
        self.use_image = use_image
        self.obs_horizon = obs_horizon
        self.action_horizon = action_horizon
        self.kin_helper = KinHelper(ee_names=['left_hand_gripper'])
        if device == "gpu":
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device("cpu")
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
        self.joint_qpos = [0.0 for _ in range(6)]
        self._last_gripper_state = None
        self._gripper_threshold = 0.5
        self._qpos_thread = threading.Thread(
            target=self._receive_qpos_thread, daemon=True)
        self._qpos_thread.start()
        
        
        
    def step(self, action_list):
        time_start = time.time()
        for action_id in range(self.action_horizon):
            act = action_list[action_id]
            self.action_array.append(act)
            self._execute_action(act)
            elapsed = time.time() - time_start
            sleep_time = (action_id + 1) * self.freq - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

            self.rgb_array.append(self.rgb_frame)
            self.depth_array.append(self.depth_frame)
            env_qpos = np.copy(self.joint_qpos)

            # print(env_qpos)
            self.env_qpos_array.append(env_qpos)

        self.cloud_array.append(self.extract_pcs_from_frame(
            self.rgb_array[-self.action_horizon:], self.depth_array[-self.action_horizon:], self.env_qpos_array[-self.action_horizon:]))

        agent_pos = np.stack(self.env_qpos_array[-self.obs_horizon:], axis=0)
        obs_cloud = np.stack(self.cloud_array[-self.obs_horizon:], axis=0)

        obs_dict = {
            'agent_pos': torch.from_numpy(agent_pos).unsqueeze(0).to(self.device),
        }
        obs_dict['point_cloud'] = torch.from_numpy(
            obs_cloud).unsqueeze(0).to(self.device)
        return obs_dict

    def reset(self, first_init=True):
        self.rgb_array = []
        self.depth_array = []
        self.cloud_array = []
        self.env_qpos_array = []
        self.action_array = []

        self.rgb_array.append(self.rgb_frame)
        self.depth_array.append(self.depth_frame)
        env_qpos = np.copy(self.joint_qpos)
        self.env_qpos_array.append(env_qpos)

        agent_pos = np.stack([self.env_qpos_array[-1]]
                             * self.obs_horizon, axis=0)
        rgb = np.stack([self.rgb_array[-1]]*self.obs_horizon, axis=0)
        depth = np.stack([self.depth_array[-1]]*self.obs_horizon, axis=0)

        obs_cloud = self.extract_pcs_from_frame(rgb, depth, agent_pos)
        self.cloud_array.append(obs_cloud[-1])

        obs_dict = {
            'agent_pos': torch.from_numpy(agent_pos).unsqueeze(0).to(self.device),
        }
        obs_dict['point_cloud'] = torch.from_numpy(
            obs_cloud).unsqueeze(0).to(self.device)

        return obs_dict
    

    def _execute_action(self, action):
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

    def _init_camera(self, weight=640, height=480, fps=30):
        # init camera frame stream and align it
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, weight,
                             height, rs.format.z16, fps)
        config.enable_stream(rs.stream.color, weight,
                             height, rs.format.bgr8, fps)
        try:
            self.profile = self.pipeline.start(config)
        except Exception as e:
            print("RealSense Error:", e)

        depth_sensor = self.profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()

        color_profile = self.profile.get_stream(rs.stream.color)
        self.color_intrinsics = color_profile.as_video_stream_profile().get_intrinsics()

        align_to = rs.stream.color
        self.align = rs.align(align_to)

        # 相机坐标转换
        R_world_cam = np.array([
            [-0.07826698, -0.92756758,  0.36536649],
            [-0.99663678,  0.06387478, -0.05133362],
            [0.0242777,  -0.36815541, -0.92944725]
        ])
        t_world_cam = np.array([-0.62249788, -0.08463483, 0.67800801])
        T_world_cam = np.eye(4)
        T_world_cam[:3, :3] = R_world_cam
        T_world_cam[:3, 3] = t_world_cam
        self.T_world_cam = T_world_cam

        self.uu, self.vv = np.meshgrid(np.arange(
            self.color_intrinsics.width), np.arange(self.color_intrinsics.height))

    def _receive_image_thread(self):
        # use realsense to get latest depth image and rgb image
        # update class variable: rgb_frame and depth_frame
        while True:
            frames = self.pipeline.wait_for_frames()
            aligned_frames = self.align.process(frames)
            self.rgb_frame = aligned_frames.get_color_frame()
            self.depth_frame = aligned_frames.get_depth_frame()

    def _receive_qpos_thread(self):
        # use robot api to get latest joint pos
        # update class variable: joint_qpos
        poll_period = 0.01
        while True:
            try:
                joint_dict = self.rm_interface.get_joint_angles()
                
                if joint_dict:
                    left_arm_angles = joint_dict['left_arm']
                    if left_arm_angles is not None:
                        # [1.4429999589920044, 8.208999633789062, 120.58399963378906, 0.6740000247955322, 46.41699981689453, -135.7220001220703]
                        # convert radians
                        self.joint_qpos = np.radians(left_arm_angles).tolist()
            except Exception as exc:
                cprint(f"Failed to read joint state: {exc}", "yellow")
            time.sleep(poll_period)

    def extract_pcs_from_frame(self, rgb_array, depth_array, qpos_array):
        # return point cloud with input depth frame array and rgb frame array
        # rgb array; a list of rgb frames, ...
        # return pcd with length(rgb_array)
        pcd = []
        size = len(rgb_array)
        if size != len(depth_array):
            print(f"错误: RGB ({size}) 和 Depth ({len(depth_array)}) 列表的长度不匹配。")
            return []

        for rgb_frame, depth_frame, qpos in zip(rgb_array, depth_array, qpos_array):
            z_cam = depth_frame * self.depth_scale  # (H, W)
            x_cam = (self.uu - self.color_intrinsics.ppx) * \
                z_cam / self.color_intrinsics.fx
            y_cam = (self.vv - self.color_intrinsics.ppy) * \
                z_cam / self.color_intrinsics.fy
            points_cam_np = np.stack([x_cam, y_cam, z_cam], axis=-1)

            valid = (z_cam > 0) & (z_cam < 2.0)  # (H, W) bool

            points_cam_flat = points_cam_np[valid]  # (N, 3)
            colors_bgr_flat = rgb_frame[valid]      # (N, 3)

            frame_rgb = rgb_frame[..., ::-1].copy()  # BGR to RGB
            colors_rgb_flat = colors_bgr_flat[..., ::-1] / 255.0

            # --- 坐标变换 (相机 -> 世界) ---
            points_cam_homo = np.hstack(
                (points_cam_flat, np.ones((points_cam_flat.shape[0], 1))))
            points_world_homo = (self.T_world_cam @ points_cam_homo.T).T
            points_world_np = points_world_homo[:, :3]
            xyz_rgb_data = np.hstack(
                (points_world_np, colors_rgb_flat)) if points_world_np.shape[0] > 0 else np.empty((0, 6))

            robot_joint_name = ['joint1', 'joint2',
                                'joint3', 'joint4', 'joint5', 'joint6']

            joint_states_degrees = qpos[:6]  # 假设前6个是FK所需的
            current_joint_state_rad = np.radians(joint_states_degrees)

            gripper_pcd_list = self.kin_helper.get_ee_pcs_through_joint(
                robot_joint_name=robot_joint_name,
                robot_joint_state=current_joint_state_rad,
                num_claw_pts=[500])

            # 合并点云并重采样
            gripper_pcd_xyz = np.array(gripper_pcd_list)
            gripper_color = np.array([0.0, 0.0, 1.0])  # 蓝色夹爪
            gripper_colors = np.tile(
                gripper_color, (gripper_pcd_xyz.shape[0], 1))
            xyz_rgb_data_gripper = np.hstack(
                (gripper_pcd_xyz, gripper_colors)) if gripper_pcd_xyz.shape[0] > 0 else np.empty((0, 6))
            combined_pcd_data = np.vstack((xyz_rgb_data, xyz_rgb_data_gripper))
            resampled_pcd_combined = resample_point_cloud(
                combined_pcd_data,
                4096
            )
            pcd.append(resampled_pcd_combined.astype(np.float32))
        return pcd


@hydra.main(
    config_path=str(pathlib.Path(__file__).parent.joinpath(
        'diffusion_policy_3d', 'config'))
)
def main(cfg: OmegaConf):
    torch.manual_seed(42)
    OmegaConf.resolve(cfg)
    cls = hydra.utils.get_class(cfg._target_)
    workspace: BaseWorkspace = cls(cfg)
    use_point_cloud = True

    policy = workspace.get_model(tag="400")
    action_horizon = policy.horizon - policy.n_obs_steps + 1

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
                        num_points=num_points)

    # while not hasattr(env, "cloud") or not hasattr(env, "joint_qpos"):
    #     time.sleep(0.3)

    obs_dict = env.reset(first_init=first_init)
    step_count = 0

    # while step_count < roll_out_length:
    #     with torch.no_grad():
    #         action = policy(obs_dict)[0]
    #         action_list = [act.numpy() for act in action]

    #     obs_dict = env.step(action_list)
    #     step_count += action_horizon
    #     print(f"step: {obs_dict['agent_pos'][0]}")


if __name__ == "__main__":
    main()
