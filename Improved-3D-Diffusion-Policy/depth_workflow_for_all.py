# --- 导入你定义在上面或在 .py 文件中的类 ---
# from your_processor_file import OfflinePointCloudProcessor
import numpy as np
import cv2
import open3d as o3d
import os
import json
from types import SimpleNamespace
from depthto3d import OfflinePointCloudProcessor
import glob
import zarr
import sapien.core as sapien
import copy
import time
import numpy as np
import open3d as o3d
import urchin
import warnings
import os
import sys
import tempfile
import cv2
import torch
import pyrealsense2 as rs
import supervision as sv
import traceback # 用于错误处理
import utils.rotation_util as rotation_util


warnings.filterwarnings("always", category=RuntimeWarning)

# --- (保持 KinHelper, rearrange_array, np2o3d 不变) ---
# --- (但需要修改 KinHelper 的 __init__ ) ---

def rearrange_array(robot_joint_name, input_qpos):
    # ... (保持不变) ...
    func_joint_sequence = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']

    sequence = [0]*len(robot_joint_name)
    for i in range(len(robot_joint_name)):
        for j in range(len(func_joint_sequence)):
            if robot_joint_name[i] == func_joint_sequence[j]:
                sequence[i] = j

    sequence_dict = dict(zip(robot_joint_name,sequence))
    joint_angles = [0.0]*len(func_joint_sequence)
    for i in range(len(robot_joint_name)):
        joint_angles[sequence_dict[robot_joint_name[i]]] = input_qpos[i]
    return joint_angles

def np2o3d(pcd, color=None):
    # ... (保持不变) ...
    pcd = pcd.reshape(-1,3)
    pcd_o3d = o3d.geometry.PointCloud()
    if pcd.shape[0] == 0:
        return pcd_o3d
    pcd_o3d.points = o3d.utility.Vector3dVector(pcd)
    if color is not None:
        color = color.reshape(-1,3)
        assert pcd.shape[0] == color.shape[0]
        assert color.max() <= 1

        assert color.min() >= 0
        pcd_o3d.colors = o3d.utility.Vector3dVector(color)
    return pcd_o3d

class KinHelper():
    # [!! 修改 !!] 将 urdf_path 作为参数传入
    def __init__(self, urdf_path= "/home/shui/cloudfusion/DA_D03_description/urdf/rm_65_with_gripper.urdf", ee_names=None):
        # urdf_path = "/home/shui/cloudfusion/DA_D03_description/urdf/rm_65_with_gripper.urdf" # <- 不再硬编码
        self.robot_name = 'rm_65_with_gripper'
        self.urdf_robot = urchin.URDF.load(urdf_path) # <- 使用传入的路径

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
            self.ee_names= ["left_hand_gripper"]
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
        curr_qpos = rearrange_array(robot_joint_name,curr_qpos)

        if not self.init_flag:
            # left_claw_pcd, right_claw_pcd = self.init_robot_pcd(curr_qpos,num_claw_pts)
            left_claw_pcd = self.init_robot_pcd(curr_qpos,num_claw_pts)
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

    def init_robot_pcd(self,curr_qpos,num_claw_pts):
        # ... (保持不变) ...
        left_claw_pcd = self.compute_robot_pcd(curr_qpos, link_names=[self.ee_names[0]], num_pts=[num_claw_pts[0]], pcd_name=None)
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
        left_relative_transform = np.array(curr_pose[0]) @ np.linalg.pinv(self.init_pose[0])
        # right_relative_transform = np.array(curr_pose[1]) @ np.linalg.pinv(self.init_pose[1])

        augment_left_pcd = np.concatenate(
            [np.array(self.left_hand_pcd), np.ones((self.left_hand_pcd.shape[0], 1))], axis=1
        )
        # augment_right_pcd = np.concatenate(
        #     [np.array(self.right_hand_pcd), np.ones((self.right_hand_pcd.shape[0], 1))], axis=1
        # )

        left_transformed_pcd = (left_relative_transform @ augment_left_pcd.T).T[:, :3]
        # right_transformed_pcd = (right_relative_transform @ augment_right_pcd.T).T[:, :3]
        # print("update") # 减少打印
        # return left_transformed_pcd, right_transformed_pcd
        return left_transformed_pcd


# --- (保持 resample_point_cloud 不变) ---
NUM_POINTS_PER_FRAME = 4096

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

# --- [!! 重构 !!] 将旧的 main() 改为 process_episode() ---
def process_episode(episode_root_dir, data_group, processor, kin_helper, visualize=False):
    """
    处理单个 episode 目录，并将数据追加到 Zarr data_group 中。
    
    返回:
        int: 此 episode 中成功处理并添加的帧数。
    """
    
    # --- 1. 配置 ---
    # (可选) 你想提取的特定物体ID。设为 None 可提取所有物体。
    TARGET_MASK_ID = None 
    
    # --- 2. 自动路径设置 (基于 episode_root_dir) ---
    depth_dir = os.path.join(episode_root_dir, "depth_data")
    mask_dir = os.path.join(episode_root_dir, "mask_data")
    image_dir = os.path.join(episode_root_dir, "images")
    masked_depth_dir = os.path.join(episode_root_dir, "masked_depth_data")
    pcd_output_dir = os.path.join(episode_root_dir, "point_clouds") # (可选) 仍然保存PCD
    ALIGNED_DATA_FILE = os.path.join(episode_root_dir, "aligned_data.json")

    os.makedirs(masked_depth_dir, exist_ok=True)
    os.makedirs(pcd_output_dir, exist_ok=True) # (可选)

    # --- 3. 加载 Episode 特有数据 ---
    try:
        with open(ALIGNED_DATA_FILE, 'r') as f:
            aligned_data = json.load(f)
        print(f"成功加载 {ALIGNED_DATA_FILE}，包含 {len(aligned_data)} 帧的数据。")
    except Exception as e:
        print(f"错误: 无法加载 {ALIGNED_DATA_FILE}: {e}")
        return 0 # 返回0帧

    # --- 4. 查找所有要处理的帧 ---
    search_path = os.path.join(depth_dir, "depth_*.npy")
    original_depth_files = sorted(glob.glob(search_path))
    
    if not original_depth_files:
        print(f"错误: 在 {depth_dir} 中未找到 'depth_*.npy' 文件。")
        return 0
        
    print(f"--- 找到 {len(original_depth_files)} 帧，开始处理 ---")

    # --- 5. 设置可视化 (如果需要) ---
    vis = None
    if visualize:
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        pcd = o3d.geometry.PointCloud()
        is_first_frame = True
        vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5))
        
    # [!! 修改 !!] 不在这里打开Zarr，而是获取传入的 datasets
    # 我们将数据缓存在列表中，然后在 episode 结束时一次性追加，这样更快。
    episode_pcd_buffer = []
    episode_action_buffer = []
    episode_state_buffer = []

    point_cloud_arr = data_group['point_cloud']
    action_arr = data_group['action']
    state_arr = data_group['state']

    valid_frames_added = 0

    # --- 6. 遍历每一帧并处理 ---
    for frame_idx, original_depth_path in enumerate(original_depth_files):
        
        # a. 从深度文件构建所有其他路径
        frame_name = os.path.basename(original_depth_path)
        frame_id = frame_name.split('_')[1].split('.')[0]

        frame_num = int(frame_id)
        next_frame_num = frame_num + 1
        original_length = len(frame_id) # 结果是 4
        next_frame_id = str(next_frame_num).zfill(original_length)



        mask_path = os.path.join(mask_dir, f"mask_{frame_id}.npy")
        image_path = os.path.join(image_dir, f"image_{frame_id}.jpg")
        masked_depth_path = os.path.join(masked_depth_dir, f"masked_depth_{frame_id}.npy")
        pcd_save_path = os.path.join(pcd_output_dir, f"pcd_{frame_id}.pcd") # (可选)

        print(f"\n[帧 {frame_id}]")
        
        # b. 检查所有文件是否存在
        if not os.path.exists(mask_path) or not os.path.exists(image_path):
            print(f"  警告: 找不到对应的 mask 或 image 文件。跳过此帧。")
            continue
        
        frame_data = aligned_data.get(frame_id)
        #取下一帧数据
        if frame_idx != len(original_depth_files) -1:
            next_frame_data = aligned_data.get(next_frame_id)
        else:
            next_frame_data = aligned_data.get(frame_id)

        # --- [!! 用户操作 !!] ---
        # 在这里从 frame_data 中提取 action 和 state
        # 你的 schema 需要 (7,) 的 float32。
        # 假设 "joint_angles" 是你的 7 维 state
        # 假设 "action" 字段是你的 7 维 action
        if not frame_data or "joint_angles" not in frame_data:
            print(f"  警告: 在 aligned_data.json 中未找到 {frame_id} 的 'joint_angles' (state) 或 'action'。跳过。")
            continue
        if not frame_data or "ee_pose" not in frame_data:    
            print(f"  警告: 在 aligned_data.json 中未找到 {frame_id} 的 'ee_pose' (action)。跳过。")
            continue
        
        # [TODO]
        joint_angles = frame_data["joint_angles"]
        ee_pose = frame_data["ee_pose"]
        joint_angles_next = next_frame_data["joint_angles"]
        ee_pose_next = next_frame_data["ee_pose"] 

        try:
            xyz = np.array(ee_pose[0:3])
            rpy = np.array(ee_pose[3:6])
            rpy = torch.from_numpy(rpy).float()
            quat = rotation_util.eulerToQuat(rpy)
            quat = torch.from_numpy(quat).float()
            rotation_6d = rotation_util.quaternion_to_rotation_6d(quat)
            rotation_6d = rotation_6d.numpy()
            print("rotation 6d" , rotation_6d)
            state_vec = np.concatenate([xyz, rotation_6d, np.array([joint_angles[-1]])]) #(10, 
            print("state vec:",state_vec)
        except Exception as e:
            print("创建 state_vec 失败", e)

        try:
            xyz_next = np.array(ee_pose[0:3])
            rpy_next = np.array(ee_pose_next[3:6])
            rpy_next = torch.from_numpy(rpy_next).float()
            quat = rotation_util.eulerToQuat(rpy_next)
            quat = torch.from_numpy(quat).float()
            rotation_6d_next = rotation_util.quaternion_to_rotation_6d(quat)
            rotation_6d_next = rotation_6d_next.numpy()
            print("rotation 6d next:" , rotation_6d_next)
            action_vec = np.concatenate([xyz_next, rotation_6d_next, np.array([joint_angles_next[-1]])]) #(10,
            print("action vec:",action_vec)
        except Exception as e:
            print("创建action_vec 失败", e)

        try:
        
            # # (示例) 你需要根据你的 .json 结构调整
            action_for_gripper = np.array(frame_data["joint_angles"], dtype=np.float32) # (7,)
            # current_state = state_vec.astype(np.float32) # (7,)
            # current_action = action_vec.astype(np.float32) # (7,)
            current_state = state_vec.astype(np.float32)
            current_action = action_vec.astype(np.float32)
        except Exception as e:
            print(f"  警告: 从 frame_data 提取 state/action 失败: {e}。跳过。")
            continue

        if current_state.shape != (10,) or current_action.shape != (10,):
            print(f"  警告: State/Action shape 错误。需要 (10,)，但得到 {current_state.shape}, {current_action.shape}。跳过。")
            continue
        # --- [!! 用户操作结束 !!] ---


        # --- c. 工作流 步骤 1: 应用掩码 ---
        print(f"  1. 应用掩码...")
        processor.apply_segmentation_mask(
            original_depth_path, 
            mask_path, 
            masked_depth_path, 
            target_id=TARGET_MASK_ID
        )

        # --- d. 工作流 步骤 2: 转换为 3D ---
        print(f"  2. 转换为 3D 点云...")
        points, colors, rgb_image, valid_mask = processor.process_frame_from_files(
            masked_depth_path, 
            image_path
        )

        if points.shape[0] == 0:
            print(f"  警告: 未能从该帧生成任何点云。")
            continue
        
        print(f"  3. 从npy生成了 {points.shape[0]} 个点。")

        # 生成目标点云数据
        xyz_rgb_data = np.hstack((points, colors)) if points.shape[0] > 0 else np.empty((0, 6))

        # 生成夹爪点云数据
        robot_joint_name = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']
        # [!! 修改 !!] 使用我们之前提取的 action_for_gripper
        joint_states_degrees = action_for_gripper[:6] # 假设前6个是FK所需的
        current_joint_state_rad = np.radians(joint_states_degrees)

        gripper_pcd_list = kin_helper.get_ee_pcs_through_joint(
            robot_joint_name=robot_joint_name,
            robot_joint_state=current_joint_state_rad,
            num_claw_pts=[500])

        # 合并点云并重采样
        gripper_pcd_xyz = np.array(gripper_pcd_list)
        gripper_color = np.array([0.0, 0.0, 1.0]) # 蓝色夹爪
        gripper_colors = np.tile(gripper_color, (gripper_pcd_xyz.shape[0], 1))
        xyz_rgb_data_gripper = np.hstack((gripper_pcd_xyz, gripper_colors)) if gripper_pcd_xyz.shape[0] > 0 else np.empty((0, 6))
        combined_pcd_data = np.vstack((xyz_rgb_data, xyz_rgb_data_gripper))

        print(f"  4. 重采样夹爪+目标 {combined_pcd_data.shape[0]} 个点到 {NUM_POINTS_PER_FRAME}...")
        resampled_pcd_combined = resample_point_cloud(
            combined_pcd_data, 
            NUM_POINTS_PER_FRAME 
        )
        
        # [!! 修改 !!] 不立即写入Zarr，而是添加到缓冲区
        episode_pcd_buffer.append(resampled_pcd_combined.astype(np.float32))
        episode_action_buffer.append(current_action)
        episode_state_buffer.append(current_state)
        valid_frames_added += 1
        print(f"  5. 帧 {frame_id} 已缓存。")


        # --- e. (可选) 保存并更新可视化 ---
        if visualize:
            pcd.points = o3d.utility.Vector3dVector(combined_pcd_data[:, :3])
            pcd.colors = o3d.utility.Vector3dVector(combined_pcd_data[:, 3:6])
            
            # (可选) 保存单个PCD文件
            o3d.io.write_point_cloud(pcd_save_path, pcd)
            print(f"  6. (可选) 已保存到 {pcd_save_path}")

            if is_first_frame:
                vis.add_geometry(pcd)
                is_first_frame = False
            else:
                vis.update_geometry(pcd)
            
            vis.poll_events()
            vis.update_renderer()
            
            cv2.imshow("RGB Feed", rgb_image)
            cv2.imshow("Final Valid Mask", valid_mask.astype(np.uint8) * 255)
            
            if cv2.waitKey(30) & 0xFF == ord('q'):
                visualize = False # 停止后续帧的可视化
                break

    # --- 7. 清理与Zarr追加 ---
    if vis:
        vis.destroy_window()
        cv2.destroyAllWindows()
    
    if valid_frames_added > 0:
        print(f"\n--- Episode 处理完毕 ---")
        print(f"--- 正在将 {valid_frames_added} 帧追加到 Zarr... ---")
        
        # 将列表转换为堆叠的 numpy 数组
        pcd_to_append = np.stack(episode_pcd_buffer)     # (N, 4096, 6)
        action_to_append = np.stack(episode_action_buffer) # (N, 10)
        state_to_append = np.stack(episode_state_buffer)   # (N, 10)

        # 高效地追加数据
        point_cloud_arr.append(pcd_to_append)
        action_arr.append(action_to_append)
        state_arr.append(state_to_append)
        
        print("--- 追加完成。 ---")
    else:
        print("\n--- Episode 处理完毕，没有有效的帧被添加。 ---")

    return valid_frames_added


# --- [!! 新 !!] 这是你的新主函数 ---
def main():
    
    # --- 1. 数据集配置 ---
    # [!! 用户操作 !!] 设置你的基础目录和输出文件
    BASE_DATASET_DIR = "/media/shui/Lexar/put_cup_into_basket_1106_1600_1800_processed" # 包含所有 "episode_..." 文件夹的目录
    ZARR_OUTPUT_FILE = "/media/shui/Lexar/put_cup_into_basket_1106_1600_1800_processed/put_cup_into_basket_dataset_1109_6d.zarr"
    VISUALIZE_FIRST_EPISODE = True # 设置为 True 以调试并查看第一个 episode 的处理

    # [!! 用户操作 !!] 提供这些文件的 *一个* 示例路径。
    # 假设它们在所有 episodes 中都是通用的。
    # (如果 T_world_cam 每一集都变，你需要修改 process_episode 来加载它)
    T_WORLD_CAM_FILE = "processed/episode_0001_processed/T_world_cam.npy" 
    INTRINSICS_FILE = "processed/episode_0001_processed/intrinsics.json"
    DEPTH_SCALE = 0.001
    
    # [!! 用户操作 !!] 确保这个 URDF 路径正确
    URDF_PATH = "/home/shui/cloudfusion/DA_D03_description/urdf/rm_65_with_gripper.urdf"

    # --- 2. 初始化可重用对象 ---
    print("--- 1. 初始化 KinHelper ---")
    try:
        kin_helper = KinHelper(urdf_path=URDF_PATH, ee_names=['left_hand_gripper'])
    except Exception as e:
        print(f"错误: 无法初始化 KinHelper (检查 URDF 路径: {URDF_PATH}): {e}")
        return

    print("--- 2. 初始化 PointCloud Processor ---")
    try:
        # (假设 T_world_cam 在所有 episodes 中是恒定的)
        # T_world_cam = np.load(T_WORLD_CAM_FILE) 
        R_world_cam = np.array([
        [-0.07826698, -0.92756758,  0.36536649],
        [-0.99663678,  0.06387478, -0.05133362],
        [ 0.0242777,  -0.36815541, -0.92944725]
        ])
        t_world_cam = np.array([-0.62249788, -0.08463483, 0.67800801]) 
        T_world_cam = np.eye(4)
        T_world_cam[:3, :3] = R_world_cam
        T_world_cam[:3, 3] = t_world_cam 
        processor = OfflinePointCloudProcessor(
            T_world_cam, 
            INTRINSICS_FILE, 
            DEPTH_SCALE
        )
    except Exception as e:
        print(f"错误: 无法初始化 OfflinePointCloudProcessor (检查相机文件): {e}")
        return

    # --- 3. 查找所有 Episodes ---
    episode_dirs = sorted(glob.glob(os.path.join(BASE_DATASET_DIR, "episode_*_processed")))
    if not episode_dirs:
        print(f"错误: 在 {BASE_DATASET_DIR} 中未找到 'episode_*_processed' 目录。")
        return
    print(f"--- 3. 找到 {len(episode_dirs)} 个 episodes ---")

    # --- 4. 创建 Zarr 文件和数据集 ---
    print(f"--- 4. 创建 Zarr 文件: {ZARR_OUTPUT_FILE} ---")
    zarr_root = zarr.open(ZARR_OUTPUT_FILE, mode='w')
    
    data_group = zarr_root.create_group('data')
    meta_group = zarr_root.create_group('meta')
    
    # 创建可追加的数据集 (shape 从 0 开始)
    # 块 (chunks) 对性能很重要。
    point_cloud_arr = data_group.create_dataset(
        'point_cloud',
        shape=(0, NUM_POINTS_PER_FRAME, 6),
        dtype='float32',
        chunks=(1, NUM_POINTS_PER_FRAME, 6) # 按"帧"分块
    )
    action_arr = data_group.create_dataset(
        'action',
        shape=(0, 10),
        dtype='float32',
        chunks=(1024, 10) # 1024 帧一个块
    )
    state_arr = data_group.create_dataset(
        'state',
        shape=(0, 10),
        dtype='float32',
        chunks=(1024, 10)
    )
    
    episode_ends_list = []
    total_frames_processed = 0

    # --- 5. 循环处理所有 Episodes ---
    print("--- 5. 开始循环处理 Episodes ---")
    for i, episode_dir in enumerate(episode_dirs):
        print(f"\n=========================================")
        print(f"  处理 Episode {i+1}/{len(episode_dirs)}: {episode_dir}")
        print(f"=========================================")
        
        # 仅在标志为True且是第一个episode时可视化
        should_visualize = (i == 0) and VISUALIZE_FIRST_EPISODE
        
        try:
            num_added = process_episode(
                episode_root_dir=episode_dir,
                data_group=data_group,
                processor=processor,
                kin_helper=kin_helper,
                visualize=should_visualize
            )
            
            if num_added > 0:
                total_frames_processed += num_added
                episode_ends_list.append(total_frames_processed)
                print(f"Episode 处理完毕。Zarr 中总帧数: {total_frames_processed}")
            else:
                print(f"警告: Episode {episode_dir} 未添加有效帧。")

        except Exception as e:
            print(f"!!!!!!!! 处理 {episode_dir} 时发生严重错误 !!!!!!!!")
            print(f"错误: {e}")
            traceback.print_exc()
            print("...跳过此 episode 并继续...")

    # --- 6. 完成并写入 Metadata ---
    print("\n--- 6. 所有 Episodes 处理完毕。正在写入元数据... ---")
    
    # 创建 'episode_ends' 数据集
    meta_group.create_dataset(
        'episode_ends',
        data=np.array(episode_ends_list, dtype=np.int64),
        shape=(len(episode_ends_list),),
        dtype='int64'
    )
    
    print(f"数据集创建完成: {ZARR_OUTPUT_FILE}")
    print("最终 Zarr 结构:")
    print(zarr_root.tree())


# 运行新的主函数
if __name__ == "__main__":
    main()