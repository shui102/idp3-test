import sapien.core as sapien
import copy
import time
import numpy as np
import open3d as o3d
import urchin
import warnings
import os
import sys
# import pinocchio as pin  # [新增] 需要直接引用 pinocchio
# 假设接口没有变，还是引用这个
from realman65.my_robot.realman_65_interface_dual import Realman65Interface


warnings.filterwarnings("always", category=RuntimeWarning)

def rearrange_array(robot_joint_name, input_qpos):
    # 双臂关节顺序映射
    func_joint_sequence = [
        'left_joint1', 'left_joint2', 'left_joint3', 'left_joint4', 'left_joint5', 'left_joint6',
        'right_joint1', 'right_joint2', 'right_joint3', 'right_joint4', 'right_joint5', 'right_joint6'
    ]

    sequence = [0]*len(robot_joint_name)
    for i in range(len(robot_joint_name)):
        for j in range(len(func_joint_sequence)):
            if robot_joint_name[i] == func_joint_sequence[j]:
                sequence[i] = j

    sequence_dict = dict(zip(robot_joint_name, sequence))
    joint_angles = [0.0]*len(func_joint_sequence)
    
    try:
        for i in range(len(robot_joint_name)):
            idx = sequence_dict.get(robot_joint_name[i])
            if idx is not None:
                joint_angles[idx] = input_qpos[i]
    except Exception as e:
        print(f"Joint mapping error: {e}")
        
    return joint_angles

import sapien.core as sapien
import copy
import time
import numpy as np
import open3d as o3d
import urchin
import warnings
import os
import sys

# from my_robot.realman_65_interface import *

warnings.filterwarnings("always", category=RuntimeWarning)

def np2o3d(pcd, color=None):
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
    def __init__(self, ee_names=None):
        urdf_path = "/home/shui/cloudfusion/DA_D03_description/urdf/rm_65_dual.urdf"
        self.assets_path = "/home/shui/cloudfusion/DA_D03_description"
        
        # 1. 定义我们要控制的“白名单”关节
        # 只有在列表里的关节会被赋值，其他关节（如夹爪）将保持为 0
        self.controlled_names = [
            'left_joint1', 'left_joint2', 'left_joint3', 'left_joint4', 'left_joint5', 'left_joint6',
            'right_joint1', 'right_joint2', 'right_joint3', 'right_joint4', 'right_joint5', 'right_joint6'
        ]

        # 2. 加载模型 (Urchin 用于 Mesh)
        self.urdf_robot = urchin.URDF.load(urdf_path)

        # 3. 加载 Sapien
        self.engine = sapien.Engine()
        self.scene = self.engine.create_scene()
        loader = self.scene.create_urdf_loader()
        
        # 【关键步骤 1】固定根节点
        # 这会强制移除 world_to_base 这种 6自由度的浮动关节
        loader.fix_root_link = True 
        
        self.sapien_robot = loader.load(urdf_path)
        
        # 【关键步骤 2】建立索引映射
        # get_active_joints() 会自动排除 URDF 中的 fixed joint
        self.active_joints = self.sapien_robot.get_active_joints()
        self.active_joint_names = [j.name for j in self.active_joints]
        self.dof = len(self.active_joints)
        
        print(f"Sapien Active Joints (DoF={self.dof}): {self.active_joint_names}")
        
        # 构建 map: { 'joint_name': index_in_sapien_qpos }
        self.joint_name_to_index = {name: i for i, name in enumerate(self.active_joint_names)}
        
        # 验证是否所有受控关节都在模型里
        self.controlled_indices = [] # 存储这12个关节在全长数组中的下标
        for name in self.controlled_names:
            if name in self.joint_name_to_index:
                self.controlled_indices.append(self.joint_name_to_index[name])
            else:
                print(f"Error: Controlled joint '{name}' not found in Sapien active joints!")
                # 如果找不到，给一个 dummy index 或者报错，这里设为 -1 方便 debug
                self.controlled_indices.append(-1)

        # 创建 Pinocchio Model (Sapien 内置的)
        self.robot_model = self.sapien_robot.create_pinocchio_model()
        
        # 加载 Meshes (保持不变)
        self.meshes = {}
        self.scales = {}
        self.offsets = {}
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
        
        self.pcd_dict = {}

        if ee_names is None:
            self.ee_names = ["left_hand_gripper", "right_hand_gripper"]
        else:
            self.ee_names = ee_names
            
        self.ee_ids = []
        for link_name in self.ee_names:
            found = False
            for link_idx, link in enumerate(self.sapien_robot.get_links()):
                if link.name == link_name:
                    self.ee_ids.append(link_idx)
                    found = True
                    break
            if not found:
                print(f"Warning: Link {link_name} not found in URDF!")

        self.left_hand_pcd = None
        self.right_hand_pcd = None 
        self.init_qpos = None
        self.init_pose = None
        self.init_flag = False

    def get_full_qpos(self, input_values):
        """
        核心逻辑：将 12 维的输入，扩展为 Sapien 需要的 N 维输入
        """
        # 1. 创建全 0 数组，长度等于模型实际的 DoF (包含夹爪等)
        full_qpos = np.zeros(self.dof)
        
        # 2. 只填充我们控制的关节
        # input_values 的顺序必须对应 self.controlled_names
        for i, val in enumerate(input_values):
            sapien_idx = self.controlled_indices[i]
            if sapien_idx != -1:
                full_qpos[sapien_idx] = val
                
        # 3. 其他关节 (如夹爪) 保持为 0
        return full_qpos

    def compute_fk_sapien_links(self, full_qpos, link_idx):
        # 此时传入的是全长 qpos
        fk = self.robot_model.compute_forward_kinematics(full_qpos)
        link_pose_ls = []
        for i in link_idx:
            link_pose_ls.append(self.robot_model.get_link_pose(i).to_transformation_matrix())
        return link_pose_ls

    def compute_robot_pcd(self, full_qpos, link_names=None, num_pts=None, pcd_name=None):
        fk = self.robot_model.compute_forward_kinematics(full_qpos)
        if link_names is None:
            link_names = self.meshes.keys()
        
        if num_pts is None:
            num_pts = [500] * len(link_names)
            
        link_idx_ls = []
        for link_name in link_names:
            found = False
            for link_idx, link in enumerate(self.sapien_robot.get_links()):
                if link.name == link_name:
                    link_idx_ls.append(link_idx)
                    found = True
                    break
        
        link_pose_ls = np.stack([self.robot_model.get_link_pose(
            link_idx).to_transformation_matrix() for link_idx in link_idx_ls])
            
        meshes_ls = [self.meshes[link_name] for link_name in link_names]
        offsets_ls = [self.offsets[link_name] for link_name in link_names]
        scales_ls = [self.scales[link_name] for link_name in link_names]
        
        pcd = self._mesh_poses_to_pc(poses=link_pose_ls, meshes=meshes_ls,
                                     offsets=offsets_ls, num_pts=num_pts, scales=scales_ls, pcd_name=pcd_name)
        return pcd

    def _mesh_poses_to_pc(self, poses, meshes, offsets, num_pts, scales, pcd_name=None):
        # ... (保持原样) ...
        try:
            assert poses.shape[0] == len(meshes)
        except:
            raise RuntimeError('poses and meshes must have the same length')

        N = poses.shape[0]
        all_pc = []
        for index in range(N):
            mat = poses[index]
            if pcd_name is None or pcd_name not in self.pcd_dict or len(self.pcd_dict[pcd_name]) <= index:
                mesh = copy.deepcopy(meshes[index])
                mesh.scale(scales[index], center=np.array([0, 0, 0]))
                sampled_cloud = mesh.sample_points_poisson_disk(number_of_points=num_pts[index])
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
        
        if len(all_pc) > 0:
            all_pc = np.concatenate(all_pc, axis=0)
        else:
            all_pc = np.array([])
        return all_pc

    def get_ee_pcs_through_joint(self, robot_joint_name, robot_joint_state, num_claw_pts=[150, 150], return_o3d=False):
        # 1. 整理输入数据，确保也是按照 controlled_names 的顺序
        input_dict = dict(zip(robot_joint_name, robot_joint_state))
        
        sorted_values = []
        for name in self.controlled_names:
            # 如果输入里没有某个关节，默认给0
            sorted_values.append(input_dict.get(name, 0.0))
            
        # 2. 转换为 Sapien 全长数组 (N维)
        full_qpos = self.get_full_qpos(sorted_values)

        if not self.init_flag:
            left_claw_pcd, right_claw_pcd = self.init_robot_pcd(full_qpos, num_claw_pts)
        else:
            left_claw_pcd, right_claw_pcd = self.update_robot_pcd(full_qpos)

        left_claw_pcd = left_claw_pcd.tolist()
        right_claw_pcd = right_claw_pcd.tolist()
        claw_pcd = left_claw_pcd + right_claw_pcd
        claw_pcd_o3d = np2o3d(np.array(claw_pcd))
        
        if return_o3d:
            return claw_pcd_o3d      
        return claw_pcd

    def init_robot_pcd(self, full_qpos, num_claw_pts):
        left_claw_pcd = self.compute_robot_pcd(full_qpos, link_names=[self.ee_names[0]], num_pts=[num_claw_pts[0]], pcd_name='left')
        right_claw_pcd = self.compute_robot_pcd(full_qpos, link_names=[self.ee_names[1]], num_pts=[num_claw_pts[1]], pcd_name='right')
        
        self.left_hand_pcd = left_claw_pcd
        self.right_hand_pcd = right_claw_pcd
        
        self.init_qpos = full_qpos # 这里的 init_qpos 也是全长的
        self.init_pose = self.compute_fk_sapien_links(full_qpos, self.ee_ids)
        self.init_flag = True
        
        return left_claw_pcd, right_claw_pcd

    def update_robot_pcd(self, full_qpos):
        curr_pose = self.compute_fk_sapien_links(full_qpos, self.ee_ids)
        
        left_relative_transform = np.array(curr_pose[0]) @ np.linalg.pinv(self.init_pose[0])
        right_relative_transform = np.array(curr_pose[1]) @ np.linalg.pinv(self.init_pose[1])

        augment_left_pcd = np.concatenate(
            [np.array(self.left_hand_pcd), np.ones((self.left_hand_pcd.shape[0], 1))], axis=1
        )
        augment_right_pcd = np.concatenate(
            [np.array(self.right_hand_pcd), np.ones((self.right_hand_pcd.shape[0], 1))], axis=1
        )

        left_transformed_pcd = (left_relative_transform @ augment_left_pcd.T).T[:, :3]
        right_transformed_pcd = (right_relative_transform @ augment_right_pcd.T).T[:, :3]
        
        return left_transformed_pcd, right_transformed_pcd