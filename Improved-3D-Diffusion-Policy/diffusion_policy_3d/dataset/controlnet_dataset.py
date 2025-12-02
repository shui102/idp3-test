# rm65_dataloader_stage.py
from typing import Dict, Optional
import torch
import numpy as np
import copy
from termcolor import cprint
from diffusion_policy_3d.common.pytorch_util import dict_apply
from diffusion_policy_3d.common.replay_buffer import ReplayBuffer
from diffusion_policy_3d.common.sampler import (SequenceSampler, get_val_mask, downsample_mask)
from diffusion_policy_3d.model.common.normalizer import (
    LinearNormalizer, SingleFieldLinearNormalizer
)
from diffusion_policy_3d.dataset.base_dataset import BaseDataset
import diffusion_policy_3d.model.vision_3d.point_process as point_process

SELECTED_INDICES = [i for i in range(14)]

class RM65_Dataset3D_Stage(BaseDataset):
    """
    两阶段数据加载器:
      - stage='unet': 输出 4096x6 点云
      - stage='controlnet': 输出控制点云 1024x3
    """

    def __init__(self,
                 zarr_path,
                 horizon=1,
                 pad_before=0,
                 pad_after=0,
                 seed=42,
                 val_ratio=0.0,
                 max_train_episodes=None,
                 task_name=None,
                 stage: str = "unet",
                 num_points_unet=4096,
                 num_points_ctrl=1024,
                 max_duration: Optional[int] = None): # [新增参数] 限制最大帧数
        super().__init__()

        cprint(f"[RM65_Dataset3D_Stage] Loading from {zarr_path}, stage={stage}", "green")

        self.task_name = task_name
        self.stage = stage
        self.num_points_unet = num_points_unet
        self.num_points_ctrl = num_points_ctrl
        self.max_duration = max_duration # 保存参数

        buffer_keys = ['state', 'point_cloud']
        if stage == "controlnet":
            buffer_keys.append('control_point_cloud')
        self.replay_buffer = ReplayBuffer.copy_from_path(zarr_path, keys=buffer_keys)

        # === train/val mask ===
        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes,
            val_ratio=val_ratio,
            seed=seed
        )
        train_mask = ~val_mask
        train_mask = downsample_mask(train_mask, max_n=max_train_episodes, seed=seed)

        # === sampler ===
        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=train_mask
        )
        
        # [新增逻辑] 如果设定了 max_duration，过滤掉超过该帧数的索引
        self._apply_max_duration_filter(self.sampler, max_duration)

        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after

    def _apply_max_duration_filter(self, sampler, max_duration):
            """
            内部辅助函数：修改 sampler.indices，剔除每条轨迹 max_duration 之后的帧。
            兼容 1D indices 和 2D cached indices。
            """
            if max_duration is None:
                return

            cprint(f"[Dataset] Filtering episodes to first {max_duration} frames...", "yellow")
            
            # 1. 获取所有 Episode 的起止位置，构建 Mask
            episode_ends = self.replay_buffer.episode_ends[:]
            episode_starts = np.concatenate([[0], episode_ends[:-1]])
            
            n_steps = self.replay_buffer.n_steps
            valid_step_mask = np.zeros(n_steps, dtype=bool)
            
            for start, end in zip(episode_starts, episode_ends):
                # 标记每条轨迹的前 max_duration 帧为 True
                cutoff = min(end, start + max_duration)
                valid_step_mask[start:cutoff] = True

            # ============================================================
            # [核心修复] 
            # 1. 自动扩展 mask 防止 IndexError
            # 2. 区分 1D 和 2D indices 防止 TypeError
            # ============================================================
            
            indices = sampler.indices
            
            # 处理可能的 indices 维度问题
            if indices.ndim == 1:
                # 标准情况：indices 是 1D 的起始索引
                query_indices = indices
            elif indices.ndim == 2:
                # 优化情况：indices 是 (N, 4) 等形状，第一列通常是 buffer_start_idx
                query_indices = indices[:, 0] 
            else:
                raise ValueError(f"Unsupported sampler.indices shape: {indices.shape}")

            # 检查是否越界并扩展 mask
            if len(query_indices) > 0:
                max_idx = np.max(query_indices)
                if max_idx >= len(valid_step_mask):
                    pad_len = max_idx - len(valid_step_mask) + 1
                    valid_step_mask = np.concatenate([valid_step_mask, np.zeros(pad_len, dtype=bool)])
            
            # 生成基于 query_indices 的布尔掩码
            # query_indices 中的值对应的 mask 位为 True 时，保留该样本
            keep_mask = valid_step_mask[query_indices]
            
            # 应用掩码，按行过滤
            original_len = len(sampler.indices)
            sampler.indices = sampler.indices[keep_mask]
            
            cprint(f"[Dataset] Filter done. Indices count: {original_len} -> {len(sampler.indices)}", "yellow")
            
    def get_validation_dataset(self):
        """返回验证集"""
        val_set = copy.copy(self)
        # 重新创建 Sampler
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=~self.train_mask
        )
        # [关键修改] 验证集也需要应用同样的帧数截断逻辑
        val_set._apply_max_duration_filter(val_set.sampler, self.max_duration)
        
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, mode='limits', **kwargs):
        """根据阶段返回对应字段的 normalizer"""
        # 注意：这里的数据统计最好也只统计前 400 帧的数据，否则 Normalizer 会受到后面数据的影响
        # 我们可以利用 self.sampler.indices 来筛选数据
        
        if self.max_duration is None:
            action = self.replay_buffer['state'][..., SELECTED_INDICES]
            agent_pos = self.replay_buffer['state'][..., SELECTED_INDICES]
        else:
            # 如果截断了数据，我们只用截断后的数据来 fit normalizer
            # 获取所有合法的 indices
            indices = self.sampler.indices
            # 由于 indices 是一维的，直接取 ReplayBuffer 可能会比较慢/占内存，
            # 简单起见，如果数据量不大，可以直接索引。如果数据量巨大，可以沿用上面的 mask 逻辑
            # 这里为了代码简洁和准确性，我们使用 mask 逻辑重新切片
            
            episode_ends = self.replay_buffer.episode_ends[:]
            episode_starts = np.concatenate([[0], episode_ends[:-1]])
            mask = np.zeros(self.replay_buffer.n_steps, dtype=bool)
            for start, end in zip(episode_starts, episode_ends):
                cutoff = min(end, start + self.max_duration)
                mask[start:cutoff] = True
            
            action = self.replay_buffer['state'][mask][..., SELECTED_INDICES]
            agent_pos = self.replay_buffer['state'][mask][..., SELECTED_INDICES]

        data = {'action': action, 'agent_pos': agent_pos}
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)

        # 点云部分恒等映射
        normalizer['point_cloud'] = SingleFieldLinearNormalizer.create_identity()
        if self.stage == "controlnet":
            normalizer['control_point_cloud'] = SingleFieldLinearNormalizer.create_identity()

        return normalizer

    def __len__(self):
        return len(self.sampler)

    def _sample_to_data(self, sample):
        # ... (保持不变) ...
        agent_pos = sample['state'][..., SELECTED_INDICES].astype(np.float32)
        action = sample['state'][..., SELECTED_INDICES].astype(np.float32)
        point_cloud = sample['point_cloud'].astype(np.float32)

        if self.stage == "unet":
            pc_main = point_process.uniform_sampling_numpy(point_cloud, self.num_points_unet)
            data = {
                'obs': {
                    'agent_pos': agent_pos,
                    'point_cloud': pc_main
                },
                'action': action
            }

        elif self.stage == "controlnet":
            control = sample['control_point_cloud'].astype(np.float32)
            pc_main = point_process.uniform_sampling_numpy(point_cloud, self.num_points_unet)
            pc_ctrl = point_process.uniform_sampling_numpy(control, self.num_points_ctrl)

            data = {
                'obs': {
                    'agent_pos': agent_pos,
                    'point_cloud': pc_main
                },
                'action': action,
                'control': {
                    'control_point_cloud': pc_ctrl
                }
            }
        else:
            raise ValueError(f"Unknown stage: {self.stage}")

        return data
    
    # ... (__getitem__ 保持不变) ...
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        torch_data = dict_apply(data, torch.from_numpy)
        return torch_data

if __name__ == "__main__":
    # 测试代码
    dataset_ctrl = RM65_Dataset3D_Stage(
        zarr_path='/media/shui/Lexar/obs_temp/fake_control_point_cloud.zarr',
        stage='controlnet',
        max_duration=400  # <--- 在这里传入 400
    )
    
    print(f"Dataset length with max_duration=400: {len(dataset_ctrl)}")
    
    batch2 = dataset_ctrl[0]
    print("\n[Stage2: ControlNet]")
    print("control_point_cloud:", batch2['control']['control_point_cloud'].shape)

    valset = dataset_ctrl.get_validation_dataset()
    print(f"Val Dataset length: {len(valset)}")