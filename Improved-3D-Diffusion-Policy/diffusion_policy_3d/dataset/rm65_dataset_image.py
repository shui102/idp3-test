from typing import Dict
import torch
import numpy as np
import copy
from termcolor import cprint
from diffusion_policy_3d.common.pytorch_util import dict_apply
from diffusion_policy_3d.common.replay_buffer import ReplayBuffer
from diffusion_policy_3d.common.sampler import (SequenceSampler, get_val_mask, downsample_mask)
from diffusion_policy_3d.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from diffusion_policy_3d.dataset.base_dataset import BaseDataset

# 保留RM65特有的前14维截取逻辑
SELECTED_INDICES = [i for i in range(14)]

class RM65DatasetImage(BaseDataset):
    """
    RM65机器人的2D图片版数据集（适配Diffusion Policy 2D版）
    核心：加载RGB图片+低维状态，保留前14维有效特征，适配DP的归一化/采样逻辑
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
            use_img=True,  # 启用RGB图片（DP核心）
            use_depth=False, # 可选：启用深度图
            ):
        super().__init__()
        cprint(f'Loading RM65DatasetImage from {zarr_path}', 'green')
        self.task_name = task_name
        self.use_img = use_img
        self.use_depth = use_depth

        # ===================== 关键修改1：替换点云为图片字段 =====================
        # 移除point_cloud，加入img/depth（按需）
        buffer_keys = ['state', 'action']  # RM65需显式加载action（区别于3D版）
        if self.use_img:
            buffer_keys.append('img')
        if self.use_depth:
            buffer_keys.append('depth')

        # 加载Zarr数据集
        self.replay_buffer = ReplayBuffer.copy_from_path(zarr_path, keys=buffer_keys)

        # ===================== 保留3D版的采样逻辑 =====================
        # 划分训练/验证集mask
        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes, 
            val_ratio=val_ratio,
            seed=seed)
        train_mask = ~val_mask
        train_mask = downsample_mask(
            mask=train_mask, 
            max_n=max_train_episodes, 
            seed=seed)
        
        # 序列采样器（适配DP的horizon/n_obs_steps/n_action_steps）
        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=horizon,
            pad_before=pad_before, 
            pad_after=pad_after,
            episode_mask=train_mask)
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after

    def get_validation_dataset(self):
        """保留验证集逻辑（与3D版一致）"""
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=self.horizon,
            pad_before=self.pad_before, 
            pad_after=self.pad_after,
            episode_mask=~self.train_mask
            )
        val_set.train_mask = ~self.train_mask
        return val_set

    # ===================== 关键修改2：适配DP的归一化逻辑 =====================
    def get_normalizer(self, mode='limits', **kwargs):
        # 只截取前14维（RM65特有）
        action_14d = self.replay_buffer['action'][..., SELECTED_INDICES]
        agent_pos_14d = self.replay_buffer['state'][..., SELECTED_INDICES]

        # 动作/低维状态归一化（DP需要）
        data = {
            'action': action_14d,
            'agent_pos': agent_pos_14d,
        }
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)

        # 图片/深度图恒等映射（DP中图片会手动归一化到0-1）
        if self.use_img:
            normalizer['image'] = SingleFieldLinearNormalizer.create_identity()
        if self.use_depth:
            normalizer['depth'] = SingleFieldLinearNormalizer.create_identity()

        # RM65特有：防止标准差过小导致训练不稳定
        for k in ['action', 'agent_pos']:
            if (k in normalizer.params_dict and 
                hasattr(normalizer[k], 'params_dict') and
                'input_stats' in normalizer[k].params_dict and
                'std' in normalizer[k].params_dict.input_stats):
                std_param = normalizer[k].params_dict.input_stats.std
                with torch.no_grad():
                    std_param.data.clamp_(min=1e-1) 

        return normalizer

    def __len__(self) -> int:
        return len(self.sampler)

    # ===================== 关键修改3：数据格式适配DP =====================
    def _sample_to_data(self, sample):
        # 1. 截取前14维低维状态/动作（RM65特有）
        agent_pos = sample['state'][..., SELECTED_INDICES].astype(np.float32)
        action = sample['action'][..., SELECTED_INDICES].astype(np.float32)

        # 2. 组装DP需要的输出格式（obs包含image/agent_pos，action为14维）
        data = {
            'obs': {
                'agent_pos': agent_pos,
            },
            'action': action
        }

        # 3. 加入图片/深度图（DP的核心输入）
        if self.use_img:
            # 图片维度：(horizon, 224, 224, 3) → DP中会转置为(horizon, 3, 224, 224)
            image = sample['img'].astype(np.float32)
            data['obs']['image'] = image
        if self.use_depth:
            depth = sample['depth'].astype(np.float32)
            data['obs']['depth'] = depth

        return data
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """最终输出DP需要的Tensor格式"""
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        # 转numpy为torch.Tensor
        to_torch_function = lambda x: torch.from_numpy(x) if isinstance(x, np.ndarray) else x
        torch_data = dict_apply(data, to_torch_function)
        return torch_data


# 测试代码（验证数据集加载是否正确）
if __name__ == "__main__":
    # 替换为你的RM65 2D数据集路径
    dataset = RM65DatasetImage(zarr_path='/your/rm65_2d_dataset.zarr', horizon=18)
    sample = dataset[0]
    
    # 打印维度（验证是否符合DP要求）
    print("agent_pos shape:", sample['obs']['agent_pos'].shape)  # [horizon, 14]
    print("action shape:", sample['action'].shape)                # [horizon, 14]
    print("image shape:", sample['obs']['image'].shape)           # [horizon, 224, 224, 3]