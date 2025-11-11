# rm65_dataloader_stage.py
from typing import Dict
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
                 num_points_ctrl=1024):
        super().__init__()

        cprint(f"[RM65_Dataset3D_Stage] Loading from {zarr_path}, stage={stage}", "green")

        self.task_name = task_name
        self.stage = stage
        self.num_points_unet = num_points_unet
        self.num_points_ctrl = num_points_ctrl

        buffer_keys = ['state', 'action', 'point_cloud']
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
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after

    def get_validation_dataset(self):
        """返回验证集"""
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

    def get_normalizer(self, mode='limits', **kwargs):
        """根据阶段返回对应字段的 normalizer"""
        action = self.replay_buffer['action'][..., :10]
        agent_pos = self.replay_buffer['state'][..., :10]

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
        """采样转换为模型输入格式"""
        agent_pos = sample['state'][..., :10].astype(np.float32)
        action = sample['action'][..., :10].astype(np.float32)
        point_cloud = sample['point_cloud'].astype(np.float32)

        # Stage 区分逻辑
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
            # cprint(f"sample:{sample.keys()}","red")
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

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        torch_data = dict_apply(data, torch.from_numpy)
        return torch_data

if __name__ == "__main__":
    # dataset_unet = RM65_Dataset3D_Stage(
    #     zarr_path='/home/shui/idp3_test/Improved-3D-Diffusion-Policy/data/dataset_11092057_processed.zarr',
    #     stage='unet'
    # )
    # dataset_unet.get_normalizer()
    # batch = dataset_unet[0]
    # print("\n[Stage1: UNet]")
    # print("agent_pos:", batch['obs']['agent_pos'].shape)
    # print("point_cloud:", batch['obs']['point_cloud'].shape)
    # print("action:", batch['action'].shape)

    dataset_ctrl = RM65_Dataset3D_Stage(
        zarr_path='/media/shui/Lexar/obs_temp/fake_control_point_cloud.zarr',
        stage='controlnet'
    )
    batch2 = dataset_ctrl[0]
    print("\n[Stage2: ControlNet]")
    print("control_point_cloud:", batch2['control']['control_point_cloud'].shape)

    valset = dataset_ctrl.get_validation_dataset()
    normalizer = dataset_ctrl.get_normalizer()
    print("val length:", len(valset))
    print("normalizer keys:", list(normalizer.params_dict.keys()))
