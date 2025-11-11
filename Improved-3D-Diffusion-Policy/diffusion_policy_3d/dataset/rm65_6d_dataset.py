from typing import Dict
import torch
import numpy as np
import copy
from diffusion_policy_3d.common.pytorch_util import dict_apply
from diffusion_policy_3d.common.replay_buffer import ReplayBuffer
from diffusion_policy_3d.common.sampler import (SequenceSampler, get_val_mask, downsample_mask)
from diffusion_policy_3d.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer, StringNormalizer
from diffusion_policy_3d.dataset.base_dataset import BaseDataset
import diffusion_policy_3d.model.vision_3d.point_process as point_process
from termcolor import cprint
# import IPython
# e = IPython.embed

class RM65_Dataset3D(BaseDataset):
    def __init__(self,
            zarr_path, 
            horizon=1,
            pad_before=0,
            pad_after=0,
            seed=42,
            val_ratio=0.0,
            max_train_episodes=None,
            task_name=None,
            num_points=4096,
            use_6d_loss=True
            ):
        super().__init__()
        cprint(f'Loading GR1DexDataset from {zarr_path}', 'green')
        self.task_name = task_name
        self.num_points = num_points

        buffer_keys = ['state', 'action']
        buffer_keys.append('point_cloud')

        self.replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path, keys=buffer_keys)

        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes, 
            val_ratio=val_ratio,
            seed=seed)
        train_mask = ~val_mask
        train_mask = downsample_mask(
            mask=train_mask, 
            max_n=max_train_episodes, 
            seed=seed)

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
        self.use_6d_loss=use_6d_loss

    def get_validation_dataset(self):
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
        # action_7d = self.replay_buffer['action'][..., :10]
        # agent_pos_7d = self.replay_buffer['state'][..., :10]  # ✅ 改名保持一致

        # data = {
        #     'action': action_7d,
        #     'agent_pos': agent_pos_7d,  # ✅ 改成 agent_pos
        # }

        # normalizer = LinearNormalizer()
        # normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)

        # normalizer['point_cloud'] = SingleFieldLinearNormalizer.create_identity()
        # return normalizer
        actions = self.replay_buffer['action'][..., :10]   # [E, T, 10]
        states = self.replay_buffer['state'][..., :10]     # [E, T, 10]
        D = actions.shape[-1]
        print(f"[Normalizer] Fitting LinearNormalizer with dim={D} (use_6d_loss={self.use_6d_loss})")

        # ---- Step 2: 拟合 ----
        data = {
            'action': actions,
            'agent_pos': states,
        }

        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)

        if getattr(self, "use_6d_loss", False):
            if D >= 9:
                pos_idx = slice(0, 3)
                rot_idx = slice(3, 9)
                grip_idx = slice(9, 10)

                for key in ['action', 'agent_pos']:
                    field = normalizer[key]
                    scale = field.params_dict['scale']
                    offset = field.params_dict['offset']

                    scale[..., rot_idx] = 1.0
                    offset[..., rot_idx] = 0.0
                    scale[..., grip_idx] = 1.0
                    offset[..., grip_idx] = 0.0

                print("[Normalizer] use_6d_loss=True → rot6d维度 [3:9] 设置为 identity (scale=1, offset=0)")
            else:
                print("[Normalizer] use_6d_loss=True 但维度<9，跳过rot identity设定")

        normalizer['point_cloud'] = SingleFieldLinearNormalizer.create_identity()
        return normalizer


    def __len__(self) -> int:
        return len(self.sampler)

    # =========================================================
    # ✅ 修改点2: 返回数据时只保留前7维
    # =========================================================
    def _sample_to_data(self, sample):
        # 截取前7维 agent_pos / action
        agent_pos = sample['state'][..., :10].astype(np.float32)
        action = sample['action'][..., :10].astype(np.float32)

        # 点云下采样
        point_cloud = sample['point_cloud'].astype(np.float32)
        point_cloud = point_process.uniform_sampling_numpy(point_cloud, self.num_points)

        data = {
            'obs': {
                'agent_pos': agent_pos,
                'point_cloud': point_cloud,
            },
            'action': action
        }
        return data

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        to_torch_function = lambda x: torch.from_numpy(x) if isinstance(x, np.ndarray) else x
        torch_data = dict_apply(data, to_torch_function)
        return torch_data


if __name__ == "__main__":
    dataset = RM65_Dataset3D(zarr_path='/extra/waylen/diffusion_policy/Improved-3D-Diffusion-Policy/data/training_data_example')
    sample = dataset[0]
    dataset.get_normalizer()