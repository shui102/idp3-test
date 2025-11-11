# fake_dataloader.py
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Dict
from termcolor import cprint
from diffusion_policy_3d.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer


class Fake3DDataset(Dataset):
    """
    用于伪造训练数据的Dataset
    支持两种模式:
        stage = "unet"       -> 输出4096×6点云
        stage = "controlnet" -> 输出1024×3控制点云
    """
    def __init__(self,
                 num_samples: int = 100,
                 horizon: int = 16,
                 n_obs_steps: int = 2,
                 stage: str = "unet",
                 device: str = "cpu"):
        super().__init__()
        self.num_samples = num_samples
        self.horizon = horizon
        self.n_obs_steps = n_obs_steps
        self.stage = stage
        self.device = device

        cprint(f"[Fake3DDataset] Initialized with stage={stage}, samples={num_samples}", "green")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        obs = {
            "agent_pos": torch.randn(self.n_obs_steps, 10),
            "point_cloud": torch.randn(self.n_obs_steps, 4096, 6)
        }
        action = torch.randn(self.horizon, 30)

        if self.stage == "controlnet":
            control = {
                "control_point_cloud": torch.randn(self.n_obs_steps, 1024, 3)
            }
            return {"obs": obs, "action": action, "control": control}
        else:
            return {"obs": obs, "action": action}

    # =========================================================
    # 为 workspace 兼容：提供 get_validation_dataset / get_normalizer
    # =========================================================
    def get_validation_dataset(self):
        """返回验证集（伪数据直接复制自身）"""
        val_set = Fake3DDataset(
            num_samples=self.num_samples // 5,
            horizon=self.horizon,
            n_obs_steps=self.n_obs_steps,
            stage=self.stage,
            device=self.device
        )
        return val_set

    def get_normalizer(self):
        """返回一个恒等 normalizer（不做归一化）"""
        normalizer = LinearNormalizer()

        # 动作
        data = {"action": torch.zeros(1, 10)}
        normalizer.fit(data=data, last_n_dims=1, mode="limits")

        # 其他字段（恒等映射）
        normalizer["point_cloud"] = SingleFieldLinearNormalizer.create_identity()
        normalizer["agent_pos"] = SingleFieldLinearNormalizer.create_identity()
        normalizer["agent_rot"] = SingleFieldLinearNormalizer.create_identity()

        # controlnet 模式额外字段
        if self.stage == "controlnet":
            normalizer["control_point_cloud"] = SingleFieldLinearNormalizer.create_identity()

        return normalizer


def get_fake_dataloader(batch_size=2,
                        num_samples=100,
                        horizon=16,
                        n_obs_steps=2,
                        stage="unet",
                        device="cpu"):
    dataset = Fake3DDataset(num_samples=num_samples,
                            horizon=horizon,
                            n_obs_steps=n_obs_steps,
                            stage=stage,
                            device=device)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader


# ------------------------- #
# 测试用例
# ------------------------- #
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dl_unet = get_fake_dataloader(stage="unet", device=device)
    batch = next(iter(dl_unet))
    print("\n[Stage1] Fake UNet batch keys:", batch.keys())
    print("agent_pos:", batch["obs"]["agent_pos"].shape)
    print("point_cloud:", batch["obs"]["point_cloud"].shape)
    print("action:", batch["action"].shape)

    dl_ctrl = get_fake_dataloader(stage="controlnet", device=device)
    batch2 = next(iter(dl_ctrl))
    print("\n[Stage2] Fake ControlNet batch keys:", batch2.keys())
    print("control_point_cloud:", batch2["control"]["control_point_cloud"].shape)

    dataset = Fake3DDataset(stage="controlnet")
    val_dataset = dataset.get_validation_dataset()
    normalizer = dataset.get_normalizer()
    print("val length:", len(val_dataset))
    print("normalizer keys:", list(normalizer.params_dict.keys()))
