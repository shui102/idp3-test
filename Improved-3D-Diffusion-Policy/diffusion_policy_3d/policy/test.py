# -*- coding: utf-8 -*-
"""
DiffusionPointcloudControlPolicy
- 两阶段训练：Stage 1 训练 Unet（4096×6），Stage 2 冻结 Unet，训练 ControlNet（1024×3）
- 与原版 DiffusionPointcloudPolicy 对齐：接口、采样流程、loss 计算、mask、normalizer 等
- 差异点：
  * Stage1 的 global_cond 来自 obs_encoder_stage1(4096×6)
  * Stage2 的 global_cond 仍来自 obs_encoder_stage1(4096×6)，control_input 来自 control_encoder(1024×3)
"""

from typing import Dict, Optional
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from termcolor import cprint

from diffusion_policy_3d.policy.base_policy import BasePolicy
from diffusion_policy_3d.model.common.normalizer import LinearNormalizer
from diffusion_policy_3d.model.diffusion.mask_generator import LowdimMaskGenerator
from diffusion_policy_3d.common.pytorch_util import dict_apply
from diffusion_policy_3d.common.model_util import print_params

# 你的 UNet 与 ControlNet（上一条消息里给出的 ConditionalControlUnet1D）
from diffusion_policy_3d.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusion_policy_3d.model.diffusion.stateful_unet1d import ConditionalControlUnet1D

# iDP3 点云编码器（与原版一致）
from diffusion_policy_3d.model.vision_3d.pointnet_extractor import iDP3Encoder


class DiffusionPointcloudControlPolicy(BasePolicy):
    """
    两阶段：
      - train_stage='unet': 训练 ConditionalUnet1D，点云输入为 4096×6（经 obs_encoder_stage1）
      - train_stage='controlnet': 训练 ConditionalControlUnet1D 的 Control 分支，点云输入为 1024×3（经 control_encoder）
        同时加载并冻结已预训练好的 Unet 主干；global_cond 仍由 4096×6 的 obs_encoder_stage1 提供
    """
    def __init__(self,
                 shape_meta: dict,
                 noise_scheduler: DDPMScheduler,
                 horizon: int,
                 n_action_steps: int,
                 n_obs_steps: int,
                 num_inference_steps: Optional[int] = None,
                 obs_as_global_cond: bool = True,
                 diffusion_step_embed_dim: int = 256,
                 down_dims=(256, 512, 1024),
                 kernel_size: int = 5,
                 n_groups: int = 8,
                 condition_type: str = "film",
                 use_down_condition: bool = True,
                 use_mid_condition: bool = True,
                 use_up_condition: bool = True,
                 # 训练阶段开关：'unet' or 'controlnet'
                 train_stage: str = "unet",
                 # 预训练 Unet 权重（Stage2 时必须提供）
                 pretrained_unet_path: Optional[str] = None,
                 # 编码器配置
                 stage1_pointcloud_encoder_cfg=None,   # 用于 4096×6
                 stage2_control_encoder_cfg=None,      # 用于 1024×3
                 # 其他
                 **kwargs):
        super().__init__()

        assert train_stage in ["unet", "controlnet"], "train_stage 只能是 'unet' 或 'controlnet'"

        self.condition_type = condition_type
        self.horizon = horizon
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_global_cond = obs_as_global_cond
        self.train_stage = train_stage
        self.kwargs = kwargs

        # -------- parse shape_meta 与动作维度 --------
        action_shape = shape_meta["action"]["shape"]
        if len(action_shape) == 1:
            self.action_dim = action_shape[0]
        elif len(action_shape) == 2:
            self.action_dim = action_shape[0] * action_shape[1]
        else:
            raise NotImplementedError(f"Unsupported action shape {action_shape}")

        obs_shape_meta = shape_meta["obs"]
        obs_shapes = dict_apply(obs_shape_meta, lambda x: x["shape"])

        # -------- 构建 Stage1 的点云编码器（4096×6）--------
        # 与原版保持一致：它负责生成 global_cond
        self.obs_encoder_stage1 = iDP3Encoder(
            observation_space=obs_shapes,                    # 这里通常包含 point_cloud: (To, 4096, 6) 以及其他低维 obs
            pointcloud_encoder_cfg=stage1_pointcloud_encoder_cfg,
            use_pc_color=True,                               # 4096×6 包含 RGB
            pointnet_type="multi_stage_pointnet"
        )
        self.obs_feature_dim = self.obs_encoder_stage1.output_shape()

        # -------- 构建 Stage2 的控制编码器（1024×3）--------
        # 只在控制阶段需要，但为了 forward 统一也初始化
        # 这里我们让它的 observation_space 只包含控制点云的形状
        self.control_encoder = iDP3Encoder(
            observation_space={
                "agent_pos": {"shape": (1,)},            # 虚拟一个 1 维的状态
                "point_cloud": {"shape": (self.n_obs_steps, 1024, 3)}
            },
            pointcloud_encoder_cfg=stage2_control_encoder_cfg,
            use_pc_color=False,                              # 1024×3 无颜色
            pointnet_type="pointnet"
        )
        self.control_feature_dim = self.control_encoder.output_shape()

        # -------- 创建扩散模型（按阶段）--------
        if self.train_stage == "unet":
            # 原版：如果 obs 作为 global_cond，则模型 input_dim=action_dim，global_cond_dim=obs_feature_dim*n_obs_steps（FiLM/ADD）
            # 若为 cross_attention 模式，则 global_cond_dim=obs_feature_dim（序列）
            input_dim = self.action_dim
            global_cond_dim = None
            if self.obs_as_global_cond:
                if "cross_attention" in self.condition_type:
                    global_cond_dim = self.obs_feature_dim
                else:
                    global_cond_dim = self.obs_feature_dim * self.n_obs_steps
            else:
                # 走 impainting：模型输入包括动作与 obs_feat 拼接
                input_dim = self.action_dim + self.obs_feature_dim

            self.model = ConditionalUnet1D(
                input_dim=input_dim,
                local_cond_dim=None,
                global_cond_dim=global_cond_dim,
                diffusion_step_embed_dim=diffusion_step_embed_dim,
                down_dims=down_dims,
                kernel_size=kernel_size,
                n_groups=n_groups,
                condition_type=condition_type,
                use_down_condition=use_down_condition,
                use_mid_condition=use_mid_condition,
                use_up_condition=use_up_condition,
            )

        else:
            # controlnet 阶段：主干 Unet 输入维保持与 Stage1 一致（input_dim=action_dim；obs 作为 global_cond）
            # control_input 由 control_encoder 提供（feature 维度），传入 ConditionalControlUnet1D 的 control_dim
            input_dim = self.action_dim
            if self.obs_as_global_cond:
                if "cross_attention" in self.condition_type:
                    global_cond_dim = self.obs_feature_dim
                else:
                    global_cond_dim = self.obs_feature_dim * self.n_obs_steps
            else:
                # 这里不建议 control 阶段再走 impainting，以保持与预训练的一致性
                raise NotImplementedError("建议在 ControlNet 阶段保持 obs_as_global_cond=True")

            self.model = ConditionalControlUnet1D(
                input_dim=input_dim,
                control_dim=self.control_feature_dim,      # 注意：control_dim 对齐 control_encoder 的输出维
                local_cond_dim=None,
                global_cond_dim=global_cond_dim,
                diffusion_step_embed_dim=diffusion_step_embed_dim,
                down_dims=down_dims,
                kernel_size=kernel_size,
                n_groups=n_groups,
                condition_type=condition_type,
                use_down_condition=use_down_condition,
                use_mid_condition=use_mid_condition,
                use_up_condition=use_up_condition,
            )

            # 加载并冻结主干 Unet
            assert pretrained_unet_path is not None, \
                "ControlNet 阶段需要提供 pretrained_unet_path（上一阶段训练好的 Unet 权重）"
            cprint(f"[ControlNet] Loading pretrained UNet from: {pretrained_unet_path}", "yellow")
            unet_state = torch.load(pretrained_unet_path, map_location="cpu")
            self.model.unet.load_state_dict(unet_state, strict=True)
            for p in self.model.unet.parameters():
                p.requires_grad = False
            cprint("[ControlNet] UNet is frozen. Only training ControlNet branch.", "yellow")

        # -------- 噪声调度器/掩码/归一化器 等（与原版一致）--------
        self.noise_scheduler = noise_scheduler
        self.mask_generator = LowdimMaskGenerator(
            action_dim=self.action_dim,
            obs_dim=0 if obs_as_global_cond else self.obs_feature_dim,
            max_n_obs_steps=self.n_obs_steps,
            fix_obs_steps=True,
            action_visible=False
        )
        self.normalizer = LinearNormalizer()

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps

        print_params(self)

    # ====================== 公共：前向（推理版） ======================
    def forward(self, obs_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        与原版保持一致接口：返回未来 n_action_steps 的动作 (B, n_action_steps, action_dim)
        - 需要 obs_dict 包含：
          * Stage1 & Stage2 都要：'point_cloud' (B, To, 4096, 6) 及其它低维 obs（若你设计了）
          * Stage2 还需：'control_point_cloud' (B, To, 1024, 3)
        """
        # 归一化
        nobs = self.normalizer.normalize(obs_dict)

        value = next(iter(nobs.values()))
        B, To = value.shape[:2]
        T = self.horizon
        Da = self.action_dim
        To = self.n_obs_steps

        device = self.device
        dtype = self.dtype

        local_cond = None
        global_cond = None

        # ======= 全局条件：一律来自 4096×6 的 obs_encoder_stage1（两阶段保持一致）=======
        this_nobs = dict_apply(nobs, lambda x: x[:, :To, ...].reshape(-1, *x.shape[2:]))
        nobs_features = self.obs_encoder_stage1(this_nobs)  # (B*To, Df)
        feat_dim = nobs_features.shape[-1]
        if self.obs_as_global_cond:
            if "cross_attention" in self.condition_type:
                global_cond = nobs_features.reshape(B, To, feat_dim)
            else:
                global_cond = nobs_features.reshape(B, To * feat_dim)
        else:
            # 走 impainting（与原版一致，通常不用于本任务）
            raise NotImplementedError("建议统一使用 obs_as_global_cond=True")

        # ======= control feature：仅在 ControlNet 阶段需要（来自 1024×3）=======
        control_feat = None
        if self.train_stage == "controlnet":
            assert "control_point_cloud" in obs_dict, \
                "ControlNet 阶段需要 obs_dict['control_point_cloud']，形状 (B, To, 1024, 3)"
            ctrl_pc = obs_dict["control_point_cloud"]
            # 编码器期望 dict 输入
            ctrl_dict = {"point_cloud": ctrl_pc}
            ctrl_dict = dict_apply(ctrl_dict, lambda x: x[:, :To, ...].reshape(-1, *x.shape[2:]))
            control_feat = self.control_encoder(ctrl_dict)  # (B*To, Dc)
            # 这里我们将 control_feat 做平均聚合到 (B, Dc)，你也可以改成拼接/注意力等
            control_feat = control_feat.reshape(B, To, -1).mean(dim=1)  # (B, Dc)

        # ======= 条件数据（空动作轨迹）并采样 =======
        cond_data = torch.zeros((B, T, Da), device=device, dtype=dtype)
        cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)

        nsample = self.conditional_sample(
            cond_data,
            cond_mask,
            local_cond=local_cond,
            global_cond=global_cond,
            control_input=control_feat,
            **self.kwargs
        )

        # ======= 反归一化并裁剪到 n_action_steps =======
        naction_pred = nsample[..., :Da]
        action_pred = self.normalizer["action"].unnormalize(naction_pred)
        start = To - 1
        end = start + self.n_action_steps
        return action_pred[:, start:end]

    # ====================== 采样流程（与原版一致） ======================
    def conditional_sample(self,
                           condition_data,
                           condition_mask,
                           local_cond=None,
                           global_cond=None,
                           control_input=None,
                           generator=None,
                           **kwargs):
        model = self.model
        scheduler = self.noise_scheduler

        trajectory = torch.randn_like(condition_data)
        scheduler.set_timesteps(self.num_inference_steps)

        for t in scheduler.timesteps:
            trajectory[condition_mask] = condition_data[condition_mask]

            if self.train_stage == "controlnet":
                model_output = model(sample=trajectory,
                                     timestep=t,
                                     control_input=control_input,
                                     local_cond=local_cond,
                                     global_cond=global_cond)
            else:
                model_output = model(sample=trajectory,
                                     timestep=t,
                                     local_cond=local_cond,
                                     global_cond=global_cond)

            trajectory = scheduler.step(model_output, t, trajectory).prev_sample

        trajectory[condition_mask] = condition_data[condition_mask]
        return trajectory

    # ====================== 推理 API（与原版一致） ======================
    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # forward 中已经实现核心逻辑，这里与原版相同地返回 action 及完整 action_pred
        nobs = self.normalizer.normalize(obs_dict)
        value = next(iter(nobs.values()))
        B, To = value.shape[:2]
        T = self.horizon
        Da = self.action_dim
        To = self.n_obs_steps

        device = self.device
        dtype = self.dtype

        # ------ global cond from 4096×6 ------
        this_nobs = dict_apply(nobs, lambda x: x[:, :To, ...].reshape(-1, *x.shape[2:]))
        nobs_features = self.obs_encoder_stage1(this_nobs)
        feat_dim = nobs_features.shape[-1]
        if self.obs_as_global_cond:
            if "cross_attention" in self.condition_type:
                global_cond = nobs_features.reshape(B, To, feat_dim)
            else:
                global_cond = nobs_features.reshape(B, To * feat_dim)
        else:
            raise NotImplementedError("建议统一使用 obs_as_global_cond=True")

        # ------ control feat (only in controlnet stage) ------
        control_feat = None
        if self.train_stage == "controlnet":
            ctrl_pc = obs_dict["control_point_cloud"]  # (B, To, 1024, 3)
            ctrl_dict = {"point_cloud": ctrl_pc}
            ctrl_dict = dict_apply(ctrl_dict, lambda x: x[:, :To, ...].reshape(-1, *x.shape[2:]))
            control_feat_bt = self.control_encoder(ctrl_dict)  # (B*To, Dc)
            control_feat = control_feat_bt.reshape(B, To, -1).mean(dim=1)  # (B, Dc)

        cond_data = torch.zeros((B, T, Da), device=device, dtype=dtype)
        cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)

        nsample = self.conditional_sample(
            cond_data,
            cond_mask,
            local_cond=None,
            global_cond=global_cond,
            control_input=control_feat
        )

        naction_pred = nsample[..., :Da]
        action_pred = self.normalizer["action"].unnormalize(naction_pred)

        start = To - 1
        end = start + self.n_action_steps
        action = action_pred[:, start:end]

        return {
            "action": action,
            "action_pred": action_pred
        }

    # ====================== Normalizer ======================
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    # ====================== 训练 Loss ======================
    def compute_loss(self, batch):
        """
        - train_stage='unet': 训练 Unet（与原版相同的 loss 计算）
        - train_stage='controlnet': 冻结 Unet，仅训练 ControlNet（target 与 Stage1 一致）
        需要 batch 包含：
          * 'obs'：含 'point_cloud' (B, To, 4096, 6) 等
          * 'action'：(B, T, Da)
          * 若为 controlnet 阶段，还需 'control_obs'['point_cloud']：(B, To, 1024, 3)
        """
        assert 'valid_mask' not in batch, "与原实现一致：不支持 valid_mask"

        # 归一化
        nobs = self.normalizer.normalize(batch["obs"])
        nactions = self.normalizer["action"].normalize(batch["action"])

        B, T = nactions.shape[:2]
        device = nactions.device

        # ------ global_cond（统一来自 4096×6）------
        this_nobs = dict_apply(nobs, lambda x: x[:, :self.n_obs_steps, ...].reshape(-1, *x.shape[2:]))
        nobs_features = self.obs_encoder_stage1(this_nobs)     # (B*To, Df)
        feat_dim = nobs_features.shape[-1]
        if self.obs_as_global_cond:
            if "cross_attention" in self.condition_type:
                global_cond = nobs_features.reshape(B, self.n_obs_steps, feat_dim)   # (B, To, Df)
            else:
                global_cond = nobs_features.reshape(B, self.n_obs_steps * feat_dim)  # (B, To*Df)
        else:
            raise NotImplementedError("建议统一使用 obs_as_global_cond=True")

        # ------ control_feat（仅 controlnet 阶段）------
        control_feat = None
        if self.train_stage == "controlnet":
            assert "control_obs" in batch and "point_cloud" in batch["control_obs"], \
                "ControlNet 阶段需要 batch['control']['point_cloud']：(B, To, 1024, 3)"
            ctrl_pc = batch["control"]["point_cloud"]
            ctrl_dict = {"point_cloud": ctrl_pc}
            ctrl_dict = dict_apply(ctrl_dict, lambda x: x[:, :self.n_obs_steps, ...].reshape(-1, *x.shape[2:]))
            control_feat_bt = self.control_encoder(ctrl_dict)  # (B*To, Dc)
            control_feat = control_feat_bt.reshape(B, self.n_obs_steps, -1).mean(dim=1)  # (B, Dc)

        # ------ trajectory / condition 数据 与 mask（一致）------
        trajectory = nactions
        cond_data = trajectory
        condition_mask = self.mask_generator(trajectory.shape)

        # ------ 加噪（forward diffusion）------
        noise = torch.randn_like(trajectory)
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps,
                                  (B,), device=device).long()
        noisy_trajectory = self.noise_scheduler.add_noise(trajectory, noise, timesteps)

        # inpainting：mask 位置用干净 cond_data
        noisy_trajectory[condition_mask] = cond_data[condition_mask]

        # ------ 预测噪声 / 残差 ------
        if self.train_stage == "controlnet":
            pred = self.model(sample=noisy_trajectory,
                              timestep=timesteps,
                              control_input=control_feat,
                              local_cond=None,
                              global_cond=global_cond)
        else:
            pred = self.model(sample=noisy_trajectory,
                              timestep=timesteps,
                              local_cond=None,
                              global_cond=global_cond)

        # ------ 目标定义：与原版一致 ------
        pred_type = self.noise_scheduler.config.prediction_type
        if pred_type == 'epsilon':
            target = noise
        elif pred_type == 'sample':
            target = trajectory
        elif pred_type == 'v_prediction':
            # 你的调度器若提供 alpha_t/sigma_t，可按原版计算 v_t
            self.noise_scheduler.alpha_t = self.noise_scheduler.alpha_t.to(device)
            self.noise_scheduler.sigma_t = self.noise_scheduler.sigma_t.to(device)
            alpha_t = self.noise_scheduler.alpha_t[timesteps].unsqueeze(-1).unsqueeze(-1)
            sigma_t = self.noise_scheduler.sigma_t[timesteps].unsqueeze(-1).unsqueeze(-1)
            v_t = alpha_t * noise - sigma_t * trajectory
            target = v_t
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        # ------ MSE（mask 位置不计入）------
        loss = F.mse_loss(pred, target, reduction='none')
        loss = loss * (~condition_mask).type(loss.dtype)
        loss = reduce(loss, 'b ... -> b (...)', 'mean').mean()

        return loss, {"bc_loss": float(loss.item())}


# ======================= 使用示例 =======================
def _demo_main():
    """
    仅演示 API 级联（与原版 main 类似），不实际跑训练。
    """
    # —— shape_meta ——（示例）To=2
    shape_meta = {
        "obs": {
            "agent_pos": {"shape": (2, 10)},         # 举例用的低维
            "agent_rot": {"shape": (2, 6)},
            "point_cloud": {"shape": (2, 4096, 6)},  # Stage1: 4096×6
        },
        "action": {"shape": (30,)}
    }

    # —— 调度器 ——
    noise_scheduler = DDPMScheduler(num_train_timesteps=50)

    # —— 编码器 cfg（示例）——
    stage1_cfg = type("cfg", (), {
        "in_channels": 6,
        "out_channels": 128,
        "use_layernorm": True,
        "final_norm": "layernorm",
        "normal_channel": False,
        "num_points": 4096,
        "use_bn": True,
        # （可选）保留默认：iDP3Encoder 在 stage1 里通常期望有 agent_pos/agent_rot
    })()

    stage2_cfg = type("cfg", (), {
        "in_channels": 3,
        "out_channels": 128,
        "use_layernorm": True,
        "final_norm": "layernorm",
        "normal_channel": False,
        "num_points": 1024,
        "use_bn": True,
        # 关键：让 iDP3Encoder 把 point_cloud 当作 state 分支来取 shape，避免找 agent_pos
        "state_key": "point_cloud",
    })()

    # —— Stage1: 训练 Unet —— #
    policy_unet = DiffusionPointcloudControlPolicy(
        shape_meta=shape_meta,
        noise_scheduler=noise_scheduler,
        horizon=16,
        n_action_steps=15,
        n_obs_steps=2,
        obs_as_global_cond=True,
        diffusion_step_embed_dim=128,
        down_dims=(256, 512, 1024),
        kernel_size=5,
        n_groups=8,
        train_stage="unet",
        stage1_pointcloud_encoder_cfg=stage1_cfg,
        stage2_control_encoder_cfg=stage2_cfg
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy_unet.to(device)

    # 构造 batch（示例）
    B = 2
    obs_batch = {
        "agent_pos": torch.randn(B, 2, 10, device=device),
        "agent_rot": torch.randn(B, 2, 6, device=device),
        "point_cloud": torch.randn(B, 2, 4096, 6, device=device),
    }
    act_batch = torch.randn(B, 16, 30, device=device)
    batch_unet = {"obs": obs_batch, "action": act_batch}

    # 假 normalizer（示例）
    for key in ["agent_pos", "agent_rot", "point_cloud", "action"]:
        policy_unet.normalizer.params_dict[key] = nn.ParameterDict({
            "mean": nn.Parameter(torch.zeros(1, device=device)),
            "scale": nn.Parameter(torch.ones(1, device=device)),
            "offset": nn.Parameter(torch.zeros(1, device=device))
        })

    # 前向与 loss（Stage1）
    with torch.no_grad():
        _ = policy_unet.forward(obs_batch)
    loss1, _ = policy_unet.compute_loss(batch_unet)
    print("[Stage1] loss:", float(loss1.item()))

    # —— 保存 Unet ——（真实训练后保存）
    # torch.save(policy_unet.model.state_dict(), "unet_pretrained.pth")

    # —— Stage2: 训练 ControlNet（示例）—— #
    # 注意：这里为了演示直接复用 state_dict；实战中请从磁盘加载
    pretrained_unet_state = policy_unet.model.state_dict()

    policy_ctrl = DiffusionPointcloudControlPolicy(
        shape_meta=shape_meta,
        noise_scheduler=copy.deepcopy(noise_scheduler),
        horizon=16,
        n_action_steps=15,
        n_obs_steps=2,
        obs_as_global_cond=True,
        diffusion_step_embed_dim=128,
        down_dims=(256, 512, 1024),
        kernel_size=5,
        n_groups=8,
        train_stage="controlnet",
        pretrained_unet_path=None,   # 下面手动 load_state_dict
        stage1_pointcloud_encoder_cfg=stage1_cfg,
        stage2_control_encoder_cfg=stage2_cfg
    ).to(device)
    # 手动加载并冻结（等价于传 pretrained_unet_path）
    policy_ctrl.model.unet.load_state_dict(pretrained_unet_state, strict=True)
    for p in policy_ctrl.model.unet.parameters():
        p.requires_grad = False

    # ControlNet 训练 batch（多了 control_obs）
    batch_ctrl = {
        "obs": obs_batch,                 # 仍然需要 4096×6，用于 global_cond
        "action": act_batch,
        "control_obs": {
            "point_cloud": torch.randn(B, 2, 1024, 3, device=device)  # ControlNet 输入
        }
    }

    # 前向与 loss（Stage2）
    with torch.no_grad():
        out = policy_ctrl.forward({**obs_batch, "control_point_cloud": batch_ctrl["control_obs"]["point_cloud"]})
        print("forward(action) shape:", out.shape)
    loss2, _ = policy_ctrl.compute_loss(batch_ctrl)
    print("[Stage2] loss:", float(loss2.item()))


if __name__ == "__main__":
    _demo_main()
