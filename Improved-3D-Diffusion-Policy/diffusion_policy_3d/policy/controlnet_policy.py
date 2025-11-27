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

from diffusion_policy_3d.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusion_policy_3d.model.diffusion.stateful_unet1d import ConditionalControlUnet1D

from diffusion_policy_3d.model.vision_3d.pointnet_extractor import iDP3Encoder,ControlNetEncoder

class DiffusionPointcloudControlPolicy(BasePolicy):
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

        action_shape = shape_meta["action"]["shape"]
        if len(action_shape) == 1:
            self.action_dim = action_shape[0]
        elif len(action_shape) == 2:
            self.action_dim = action_shape[0] * action_shape[1]
        
        obs_shape_meta = shape_meta["obs"]
        obs_shapes = dict_apply(obs_shape_meta, lambda x: x["shape"])

        control_shape_meta = shape_meta["control"]
        control_shapes = dict_apply(control_shape_meta, lambda x: x["shape"])
        
        self.obs_encoder_stage1 = iDP3Encoder(
            observation_space=obs_shapes,                    # 这里通常包含 point_cloud: (To, 4096, 6) 以及其他低维 obs
            pointcloud_encoder_cfg=stage1_pointcloud_encoder_cfg,
            use_pc_color=True,                               # 4096×6 包含 RGB
            pointnet_type="multi_stage_pointnet"
        )
        self.obs_feature_dim = self.obs_encoder_stage1.output_shape()

        if self.train_stage=="controlnet":
            self.control_encoder2 = ControlNetEncoder(
                observation_space=control_shapes,                # 这里通常只包含 point_cloud: (Tc, 1024, 3)
                pointcloud_encoder_cfg=stage2_control_encoder_cfg,
                use_pc_color=False,                              # 1024×3 不包含 RGB
                pointnet_type="multi_stage_pointnet"
            )
            self.control_feature_dim = self.control_encoder2.output_shape()

        if self.train_stage == "unet":
            input_dim = self.action_dim
            global_cond_dim = None
            if self.obs_as_global_cond:
                if "cross_attention" in self.condition_type:
                    global_cond_dim = self.obs_feature_dim
                else:
                    global_cond_dim = self.obs_feature_dim * self.n_obs_steps
            else:
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
            input_dim = self.action_dim
            global_cond_dim = None
            if self.obs_as_global_cond:
                if "cross_attention" in self.condition_type:
                    global_cond_dim = self.obs_feature_dim
                else:
                    global_cond_dim = self.obs_feature_dim * self.n_obs_steps
            else:
                input_dim = self.action_dim + self.obs_feature_dim

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

            assert pretrained_unet_path is not None, \
                "ControlNet 阶段需要提供 pretrained_unet_path（上一阶段训练好的 Unet 权重）"
            cprint(f"[ControlNet] Loading pretrained UNet from: {pretrained_unet_path}", "yellow")
            # unet_state = torch.load(pretrained_unet_path, map_location="cpu", weights_only=False)
            # self.model.unet.load_state_dict(unet_state, strict=True)
            if pretrained_unet_path.endswith(".ckpt"):
                ckpt = torch.load(pretrained_unet_path, map_location="cpu",weights_only=False)

            #     # Lightning 保存格式兼容
            #     if "state_dict" in ckpt:
            #         unet_state = ckpt["state_dict"]
            #     elif "state_dicts" in ckpt:
            #         # 如果包含多个模块（如 ema、model、optimizer）
            #         unet_state = ckpt["state_dicts"].get("model", ckpt["state_dicts"])
            #     else:
            #         unet_state = ckpt
            # else:
            #     unet_state = torch.load(pretrained_unet_path, map_location="cpu")

            raw_state = ckpt["state_dicts"]["model"]

            # 去掉前缀 'model.' 以匹配 ConditionalUnet1D
            clean_state = {}
            encoder_dict = {}
            self.normalizer_state = {}
            for k, v in raw_state.items():
                if k.startswith("model."):
                    clean_state[k.replace("model.", "")] = v
                elif k.startswith("obs_encoder_stage1."):
                    # 去掉前缀，直接加载给 self.obs_encoder_stage1
                    clean_k = k.replace("obs_encoder_stage1.", "")
                    encoder_dict[clean_k] = v
                elif k.startswith("normalizer."):
                    # 去掉 "normalizer." 前缀
                    # 例如: "normalizer.params_dict.action.offset" -> "params_dict.action.offset"
                    clean_k = k.replace("normalizer.", "")
                    self.normalizer_state[clean_k] = v
                else:
                    clean_state[k] = v
            missing, unexpected = self.model.unet.load_state_dict(clean_state, strict=False)
            cprint(f"[ControlNet] loaded pretrained UNet with {len(missing)} missing and {len(unexpected)} unexpected keys","red")

            for p in self.model.unet.parameters():
                p.requires_grad = False
            cprint("[ControlNet] UNet is frozen. Only training ControlNet branch.", "yellow")
            if len(encoder_dict) > 0:
            # 注意：这里要加 strict=False，因为 obs_encoder 可能包含一些不需要的 buffer
                missing, unexpected = self.obs_encoder_stage1.load_state_dict(encoder_dict, strict=False)
                cprint(f"[ControlNet] Stage1 Obs Encoder loaded.missing {len(missing)},unexpected {len(unexpected)}", "green")
            else:
                cprint("[ControlNet] ❌ 警告：未在 checkpoint 中找到 Obs Encoder 权重！请确认 Stage 1 保存逻辑是否保存了整个 Policy。", "red")
                cprint("[ControlNet] 如果继续，Encoder 将是随机初始化的，训练将失败。", "red")

            for param in self.obs_encoder_stage1.parameters():
                param.requires_grad = False

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

    def forward(self, obs_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
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

        this_nobs = dict_apply(nobs, lambda x: x[:, :To, ...].reshape(-1, *x.shape[2:]))
        nobs_features = self.obs_encoder_stage1(this_nobs)  # (B*To, Df)
        feat_dim = nobs_features.shape[-1]

        if self.obs_as_global_cond:
            if "cross_attention" in self.condition_type:
                global_cond = nobs_features.reshape(B, To, feat_dim)
            else:
                global_cond = nobs_features.reshape(B, To * feat_dim)
        else:
            raise NotImplementedError("建议统一使用 obs_as_global_cond=True")
        
        control_feat = None
        if self.train_stage == "controlnet":
            assert "control_point_cloud" in obs_dict, \
                "ControlNet 阶段需要 obs_dict['control_point_cloud']，形状 (B, To, 1024, 3)"
            ctrl_pc = obs_dict["control_point_cloud"]
            # 编码器期望 dict 输入
            ctrl_dict = {"control_point_cloud": ctrl_pc}
            ctrl_dict = dict_apply(ctrl_dict, lambda x: x[:, :To, ...].reshape(-1, *x.shape[2:]))
            control_feat = self.control_encoder2(ctrl_dict)  # (B*To, Dc)
            control_feat = control_feat.reshape(B, To, -1).mean(dim=1)  # (B, Dc)

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

        naction_pred = nsample[..., :Da]
        action_pred = self.normalizer["action"].unnormalize(naction_pred)
        start = To - 1
        end = start + self.n_action_steps
        return action_pred[:, start:end]

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
            # print(obs_dict.keys())
            ctrl_pc = obs_dict["control_point_cloud"]  # (B, To, 1024, 3)
            ctrl_dict = {"control_point_cloud": ctrl_pc}
            ctrl_dict = dict_apply(ctrl_dict, lambda x: x[:, :To, ...].reshape(-1, *x.shape[2:]))
            control_feat_bt = self.control_encoder2(ctrl_dict)  # (B*To, Dc)
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
        if self.train_stage == "controlnet":
            controlnet_state = normalizer.state_dict()
            unet_state = self.normalizer_state.copy()    
            
            for k, v in controlnet_state.items():
                if k in unet_state:
                    controlnet_state[k] = v  
            missing, unexpected = self.normalizer.load_state_dict(controlnet_state, strict=False)


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
            assert "control" in batch and "control_point_cloud" in batch["control"], \
                "ControlNet 阶段需要 batch['control']['control_point_cloud']：(B, To, 1024, 3)"
            ctrl_pc = batch["control"]["control_point_cloud"]
            ctrl_dict = {"control_point_cloud": ctrl_pc}
            ctrl_dict = dict_apply(ctrl_dict, lambda x: x[:, :self.n_obs_steps, ...].reshape(-1, *x.shape[2:]))
            control_feat_bt = self.control_encoder2(ctrl_dict)  # (B*To, Dc)
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
        # loss = reduce(loss, 'b ... -> b (...)', 'mean').mean()

        feat_dim = loss.shape[-1] 
        weights = torch.ones(feat_dim, device=loss.device, dtype=loss.dtype)
        loss_dict = dict() 

        if feat_dim == 7:
            weights[:6] = 1
            weights[6] = 0.2
            loss_joint = loss[..., :6].mean().item()*weights[0].item()
            loss_gripper = loss[..., 6].mean().item()*weights[6].item()
            loss_dict["loss_joint"] = loss_joint
            loss_dict["loss_gripper"] = loss_gripper
        if feat_dim == 10:
            weights[:3] = 10
            weights[3:9] = 1
            weights[9] = 0.2
            loss_pos = loss[..., :3].mean().item()*weights[0].item()
            loss_rot = loss[..., 3:9].mean().item()*weights[3].item()
            loss_gripper = loss[..., 9].mean().item()*weights[9].item()
            loss_dict["loss_pos"] = loss_pos
            loss_dict["loss_rot"] = loss_rot
            loss_dict["loss_gripper"] = loss_gripper
        if feat_dim == 14:
            weights[:6] = 1
            weights[6] = 0.2
            weights[7:13] = 1
            weights[13] = 0.2
            loss_joint = loss[..., :6].mean().item()*weights[0].item() + loss[..., 7:13].mean().item()*weights[7].item()
            loss_gripper = loss[..., 6].mean().item()*weights[6].item() + loss[..., 13].mean().item()*weights[13].item()
            loss_dict["loss_joint"] = loss_joint
            loss_dict["loss_gripper"] = loss_gripper


        weighted_loss = loss * weights 
        total_loss = reduce(weighted_loss, 'b ... -> b (...)', 'mean')
        total_loss = total_loss.mean()
        loss_dict["total_loss"] = total_loss.item()

        return total_loss, loss_dict







def main():
    shape_meta = {
        "action": {"shape": (1+0,)},
        "obs": {
            "agent_pos": {"shape": (10, )},         # 举例用的低维
            "agent_rot": {"shape": (6, )},
            "point_cloud": {"shape": (4096, 6)},  # Stage1: 4096×6
        },
        "control": {
            "point_cloud": {"shape": (1024, 3)},  # Stage2: 1024×3
        }
    }

    noise_scheduler = DDPMScheduler(num_train_timesteps=50)
    stage1_cfg = type("cfg", (), {
        "in_channels": 6,
        "out_channels": 128,
        "use_layernorm": True,
        "final_norm": "layernorm",
        "normal_channel": False,
        "num_points": 4096,
        "use_bn": True,
    })()

    stage2_cfg = type("cfg", (), {
        "in_channels": 3,
        "out_channels": 128,
        "use_layernorm": True,
        "final_norm": "layernorm",
        "normal_channel": False,
        "num_points": 1024,
        "use_bn": True,
    })()

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

    B = 2
    obs_batch = {
        "agent_pos": torch.randn(B,2, 10, device=device),
        "agent_rot": torch.randn(B,2, 6, device=device),
        "point_cloud": torch.randn(B, 2, 4096, 6, device=device),
    }
    act_batch = torch.randn(B, 16, 30, device=device)
    batch_unet = {"obs": obs_batch, "action": act_batch}

    for key in ["agent_pos", "agent_rot", "point_cloud", "action"]:
        policy_unet.normalizer.params_dict[key] = nn.ParameterDict({
            "mean": nn.Parameter(torch.zeros(1, device=device)),
            "scale": nn.Parameter(torch.ones(1, device=device)),
            "offset": nn.Parameter(torch.zeros(1, device=device))
        })
    
    with torch.no_grad():
        _ = policy_unet.forward(obs_batch)
    loss1, _ = policy_unet.compute_loss(batch_unet)
    print("[Stage1] loss:", float(loss1.item()))

    pretrained_unet_state = policy_unet.model.state_dict()
    torch.save(pretrained_unet_state, "/extra/waylen/diffusion_policy/tmp/pretrained_unet_stage1.pth")

    cprint("\n=== Stage2: ControlNet Training ===", "green")
    shape_meta = {
        "action": {"shape": (30,)},
        "obs": {
            "agent_pos": {"shape": (10, )},         # 举例用的低维
            "agent_rot": {"shape": (6, )},
            "point_cloud": {"shape": (4096, 6)},  # Stage1: 4096×6
        },
        "control": {
            "control_point_cloud": {"shape": (1024, 3)},  # Stage2: 1024×3
        }
    }
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
        pretrained_unet_path="/extra/waylen/diffusion_policy/tmp/pretrained_unet_stage1.pth",   # 下面手动 load_state_dict
        stage1_pointcloud_encoder_cfg=stage1_cfg,
        stage2_control_encoder_cfg=stage2_cfg
    ).to(device)

    batch_ctrl = {
        "obs": obs_batch,                 # 仍然需要 4096×6，用于 global_cond
        "action": act_batch,
        "control": {
            "control_point_cloud": torch.randn(B, 2, 1024, 3, device=device)  # ControlNet 输入
        }
    }

    for key in ["agent_pos", "agent_rot", "point_cloud", "action", "control_point_cloud"]:
        policy_ctrl.normalizer.params_dict[key] = nn.ParameterDict({
            "mean": nn.Parameter(torch.zeros(1, device=device)),
            "scale": nn.Parameter(torch.ones(1, device=device)),
            "offset": nn.Parameter(torch.zeros(1, device=device))
        })

    with torch.no_grad():
        out = policy_ctrl.forward({**obs_batch, "control_point_cloud": batch_ctrl["control"]["control_point_cloud"]})
        print("forward(action) shape:", out.shape)
    loss2, _ = policy_ctrl.compute_loss(batch_ctrl)
    print("[Stage2] loss:", float(loss2.item()))

# if __name__ == "__main__":
#     main()