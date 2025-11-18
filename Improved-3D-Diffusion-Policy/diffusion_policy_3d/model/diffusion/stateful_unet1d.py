# -*- coding: utf-8 -*-
"""
IDP3 Conditional Unet + ControlNet (Control input = 1024-d)
- 修复了解码阶段的时间维不对齐问题
- ControlNet 注入遵循：先加同尺度 control 编码特征，再 skip-concat，再 zero-conv 残差注入

依赖：
- torch, einops
- diffusion_policy_3d.model.diffusion.conv1d_components: Downsample1d, Upsample1d, Conv1dBlock
- diffusion_policy_3d.model.diffusion.positional_embedding: SinusoidalPosEmb
"""

from typing import Union
import logging
from termcolor import cprint
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from einops.layers.torch import Rearrange

# 若你的工程路径不同，请按需修改导入
from diffusion_policy_3d.model.diffusion.conv1d_components import (
    Downsample1d, Upsample1d, Conv1dBlock)
from diffusion_policy_3d.model.diffusion.positional_embedding import SinusoidalPosEmb

logger = logging.getLogger(__name__)


# ----------------------------- 基础模块 -----------------------------
class CrossAttention(nn.Module):
    def __init__(self, in_dim, cond_dim, out_dim):
        super().__init__()
        self.query_proj = nn.Linear(in_dim, out_dim)
        self.key_proj = nn.Linear(cond_dim, out_dim)
        self.value_proj = nn.Linear(cond_dim, out_dim)

    def forward(self, x, cond):
        # x: [B, t_act, in_dim]
        # cond: [B, t_obs, cond_dim]
        query = self.query_proj(x)           # [B, t_act, out_dim]
        key = self.key_proj(cond)            # [B, t_obs, out_dim]
        value = self.value_proj(cond)        # [B, t_obs, out_dim]
        attn_weights = torch.matmul(query, key.transpose(-2, -1))  # [B, t_act, t_obs]
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, value)            # [B, t_act, out_dim]
        return attn_output


class ConditionalResidualBlock1D(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 cond_dim,
                 kernel_size=3,
                 n_groups=8,
                 condition_type='film'):
        super().__init__()
        self.blocks = nn.ModuleList([
            Conv1dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups),
            Conv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups),
        ])
        self.condition_type = condition_type

        cond_channels = out_channels
        if condition_type == 'film':
            cond_channels = out_channels * 2
            self.cond_encoder = nn.Sequential(
                nn.Mish(),
                nn.Linear(cond_dim, cond_channels),
                Rearrange('batch t -> batch t 1'),
            )
        elif condition_type == 'add':
            self.cond_encoder = nn.Sequential(
                nn.Mish(),
                nn.Linear(cond_dim, out_channels),
                Rearrange('batch t -> batch t 1'),
            )
        elif condition_type == 'cross_attention_add':
            self.cond_encoder = CrossAttention(in_channels, cond_dim, out_channels)
        elif condition_type == 'cross_attention_film':
            cond_channels = out_channels * 2
            self.cond_encoder = CrossAttention(in_channels, cond_dim, cond_channels)
        elif condition_type == 'mlp_film':
            cond_channels = out_channels * 2
            self.cond_encoder = nn.Sequential(
                nn.Mish(),
                nn.Linear(cond_dim, cond_dim),
                nn.Mish(),
                nn.Linear(cond_dim, cond_channels),
                Rearrange('batch t -> batch t 1'),
            )
        else:
            raise NotImplementedError(f"condition_type {condition_type} not implemented")

        self.out_channels = out_channels
        self.residual_conv = nn.Conv1d(in_channels, out_channels, 1) \
            if in_channels != out_channels else nn.Identity()

    def forward(self, x, cond=None):
        """
        x : [B, in_channels, T]
        cond : [B, cond_dim] or [B, T, cond_dim] (针对 cross_attention* )
        out : [B, out_channels, T]
        """
        out = self.blocks[0](x)
        if cond is not None:
            if self.condition_type == 'film':
                embed = self.cond_encoder(cond)                         # [B, 2*out_ch, 1]
                embed = embed.reshape(embed.shape[0], 2, self.out_channels, 1)
                scale = embed[:, 0, ...]
                bias  = embed[:, 1, ...]
                out = scale * out + bias
            elif self.condition_type == 'add':
                embed = self.cond_encoder(cond)                         # [B, out_ch, 1]
                out = out + embed
            elif self.condition_type == 'cross_attention_add':
                # x: [B, C, T] -> [B, T, C]
                embed = self.cond_encoder(x.permute(0, 2, 1), cond)     # [B, T, out_ch]
                embed = embed.permute(0, 2, 1)                          # [B, out_ch, T]
                out = out + embed
            elif self.condition_type == 'cross_attention_film':
                embed = self.cond_encoder(x.permute(0, 2, 1), cond)     # [B, T, 2*out_ch]
                embed = embed.permute(0, 2, 1)                          # [B, 2*out_ch, T]
                embed = embed.reshape(embed.shape[0], 2, self.out_channels, -1)
                scale = embed[:, 0, ...]
                bias  = embed[:, 1, ...]
                out = scale * out + bias
            elif self.condition_type == 'mlp_film':
                embed = self.cond_encoder(cond)                         # [B, 2*out_ch, 1]
                embed = embed.reshape(embed.shape[0], 2, self.out_channels, -1)
                scale = embed[:, 0, ...]
                bias  = embed[:, 1, ...]
                out = scale * out + bias
            else:
                raise NotImplementedError(f"condition_type {self.condition_type} not implemented")

        out = self.blocks[1](out)
        out = out + self.residual_conv(x)
        return out


# ----------------------------- 原始 ConditionalUnet1D -----------------------------
class ConditionalUnet1D(nn.Module):
    def __init__(self, 
        input_dim,
        local_cond_dim=None,
        global_cond_dim=None,
        diffusion_step_embed_dim=256,
        down_dims=[256,512,1024],
        kernel_size=3,
        n_groups=8,
        condition_type='film',
        use_down_condition=True,
        use_mid_condition=True,
        use_up_condition=True,
        ):
        super().__init__()
        self.condition_type = condition_type
        self.use_down_condition = use_down_condition
        self.use_mid_condition = use_mid_condition
        self.use_up_condition = use_up_condition

        all_dims = [input_dim] + list(down_dims)
        start_dim = down_dims[0]

        dsed = diffusion_step_embed_dim
        diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(dsed),
            nn.Linear(dsed, dsed * 4),
            nn.Mish(),
            nn.Linear(dsed * 4, dsed),
        )
        cond_dim = dsed
        if global_cond_dim is not None:
            cond_dim += global_cond_dim

        in_out = list(zip(all_dims[:-1], all_dims[1:]))

        local_cond_encoder = None
        if local_cond_dim is not None:
            _, dim_out = in_out[0]
            dim_in = local_cond_dim
            local_cond_encoder = nn.ModuleList([
                # down encoder
                ConditionalResidualBlock1D(
                    dim_in, dim_out, cond_dim=cond_dim, 
                    kernel_size=kernel_size, n_groups=n_groups,
                    condition_type=condition_type),
                # up encoder
                ConditionalResidualBlock1D(
                    dim_in, dim_out, cond_dim=cond_dim, 
                    kernel_size=kernel_size, n_groups=n_groups,
                    condition_type=condition_type)
            ])

        mid_dim = all_dims[-1]
        self.mid_modules = nn.ModuleList([
            ConditionalResidualBlock1D(
                mid_dim, mid_dim, cond_dim=cond_dim,
                kernel_size=kernel_size, n_groups=n_groups,
                condition_type=condition_type
            ),
            ConditionalResidualBlock1D(
                mid_dim, mid_dim, cond_dim=cond_dim,
                kernel_size=kernel_size, n_groups=n_groups,
                condition_type=condition_type
            ),
        ])

        down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            down_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(
                    dim_in, dim_out, cond_dim=cond_dim, 
                    kernel_size=kernel_size, n_groups=n_groups,
                    condition_type=condition_type),
                ConditionalResidualBlock1D(
                    dim_out, dim_out, cond_dim=cond_dim, 
                    kernel_size=kernel_size, n_groups=n_groups,
                    condition_type=condition_type),
                Downsample1d(dim_out) if not is_last else nn.Identity()
            ]))

        up_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            up_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(
                    dim_out*2, dim_in, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups,
                    condition_type=condition_type),
                ConditionalResidualBlock1D(
                    dim_in, dim_in, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups,
                    condition_type=condition_type),
                Upsample1d(dim_in) if not is_last else nn.Identity()
            ]))
        
        final_conv = nn.Sequential(
            Conv1dBlock(start_dim, start_dim, kernel_size=kernel_size),
            nn.Conv1d(start_dim, input_dim, 1),
        )
        
        self.diffusion_step_encoder = diffusion_step_encoder
        self.local_cond_encoder = local_cond_encoder
        self.up_modules = up_modules
        self.down_modules = down_modules
        self.final_conv = final_conv

        logger.info(
            "number of parameters: %e", sum(p.numel() for p in self.parameters())
        )

    def forward(self, 
            sample: torch.Tensor, 
            timestep: Union[torch.Tensor, float, int], 
            local_cond=None, global_cond=None, **kwargs):
        """
        sample: (B, C, T)
        timestep: (B,)
        local_cond: (B, C_l, T) or None
        global_cond: (B, G) or None
        return: (B, C, T)
        """
        
        x = sample
        # time embed
        if not torch.is_tensor(timestep):
            timestep = torch.tensor([timestep], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timestep) and len(timestep.shape) == 0:
            timestep = timestep[None].to(sample.device)
        timestep = timestep.expand(sample.shape[0])

        t_embed = self.diffusion_step_encoder(timestep)   # [B, dsed]
        if global_cond is not None:
            global_feature = torch.cat([t_embed, global_cond], axis=-1)
        else:
            global_feature = t_embed

        # local cond encoder (可选)
        h_local = []
        if local_cond is not None and self.local_cond_encoder is not None:
            resnet_l1, resnet_l2 = self.local_cond_encoder
            x_l = resnet_l1(local_cond, global_feature)
            h_local.append(x_l)
            x_l = resnet_l2(local_cond, global_feature)
            h_local.append(x_l)

        # down path
        h = []
        for idx, (resnet, resnet2, downsample) in enumerate(self.down_modules):
            if self.use_down_condition:
                x = resnet(x, global_feature)
                if idx == 0 and len(h_local) > 0:
                    x = x + h_local[0]
                x = resnet2(x, global_feature)
            else:
                x = resnet(x)
                if idx == 0 and len(h_local) > 0:
                    x = x + h_local[0]
                x = resnet2(x)
            h.append(x)
            x = downsample(x)

        # mid
        for mid_module in self.mid_modules:
            if self.use_mid_condition:
                x = mid_module(x, global_feature)
            else:
                x = mid_module(x)

        # up path
        for idx, (resnet, resnet2, upsample) in enumerate(self.up_modules):
            x = torch.cat((x, h.pop()), dim=1)
            if self.use_up_condition:
                x = resnet(x, global_feature)
                if idx == len(self.up_modules) and len(h_local) > 0:
                    x = x + h_local[1]
                x = resnet2(x, global_feature)
            else:
                x = resnet(x)
                if idx == len(self.up_modules) and len(h_local) > 0:
                    x = x + h_local[1]
                x = resnet2(x)
            x = upsample(x)

        x = self.final_conv(x)
        return x


# ----------------------------- 带 ControlNet 的 ConditionalControlUnet1D -----------------------------
class ConditionalControlUnet1D(nn.Module):
    def __init__(self, 
        input_dim,
        control_dim=1024,                 # ControlNet 输入维度
        local_cond_dim=None,              # 本脚本测试默认 None，方便快速验证
        global_cond_dim=None,
        diffusion_step_embed_dim=256,
        down_dims=[256,512,1024],
        kernel_size=3,
        n_groups=8,
        condition_type='film',
        use_down_condition=True,
        use_mid_condition=True,
        use_up_condition=True,
        ):
        super().__init__()

        # --------- 原始 Unet 主干（可加载 & 冻结） ---------
        self.unet = ConditionalUnet1D(
            input_dim=input_dim,
            local_cond_dim=local_cond_dim,
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

        for param in self.unet.parameters():
            param.requires_grad = False

        # --------- ControlNet 结构 ---------
        all_dims = [input_dim] + list(down_dims)
        in_out = list(zip(all_dims[:-1], all_dims[1:]))
        mid_dim = all_dims[-1]
        cond_dim = diffusion_step_embed_dim + (global_cond_dim or 0)

        # 控制输入预处理：1024 -> input_dim，然后在时间维广播
        self.control_proj = nn.Linear(control_dim, input_dim)

        # control encoder：与主干 down 对齐
        self.ctrl_downs = nn.ModuleList([
            nn.ModuleList([
                ConditionalResidualBlock1D(dim_in, dim_out, cond_dim,
                                           kernel_size=kernel_size, n_groups=n_groups,
                                           condition_type=condition_type),
                ConditionalResidualBlock1D(dim_out, dim_out, cond_dim,
                                           kernel_size=kernel_size, n_groups=n_groups,
                                           condition_type=condition_type),
                Downsample1d(dim_out) if i < len(in_out)-1 else nn.Identity()
            ]) for i, (dim_in, dim_out) in enumerate(in_out)
        ])

        # control mid
        self.ctrl_mid1 = ConditionalResidualBlock1D(mid_dim, mid_dim, cond_dim,
                                                    kernel_size=kernel_size, n_groups=n_groups,
                                                    condition_type=condition_type)
        self.ctrl_mid2 = ConditionalResidualBlock1D(mid_dim, mid_dim, cond_dim,
                                                    kernel_size=kernel_size, n_groups=n_groups,
                                                    condition_type=condition_type)

        # zero-conv 工具
        def zero_conv(in_ch, out_ch):
            conv = nn.Conv1d(in_ch, out_ch, 1)
            nn.init.zeros_(conv.weight)
            nn.init.zeros_(conv.bias)
            return conv

        # mid 注入
        self.ctrl_mid_zero = zero_conv(mid_dim, mid_dim)

        # 每层解码注入（输入是当前解码的 concat 特征）
        self.ctrl_zero_blocks = nn.ModuleList([
            zero_conv(dim_out*2, dim_in) for dim_in, dim_out in reversed(in_out[1:])
        ])

    def forward(self,
            sample: torch.Tensor,              # (B, C, T)
            timestep: Union[torch.Tensor, int],
            control_input: torch.Tensor,       # (B, 1024)
            local_cond: torch.Tensor = None,   # (B, C_l, T)
            global_cond: torch.Tensor = None,
            **kwargs):
        
        sample = einops.rearrange(sample, 'b h t -> b t h')
        B, C, T = sample.shape

        # --- Step 1. 时间与全局条件 ---
        if not torch.is_tensor(timestep):
            timestep = torch.tensor([timestep], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timestep) and len(timestep.shape) == 0:
            timestep = timestep[None].to(sample.device)
        timestep = timestep.expand(B)
        t_embed = self.unet.diffusion_step_encoder(timestep)  # [B, dsed]

        if global_cond is not None:
            global_feature = torch.cat([t_embed, global_cond], dim=-1)
        else:
            global_feature = t_embed

        # --- Step 2. 处理 local_cond（局部时序条件） ---
        # h_local = []
        # if local_cond is not None and self.unet.local_cond_encoder is not None:
        #     # 保证 (B, C_l, T)
        #     if local_cond.ndim == 2:
        #         local_cond = local_cond.unsqueeze(-1).expand(-1, -1, T)
        #     resnet_l1, resnet_l2 = self.unet.local_cond_encoder
        #     # 第一层特征注入 encoder
        #     x_l = resnet_l1(local_cond, global_feature)
        #     h_local.append(x_l)
        #     # 第二层特征注入 decoder
        #     x_l = resnet_l2(local_cond, global_feature)
        #     h_local.append(x_l)

        # --- Step 3. 主干下采样（encoder） ---
        x = sample
        h = []
        for idx, (res1, res2, down) in enumerate(self.unet.down_modules):
            x = res1(x, global_feature)
            # 在第0层融合 local_cond
            # if idx == 0 and len(h_local) > 0:
            #     x = x + h_local[0]
            x = res2(x, global_feature)
            h.append(x)
            x = down(x)

        # --- Step 4. 主干中间层 ---
        for mid in self.unet.mid_modules:
            x = mid(x, global_feature)

        # --- Step 5. ControlNet 编码器 ---
        ctrl = self.control_proj(control_input)           # [B, C_in]
        ctrl = ctrl.unsqueeze(-1).expand(-1, -1, T)       # [B, C_in, T]
        # x_ctrl = sample + ctrl
        x_ctrl = ctrl
        h_ctrl = []
        for res1, res2, down in self.ctrl_downs:
            x_ctrl = res1(x_ctrl, global_feature)
            x_ctrl = res2(x_ctrl, global_feature)
            h_ctrl.append(x_ctrl)
            x_ctrl = down(x_ctrl)

        # --- Step 6. Control mid ---
        x_ctrl = self.ctrl_mid1(x_ctrl, global_feature)
        x_ctrl = self.ctrl_mid2(x_ctrl, global_feature)

        # --- Step 7. mid 注入 ---
        x = x + self.ctrl_mid_zero(x_ctrl)

        # --- Step 8. 解码阶段（decoder） ---
        for i, (res1, res2, up) in enumerate(self.unet.up_modules):
            x = x + h_ctrl.pop()                        # control skip
            x = torch.cat((x, h.pop()), dim=1)          # 主干 skip
            x_hat = self.ctrl_zero_blocks[i](x)         # 控制残差
            x = res1(x, global_feature)
            # 在最后一层融合 local_cond
            # if i == len(self.unet.up_modules)-1 and len(h_local) > 1:
            #     x = x + h_local[1]
            x = res2(x, global_feature)
            x = x + x_hat
            # cprint(f"x_hat:{x_hat}", 'red')
            x = up(x)

        # --- Step 9. 输出 ---
        out = self.unet.final_conv(x)      # [B, C, T]
        out = einops.rearrange(out, 'b t h -> b h t')
        return out



# ----------------------------- 测试代码 -----------------------------
def test_control_unet_forward():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # 配置
    input_dim = 16           # 动作/状态通道
    control_dim = 1024       # ControlNet 输入
    global_cond_dim = 64     # 全局条件维度
    horizon = 32             # 时间窗口
    batch_size = 8

    # 创建模型
    model = ConditionalControlUnet1D(
        input_dim=input_dim,
        control_dim=control_dim,
        local_cond_dim=None,              # 测试中先关掉 local_cond，避免额外分支
        global_cond_dim=global_cond_dim,
        diffusion_step_embed_dim=256,
        down_dims=[128, 256, 512],
        kernel_size=3,
        n_groups=8,
        condition_type='film',
    ).to(device)
    model.eval()

    # 构造假输入
    sample = torch.randn(batch_size, input_dim, horizon, device=device)  # (B, C, T)
    timestep = torch.randint(0, 1000, (batch_size,), device=device)      # (B,)
    control_input = torch.randn(batch_size, control_dim, device=device)  # (B, 1024)
    global_cond = torch.randn(batch_size, global_cond_dim, device=device)# (B, 64)

    # 前向
    with torch.no_grad():
        output = model(
            sample=sample,
            timestep=timestep,
            control_input=control_input,
            local_cond=None,
            global_cond=global_cond
        )

    print(f"Input shape : {sample.shape}")
    print(f"Output shape: {output.shape}")
    assert output.shape == sample.shape, "输出形状应与输入一致 (B, C, T)"
    print("✅ Forward pass successful! Model output verified.")


if __name__ == "__main__":
    test_control_unet_forward()
