import torch
import torch.nn as nn
from diffusion_policy_3d.common.rotation_util import rotation_6d_to_matrix, matrix_to_rotation_6d
from termcolor import cprint
# import IPython
# e = IPython.embed

def _ensure_indices_valid(D, idxs, name):
    for i in idxs:
        if i >= D:
            raise ValueError(f"{name} index {i} out of range for dim {D}")

class EE6DLoss(nn.Module):
    """
    EE6DLoss: 计算位置 + 6D旋转 + 夹爪损失
    - 自动将6D旋转表示转换为3x3旋转矩阵
    - 使用Frobenius范数计算旋转误差
    - 返回与 F.mse_loss(reduction='none') 相同形状的逐维 loss map
    """

    def __init__(self, agent_pos=[[0, 3]], agent_rot=[[3, 9]], dim_action=None):
        super().__init__()
        self.agent_pos = agent_pos
        self.agent_rot = agent_rot
        self.num_arms = len(agent_pos)
        self.dim_action = dim_action

        # Loss权重缩放
        self.XYZ_SCALE = 12.0
        self.ROT_SCALE = 1.0
        self.GRIPPER_SCALE = 0.15

        # 自动生成 gripper 索引
        self.GRIPPER_IDX = [r[1] for r in agent_rot]

        self.mse = nn.MSELoss(reduction='none')
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

        print(f"[EE6DLoss] {self.num_arms} arm(s) detected")
        print(f"  pos: {self.agent_pos}")
        print(f"  rot: {self.agent_rot}")
        print(f"  grip: {self.GRIPPER_IDX}")

    def rotation_frobenius_loss(self, R_pred, R_gt):
        """
        R_pred, R_gt: [B, T, 3, 3]
        returns: [B, T, 1] 逐样本旋转误差
        """
        diff = R_pred - R_gt
        loss = torch.norm(diff, dim=(-2, -1))  # Frobenius 范数
        return loss.unsqueeze(-1)  # [B, T, 1]

    def forward(self, pred_action: torch.Tensor, target_action: torch.Tensor):
        """
        Args:
            pred_action, target_action: [B, T, D]
        Returns:
            loss_map: [B, T, D]  与 F.mse_loss(reduction='none') 形状一致
        """
        assert pred_action.shape == target_action.shape, "pred/target shapes must match"
        B, T, D = pred_action.shape

        if self.dim_action is not None:
            assert D == self.dim_action, f"Expected {self.dim_action}, got {D}"

        # 初始化逐维 loss map
        loss_map = torch.zeros_like(pred_action)

        for i in range(self.num_arms):
            pos_s, pos_e = self.agent_pos[i]
            rot_s, rot_e = self.agent_rot[i]
            grip_idx = self.GRIPPER_IDX[i]

            # ---- 平移 loss ----
            pos_err = self.mse(pred_action[:, :, pos_s:pos_e],
                               target_action[:, :, pos_s:pos_e]) * self.XYZ_SCALE
            loss_map[:, :, pos_s:pos_e] = pos_err

            # ---- 旋转 loss ----
            rot_pred_6d = pred_action[:, :, rot_s:rot_e]  # [B, T, 6]
            rot_gt_6d = target_action[:, :, rot_s:rot_e]

            # 转换为旋转矩阵
            # cprint(f"rot_pred_6d,{rot_pred_6d.shape}","red")
            # cprint(f"rot_gt_6d, {rot_gt_6d.shape}","red")
            # e()
            R_pred = rotation_6d_to_matrix(rot_pred_6d)
            R_gt = rotation_6d_to_matrix(rot_gt_6d)
            

            # 几何loss（Frobenius范数）
            rot_loss_scalar = self.rotation_frobenius_loss(R_pred, R_gt) * self.ROT_SCALE

            # 将旋转误差广播回 [B, T, 6] 对应维度（保持形状一致）
            loss_map[:, :, rot_s:rot_e] = rot_loss_scalar.expand(-1, -1, rot_e - rot_s)

            # ---- 夹爪 loss ----
            grip_pred = pred_action[:, :, grip_idx]
            grip_target = target_action[:, :, grip_idx]
            grip_err = self.bce(grip_pred, grip_target) * self.GRIPPER_SCALE
            loss_map[:, :, grip_idx] = grip_err

            # pos_loss = loss_map[:, :, :3].mean().item()
            # rot_loss = loss_map[:, :, 3:9].mean().item()
            # grip_loss = loss_map[:, :, 9].mean().item()
            # total_loss = loss_map.mean().item()

            # print("\n==== 分项 Loss 统计 ====")
            # print(f"位置 (XYZ) loss : {pos_loss:.6f}")
            # print(f"旋转 (6D→R) loss: {rot_loss:.6f}")
            # print(f"夹爪 loss        : {grip_loss:.6f}")
            # print(f"总均值 loss      : {total_loss:.6f}")

        return loss_map


if __name__ == "__main__":
    from einops import reduce
    B, T, D = 8, 4, 10
    pred = torch.randn(B, T, D)
    target = torch.randn(B, T, D)

    # 初始化 loss
    loss_fn = EE6DLoss(agent_pos=[[0, 3]], agent_rot=[[3, 9]], dim_action=10)
    loss_map = loss_fn(pred, target)  # [B, T, D]

    print(f"\n==> loss_map shape: {loss_map.shape}")

    # 各部分索引
    pos_s, pos_e = loss_fn.agent_pos[0]
    rot_s, rot_e = loss_fn.agent_rot[0]
    grip_idx = loss_fn.GRIPPER_IDX[0]

    # 计算分项平均
    pos_loss = loss_map[:, :, pos_s:pos_e].mean().item()
    rot_loss = loss_map[:, :, rot_s:rot_e].mean().item()
    grip_loss = loss_map[:, :, grip_idx].mean().item()
    total_loss = loss_map.mean().item()

    print("\n==== 分项 Loss 统计 ====")
    print(f"位置 (XYZ) loss : {pos_loss:.6f}")
    print(f"旋转 (6D→R) loss: {rot_loss:.6f}")
    print(f"夹爪 loss        : {grip_loss:.6f}")
    print(f"总均值 loss      : {total_loss:.6f}")

    # 如果想可视化每个时间步或 batch 的 loss
    loss_per_t = reduce(loss_map, "b t d -> t", "mean")
    print("\n逐时间步平均 loss:")
    print(loss_per_t)

    loss_per_b = reduce(loss_map, "b t d -> b", "mean")
    print("\n逐 batch 平均 loss:")
    print(loss_per_b)