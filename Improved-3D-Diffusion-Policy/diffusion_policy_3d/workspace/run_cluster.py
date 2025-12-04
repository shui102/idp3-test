if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.absolute())
    sys.path.append(ROOT_DIR)

import hydra
import torch
import zarr
import numpy as np
import time
from tqdm import tqdm
from omegaconf import OmegaConf
from diffusion_policy_3d.workspace.balancer_workspace import BalancerWorkspace
from diffusion_policy_3d.common.pytorch_util import dict_apply

# ==========================================
# GPU K-Means 实现
# ==========================================
def kmeans_pytorch(X, k=100, n_iter=30, batch_size=50000, tol=1e-4, verbose=True):
    """
    X: (N, d) Tensor on GPU
    """
    device = X.device
    N, d = X.shape
    
    # 随机初始化中心
    idx = torch.randperm(N, device=device)[:k]
    centers = X[idx].clone()

    for it in range(n_iter):
        t0 = time.time()
        new_centers_sum = torch.zeros((k, d), device=device)
        new_centers_count = torch.zeros(k, device=device)

        # 分批计算距离，防止显存爆炸
        for i in range(0, N, batch_size):
            xb = X[i:i + batch_size]
            dist = torch.cdist(xb, centers)
            labels = torch.argmin(dist, dim=1)
            
            for c in range(k):
                mask = (labels == c)
                if mask.any():
                    new_centers_sum[c] += xb[mask].sum(dim=0)
                    new_centers_count[c] += mask.sum()

        # 更新中心
        updated = new_centers_sum / new_centers_count.unsqueeze(1).clamp(min=1)
        shift = torch.norm(centers - updated, dim=1).mean()
        centers = updated

        if verbose:
            print(f"[Iter {it+1}] shift={shift:.6f}, time={time.time()-t0:.3f}s")
        if shift < tol:
            break

    # 最后再算一次所有数据的 label
    final_labels = torch.empty(N, dtype=torch.long, device=device)
    for i in range(0, N, batch_size):
        xb = X[i:i + batch_size]
        dist = torch.cdist(xb, centers)
        final_labels[i:i + batch_size] = torch.argmin(dist, dim=1)

    cluster_density = torch.bincount(final_labels, minlength=k).float()
    return centers, final_labels, cluster_density

# ==========================================
# 核心逻辑
# ==========================================
@hydra.main(
    config_path="/home/shui/idp3_test/Improved-3D-Diffusion-Policy/Improved-3D-Diffusion-Policy/diffusion_policy_3d/config", 
    config_name="controlnet.yaml", 
    version_base=None
)
def main(cfg):
    device = torch.device(cfg.training.device)
    
    # 1. 实例化 BalancerWorkspace
    print(f"🔄 Instantiating BalancerWorkspace...")
    workspace = BalancerWorkspace(cfg)
    
    # 2. 加载 Checkpoint
    # 请确认这个路径是否正确
    ckpt_path = pathlib.Path("/home/shui/idp3_test/Improved-3D-Diffusion-Policy/Improved-3D-Diffusion-Policy/data/outputs/-controlnet-DMP_dualarm_augment_stage1_12011325_seed0/checkpoints/latest.ckpt")    
    print(f"📥 Loading checkpoint from: {ckpt_path}")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    
    workspace.load_checkpoint(path=ckpt_path)
    
    # 3. 提取 Policy 和 Normalizer
    # 根据配置决定使用 ema_model 还是 model
    if cfg.stage1.training.use_ema and workspace.ema_model is not None:
        policy = workspace.ema_model
        print("✅ Using EMA Model")
    else:
        policy = workspace.model
        print("✅ Using Standard Model")
        
    policy.eval()
    policy.to(device)
    
    # 获取 Encoder
    encoder = policy.obs_encoder_stage1
    # 获取 Normalizer (至关重要)
    normalizer = policy.normalizer
    
    print("🔎 Encoder found:", type(encoder))

# ... 前面的代码不变 ...

    # 4. 打开 Zarr 数据
    zarr_path = "/home/shui/DMP_gen/final_merged_dataset/merged_dataset_total.zarr"
    print(f"📂 Opening Zarr: {zarr_path}")
    root = zarr.open(zarr_path, mode='r')
    
    # [修改点 1] 预先获取所有需要的 Array
    pc_array = root['data']['point_cloud']
    
    # 尝试获取 state 和 action
    state_array = root['data']['state'] if 'state' in root['data'] else None
    action_array = root['data']['action'] if 'action' in root['data'] else None
    
    total_len = pc_array.shape[0]
    print(f"📊 Total frames: {total_len}")
    if state_array is not None: print(f"   State shape: {state_array.shape}")
    if action_array is not None: print(f"   Action shape: {action_array.shape}")

    # 5. 推理 Loop
    batch_size = 64
    embeddings_list = []
    
    print("🚀 Starting Encoding...")
    with torch.no_grad():
        for i in tqdm(range(0, total_len, batch_size)):
            # --- A. 读取数据 ---
            # 1. Point Cloud
            batch_pc = torch.from_numpy(pc_array[i : i + batch_size]).float().to(device)
            
            # 2. State (如果有) -> 拆分为 agent_pos / agent_rot
            # [关键] 这里需要你确认 dataset 的 state 是怎么定义的
            # 假设: state (N, 14) -> pos (N, 10) + rot (N, 4) 或者其他切分方式
            # 下面是一个示例，请根据你的 shape_meta 修改切片索引
            if state_array is not None:
                batch_state = torch.from_numpy(state_array[i : i + batch_size]).float().to(device)
                
                # 示例切分：假设前3维是 pos，后6维是 rot (根据你的 shape_meta 调整)
                # 如果你的 normalizer 里面有 agent_pos 和 agent_rot 的统计信息
                # 你必须确保这里的数据维度和 key 能对上
                agent_pos = batch_state[:, :]  # [请修改这里]
                # agent_rot = batch_state[:, 3:9] # [请修改这里]
                
                # 如果 state 维度不够，或者对应不上，你可以用全0补全剩余维度
                # agent_pos = torch.cat([agent_pos, torch.zeros(...)], dim=-1)
            else:
                # 如果 zarr 里没 state，只能用 dummy
                agent_pos = None 
                agent_rot = None


            # --- B. 构造 Obs Dict ---
            obs_dict = {
                'point_cloud': batch_pc,
            }
            
            # 将读取到的 state 塞进去
            if agent_pos is not None: obs_dict['agent_pos'] = agent_pos
            # if agent_rot is not None: obs_dict['agent_rot'] = agent_rot

            # --- C. 兜底逻辑 (Dummy) ---
            # 如果 Normalizer 期待某些 key (如 agent_pos)，但 Zarr 里没有或者没读到
            # 必须填补 Dummy 数据，否则 normalizer 会报错
            for key in normalizer.params_dict.keys():
                if key not in obs_dict and key != 'action':
                    # 获取该 key 在 normalizer 中记录的 mean 的形状
                    # shape 通常是 (1, D)
                    param_shape = normalizer.params_dict[key]['mean'].shape
                    # 构造 (B, D) 的全0数据
                    dummy = torch.zeros((batch_pc.shape[0], *param_shape[1:]), device=device)
                    obs_dict[key] = dummy

            # --- D. 归一化 ---
            # normalizer.normalize(dict) 只会处理 dict 中存在的 key
            # 并且会忽略 'action' (因为 action 通常在 normalizer['action'] 里单独处理)
            nobs = normalizer.normalize(obs_dict)
            
            # 处理颜色
            if hasattr(policy, 'use_pc_color') and policy.use_pc_color:
                nobs['point_cloud'][..., 3:] /= 1.0
            else:
                nobs['point_cloud'] = nobs['point_cloud'][..., :]

            # --- E. Encoder 推理 ---
            encoded = encoder(nobs)
            
            if len(encoded.shape) > 2:
                encoded = encoded.reshape(encoded.shape[0], -1)
            embeddings_list.append(encoded)


    # 合并所有 batch
    X = torch.cat(embeddings_list, dim=0)
    print(f"✅ Features extracted. Shape: {X.shape}")

    # 6. K-Means 聚类
    K = 10 # 聚类数量
    print(f"🧩 Running K-Means (k={K})...")
    centers, labels, density = kmeans_pytorch(X, k=K, n_iter=1000)

    # 7. 保存结果
    output_file = "clustering_results_1130_DMP_augmented.pt"
    save_dict = {
        "centers": centers.cpu(),
        "labels": labels.cpu(),
        "density": density.cpu()
    }
    torch.save(save_dict, output_file)
    
    print("-" * 30)
    print(f"💾 Results saved to {output_file}")
    print(f"   Top 10 Cluster Counts: {torch.topk(density, 10).values}")
    print("-" * 30)
    
    # 额外：这里可以给出一个提示，如何在 BalancerWorkspace 中加载它
    print("To use in BalancerWorkspace:")
    print("self.encoder_map = torch.load('clustering_results.pt')['centers']")
    print("self.weights_map = ... (calculate based on density)")

if __name__ == "__main__":
    main()