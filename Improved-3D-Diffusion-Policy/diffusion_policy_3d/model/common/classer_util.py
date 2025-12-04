import torch
import time

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