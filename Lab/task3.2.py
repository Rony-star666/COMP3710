import math
import torch
import matplotlib.pyplot as plt

# 设备 & 精度
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dtype  = torch.float32

def levy_ifs_torch_no_burn_parallel(
    n_points=200_000,
    n_iters=24,
    seed=0,
    a=(0.0, 0.0),
    b=(1.0, 0.0),
    device=device
):
    """
    并行 IFS 采样 Lévy C 曲线。
    - n_points: 并行点数（越大越平滑，显存开销也越大）
    - n_iters : 迭代步数（可适当加大，以弥补无 burn-in 的初期过渡）
    - seed    : 随机种子（复现用）
    - a, b    : 将结果仿射到线段 a->b（缺省为单位线段）
    返回: (N,2) 的张量，每行一个 (x,y) 点
    """
    torch.manual_seed(seed)

    # 仿射映射系数：±45° 旋转并缩放 1/sqrt(2)
    r = 1.0 / math.sqrt(2.0)
    c = math.cos(math.pi / 4)
    s = math.sin(math.pi / 4)

    # 2×2 旋转缩放矩阵（行向量右乘需要转置：V @ M.T）
    M1 = r * torch.tensor([[ c, -s],
                           [ s,  c]], device=device, dtype=dtype)  # +45°
    M2 = r * torch.tensor([[ c,  s],
                           [-s,  c]], device=device, dtype=dtype)  # -45°
    t2 = torch.tensor([1.0, 0.0], device=device, dtype=dtype)      # 平移向量

    # 批量点初始化 (N,2)
    z = torch.zeros((n_points, 2), device=device, dtype=dtype)

    # 向量化迭代：每一步对所有点并行选择 w1 / w2
    for _ in range(n_iters):
        mask = (torch.rand(n_points, device=device) < 0.5)      # (N,)
        z = torch.where(mask[:, None],
                        z @ M1.T,                              # w1(z) = M1 z
                        t2 + z @ M2.T)                         # w2(z) = t2 + M2 z

    # 将 (x,y) 仿射到线段 a->b 的坐标系：基向量 u 与其垂直向量 perp
    a_t = torch.tensor(a, device=device, dtype=dtype)           # (2,)
    b_t = torch.tensor(b, device=device, dtype=dtype)           # (2,)
    u   = b_t - a_t                                             # (2,)
    perp= torch.stack([-u[1], u[0]])                            # (2,) 旋转90°

    x = z[:, :1]                                                # (N,1)
    y = z[:, 1:]                                                # (N,1)
    world = a_t + x * u + y * perp                              # (N,2)

    return world

if __name__ == "__main__":
    pts = levy_ifs_torch_no_burn_parallel(
        n_points=200_000,
        n_iters=24,
        seed=42,
        a=(0.0, 0.0),
        b=(1.0, 0.0),
        device=device
    )
    pts_np = pts.detach().cpu().numpy()

    plt.figure(figsize=(7, 7))
    plt.scatter(pts_np[:, 0], pts_np[:, 1], s=0.2, marker='.', linewidths=0, alpha=0.9)
    plt.axis('equal'); plt.axis('off'); plt.tight_layout(); plt.show()
