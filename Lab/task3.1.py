import torch
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dtype  = torch.float32

def levy_ifs_torch_no_burn(n_points=300000, seed=0, a=(0.0,0.0), b=(1.0,0.0)):
    """
    Lévy C 曲线 IFS
    """
    torch.manual_seed(seed)

    # 常量矩阵
    r = 1.0 / torch.sqrt(torch.tensor(2.0, device=device, dtype=dtype))
    c = torch.cos(torch.tensor(torch.pi/4, device=device, dtype=dtype))
    s = torch.sin(torch.tensor(torch.pi/4, device=device, dtype=dtype))

    M1 = r * torch.tensor([[ c, -s],
                           [ s,  c]], device=device, dtype=dtype)  # +45°
    M2 = r * torch.tensor([[ c,  s],
                           [-s,  c]], device=device, dtype=dtype)  # -45°
    t2 = torch.tensor([1.0, 0.0], device=device, dtype=dtype)

    # 随机选择映射
    random = torch.rand(n_points, device=device)
    choices = random < 0.5
    # 初始化点
    z = torch.zeros((n_points, 2), device=device, dtype=dtype)

    for i in range(1, n_points):
        if choices[i]:
            z[i] = (M1 @ z[i-1])
        else:
            z[i] = t2 + (M2 @ z[i-1])

    # 仿射到线段 a->b
    a_t = torch.tensor(a, device=device, dtype=dtype)
    b_t = torch.tensor(b, device=device, dtype=dtype)
    u   = b_t - a_t
    perp= torch.stack([-u[1], u[0]])  # 旋转 90°

    x = z[:, :1]
    y = z[:, 1:]
    world = a_t + x * u + y * perp

    return world


# 演示
pts = levy_ifs_torch_no_burn(n_points=100_000, seed=42)
pts_np = pts.detach().cpu().numpy()
plt.figure(figsize=(7,7))
plt.scatter(pts_np[:,0], pts_np[:,1], s=0.2, marker='.', linewidths=0)
plt.axis('equal'); plt.axis('off'); plt.tight_layout(); plt.show()
