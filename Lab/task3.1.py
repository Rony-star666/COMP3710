import torch, math
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def levy_c_curve_vectorized(iterations=15):
    # 起始只有一条水平线段 v = (1, 0)
    V = torch.tensor([[1.0, 0.0]], device=device)  # (N,2) 表示 N 条线段的位移向量
    s2 = 1.0 / math.sqrt(2.0)
    ang = math.pi / 4

    # 旋转矩阵（±45°）
    c, s = math.cos(ang), math.sin(ang)
    R_plus  = torch.tensor([[ c, -s],
                            [ s,  c]], device=device)   # 逆时针 +45°
    R_minus = torch.tensor([[ c,  s],
                            [-s,  c]], device=device)   # 顺时针 -45°

    for _ in range(iterations):
        # 对当前所有线段并行变换，长度缩为 1/√2
        Vp = (V @ R_plus.T)  * s2   # 左转段
        Vm = (V @ R_minus.T) * s2   # 右转段

        # 替换顺序：每条旧段 → [左转段, 右转段]
        V = torch.stack([Vp, Vm], dim=1).reshape(-1, 2)  # (2N, 2)

    # 从位移向量得到顶点坐标：前缀和（并行内核）
    P = torch.cat([torch.zeros(1, 2, device=device), V.cumsum(dim=0)], dim=0)  # (2^it + 1, 2)
    return P[:,0], P[:,1]  # x, y

# 绘图示例
x, y = levy_c_curve_vectorized(iterations=15)
plt.figure(figsize=(8, 8))
plt.plot(x.cpu().numpy(), y.cpu().numpy(), linewidth=0.5)
plt.axis("equal"); plt.axis("off")
plt.show()
