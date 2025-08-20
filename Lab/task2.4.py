import torch
import numpy as np
import matplotlib.pyplot as plt

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ====== Zoom 参数（自己改这三项就能放大/移动）======
cx, cy = -0.745, 0.112  # 例：海马谷 Seahorse Valley
span_x = 1                                      # 实轴半宽，越小=放大越深
span_y = span_x * (2.6 / 3.0)                   # 保持原始宽高比
step   = span_x / 100.0                        # 越小越清晰，但更慢

Y, X = np.mgrid[cy - span_y : cy + span_y : step,
                cx - span_x : cx + span_x : step]

# ====== 载入到 PyTorch（直接到目标设备）======
x = torch.tensor(X, device=device, dtype=torch.float32)
y = torch.tensor(Y, device=device, dtype=torch.float32)

# ====== Julia 集修改处 ======
# 固定常量 c（可以改不同的 c 看到不同形状）
c = torch.complex(torch.tensor(-0.8,  device=device, dtype=torch.float32),
                  torch.tensor( 0.156, device=device, dtype=torch.float32))

# 初始 z：网格
zs = torch.complex(x, y)

# 计数（用整型或浮点更合适，别用复数）
ns = torch.zeros(zs.shape, device=device, dtype=torch.int16)

# ====== 迭代（Julia: z <- z^2 + c）======
for _ in range(200):
    zs = zs * zs + c
    # 用平方模避免开方：|z|^2 <= 4 等价于 |z| <= 2
    not_diverged = (zs.real * zs.real + zs.imag * zs.imag) <= 4.0
    ns += not_diverged.to(ns.dtype)

# ====== 上色（沿用你的 NumPy 版）======
def processFractal(a):
    # Display an array of iteration counts as a colorful picture of a fractal.
    a_cyclic = (6.28 * a / 20.0).reshape(list(a.shape) + [1])
    img = np.concatenate([10 + 20 * np.cos(a_cyclic),
                          30 + 50 * np.sin(a_cyclic),
                          155 - 80 * np.cos(a_cyclic)], 2)
    img[a == a.max()] = 0
    a = np.uint8(np.clip(img, 0, 255))
    return a

plt.figure(figsize=(16, 10))
plt.imshow(processFractal(ns.cpu().numpy()))
plt.tight_layout(pad=0)
plt.show()
