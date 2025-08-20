import torch
import numpy as np
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 1) 用 Torch 生成网格（直接在 GPU）
y = torch.arange(-1.3, 1.3, 0.005, device=device, dtype=torch.float32)
x = torch.arange(-2.0, 1.0, 0.005, device=device, dtype=torch.float32)
Y, X = torch.meshgrid(y, x, indexing='ij')  # 与 mgrid 维度顺序一致

# 2) 直接用 Torch 张量构造复数 z
z  = torch.complex(X, Y)      # 常量 c
zs = z.clone()                # 迭代变量 z_n

# 3) 计数用整型/浮点，别用复数
ns = torch.zeros(z.shape, device=device, dtype=torch.int16)

#Mandelbrot Set
for i in range(200):
#Compute the new values of z: z^2 + x
    zs_ = zs*zs + z
#Have we diverged with this new value?
    not_diverged = torch.abs(zs_) < 4.0
#Update variables to compute
    ns += not_diverged
    zs = zs_

#plot
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(16,10))
def processFractal(a):
#Display an array of iteration counts as a colorful picture of a fractal.
    a_cyclic = (6.28*a/20.0).reshape(list(a.shape)+[1])
    img = np.concatenate([10+20*np.cos(a_cyclic),
    30+50*np.sin(a_cyclic),
    155-80*np.cos(a_cyclic)], 2)
    img[a == a.max()] = 0
    a = img
    a = np.uint8(np.clip(a, 0, 255))
    return a

plt.imshow(processFractal(ns.cpu().numpy()))
plt.tight_layout(pad=0)
plt.show()
