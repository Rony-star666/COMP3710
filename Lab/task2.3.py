import torch
import numpy as np
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ====== Zoom 参数（自己改这三项就能放大/移动）======
cx, cy = -0.745, 0.112  # 例：海马谷 Seahorse Valley 
#Elephant Valley（大象谷）：(-0.745, 0.112) 经典心形尖端附近：(-0.75, 0.0)
span_x = 1                                      # 实轴半宽，越小=放大越深
# 维持原始宽高比例（原来是宽3.0，高2.6）
span_y = span_x * (2.6 / 3.0)

# 步长建议：步长 ≈ span_x / 1000 约等于横向 ~2000 像素
# 步长越小越清晰，但计算更慢；按机器性能调整
step = span_x / 1000.0

Y, X = np.mgrid[cy - span_y : cy + span_y : step,
                cx - span_x : cx + span_x : step]


# load into PyTorch tensors
x = torch.Tensor(X)
y = torch.Tensor(Y)
z = torch.complex(x, y) #important!
zs = z.clone() #Updated!
ns = torch.zeros_like(z)

# transfer to the GPU device
z = z.to(device)
zs = zs.to(device)
ns = ns.to(device)

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
