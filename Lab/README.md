# COMP3710
mport numpy as np
import matplotlib.pyplot as plt

def levy_c_curve(iterations=15):
    # 初始公理
    axiom = "F"
    rules = {"F": "+F--F+"}

    # 生成 L 系统字符串
    s = axiom
    for _ in range(iterations):
        new_s = ""
        for ch in s:
            if ch in rules:
                new_s += rules[ch]
            else:
                new_s += ch
        s = new_s

    # 解释 L 系统
    angle = np.pi / 4  # 45°
    x, y = [0], [0]    # 起点
    direction = 0      # 当前方向（弧度，初始向右）

    for ch in s:
        if ch == "F":
            # 向前画一条线
            x_new = x[-1] + np.cos(direction)
            y_new = y[-1] + np.sin(direction)
            x.append(x_new)
            y.append(y_new)
        elif ch == "+":
            direction += angle
        elif ch == "-":
            direction -= angle

    return np.array(x), np.array(y)

# 示例：画 15 次迭代
x, y = levy_c_curve(iterations=15)
plt.figure(figsize=(8, 8))
plt.plot(x, y, color="blue", linewidth=0.5)
plt.axis("equal")
plt.axis("off")
plt.show()把这个代码改成使用pytorch
ChatGPT 说：

可以，把你这个 L 系统 Lévy C 曲线改成 PyTorch 版本，利用张量加速和 GPU。下面是改写后的示例：

import torch
import matplotlib.pyplot as plt

# 设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dtype = torch.float32

def levy_c_curve_torch(iterations=15):
    # ======================
    # 生成 L 系统字符串
    # ======================
    axiom = "F"
    rules = {"F": "+F--F+"}

    s = axiom
    for _ in range(iterations):
        new_s = ""
        for ch in s:
            if ch in rules:
                new_s += rules[ch]
            else:
                new_s += ch
        s = new_s

    # ======================
    # 解释 L 系统
    # ======================
    angle = torch.tensor(torch.pi / 4, device=device, dtype=dtype)  # 45°
    x = [torch.tensor(0.0, device=device, dtype=dtype)]
    y = [torch.tensor(0.0, device=device, dtype=dtype)]
    direction = torch.tensor(0.0, device=device, dtype=dtype)       # 初始方向向右

    for ch in s:
        if ch == "F":
            x_new = x[-1] + torch.cos(direction)
            y_new = y[-1] + torch.sin(direction)
            x.append(x_new)
            y.append(y_new)
        elif ch == "+":
            direction += angle
        elif ch == "-":
            direction -= angle

    # 转为张量
    x_t = torch.stack(x)
    y_t = torch.stack(y)
    return x_t, y_t

# ======================
# 绘图示例
# ======================
x, y = levy_c_curve_torch(iterations=15)
plt.figure(figsize=(8, 8))
plt.plot(x.cpu().numpy(), y.cpu().numpy(), color="blue", linewidth=0.5)
plt.axis("equal")
plt.axis("off")
plt.show()

✅ 改动说明

用 torch.tensor 替代 float。

所有计算都放在 device 上（CPU 或 GPU）。

最后绘图时，将张量 .cpu().numpy() 转为 NumPy 画图。

保留了原本 L 系统逻辑，迭代生成完整 Lévy C 曲线。