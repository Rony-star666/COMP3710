import torch
import matplotlib.pyplot as plt

# 设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dtype = torch.float32

def levy_c_curve_torch(iterations=15):
    # 生成 L 系统字符串
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

    # 解释 L 系统
    angle = torch.tensor(torch.pi / 4, device=device, dtype=dtype)  # 顺时针45°
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
            direction -= angle # 顺时针45°
        elif ch == "-":
            direction += angle # 逆时针45°

    # 转为张量
    x_t = torch.stack(x)
    y_t = torch.stack(y)
    return x_t, y_t

# 绘图示例
x, y = levy_c_curve_torch(iterations=15)
plt.figure(figsize=(8, 8))
plt.plot(x.cpu().numpy(), y.cpu().numpy(), color="blue", linewidth=0.5)
plt.axis("equal")
plt.axis("off")
plt.show()