# COMP3710
顺时针旋转矩阵
ChatGPT 说：
已思考 10s

二维顺时针旋转矩阵（角度 
𝜃
θ）：

𝑅
cw
(
𝜃
)
=
[
cos
⁡
𝜃
	
 
sin
⁡
𝜃


−
sin
⁡
𝜃
	
 
cos
⁡
𝜃
]
R
cw
	​

(θ)=[
cosθ
−sinθ
	​

 sinθ
 cosθ
	​

]

这是“逆时针矩阵”的负角形式：
𝑅
cw
(
𝜃
)
=
𝑅
ccw
(
−
𝜃
)
R
cw
	​

(θ)=R
ccw
	​

(−θ)。

逆时针矩阵为：
𝑅
ccw
(
𝜃
)
=
[
cos
⁡
𝜃
	
−
sin
⁡
𝜃


sin
⁡
𝜃
	
cos
⁡
𝜃
]
R
ccw
	​

(θ)=[
cosθ
sinθ
	​

−sinθ
cosθ
	​

]。

如何使用：

若向量是列向量 
𝑣
=
[
𝑥
 
𝑦
]
𝑇
v=[x y]
T
：
𝑣
′
=
𝑅
cw
(
𝜃
)
 
𝑣
v
′
=R
cw
	​

(θ)v

若向量是行向量（常见于张量按行存点）：
𝑣
′
=
𝑣
 
𝑅
cw
(
𝜃
)
v
′
=vR
cw
	​

(θ)；批量点 
𝑉
∈
𝑅
𝑁
×
2
V∈R
N×2
 用 
𝑉
′
=
𝑉
 
𝑅
cw
(
𝜃
)
V
′
=VR
cw
	​

(θ)。

快速校验（
𝜃
=
90
∘
θ=90
∘
）

cos
⁡
90
∘
=
0
,
 
sin
⁡
90
∘
=
1
⇒
𝑅
cw
=
[
0
	
1


−
1
	
0
]
cos90
∘
=0, sin90
∘
=1⇒R
cw
	​

=[
0
−1
	​

1
0
	​

]。
把 
(
1
,
0
)
(1,0) 顺时针旋转 90°：
[
0
	
1


−
1
	
0
]
 ⁣
[
1


0
]
=
[
0


−
1
]
[
0
−1
	​

1
0
	​

][
1
0
	​

]=[
0
−1
	​

] ✅

PyTorch 代码（批量点按行存放）：

import torch

theta_deg = 45.0
theta = torch.deg2rad(torch.tensor(theta_deg))
c, s = torch.cos(theta), torch.sin(theta)

R_cw = torch.stack([
    torch.stack([ c,  s]),
    torch.stack([-s,  c]),
])  # shape (2,2)

# 假设点集 V 形状为 (N, 2)，每行一个 (x,y)
# 顺时针旋转：右乘 R_cw
V_rot = V @ R_cw  # (N,2)


**二维齐次坐标（含平移时常用）**的顺时针旋转矩阵：

[
cos
⁡
𝜃
	
 
sin
⁡
𝜃
	
0


−
sin
⁡
𝜃
	
 
cos
⁡
𝜃
	
0


0
	
0
	
1
]
	​

cosθ
−sinθ
0
	​

 sinθ
 cosθ
0
	​

0
0
1
	​

	​

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
怎么判定z发散
ChatGPT 说：

好问题 👍 我来帮你整理一下 怎么判定迭代序列 
𝑧
𝑛
+
1
=
𝑧
𝑛
2
+
𝑐
z
n+1
	​

=z
n
2
	​

+c 是否发散。

1. 理论上的判定方法

在 Mandelbrot/Julia 集的迭代中，如果迭代数列 
{
𝑧
𝑛
}
{z
n
	​

} 越来越大，我们就说它“发散”。

数学上有一个重要结论：

如果 
∣
𝑧
𝑛
∣
>
2
，那么
{
𝑧
𝑛
}
一定会发散到无穷大。
如果 ∣z
n
	​

∣>2，那么{z
n
	​

}一定会发散到无穷大。

👉 换句话说，只要某一步迭代时 
∣
𝑧
∣
∣z∣ 超过 2，就可以认为它会发散。

2. 实际计算中的做法

在程序里不可能无限迭代，所以我们设定：

最大迭代次数（例如 200 次）

逃逸半径（escape radius，一般取 2 或更大，比如 4，为了数值安全）

算法：

从初始值 
𝑧
0
z
0
	​

 开始迭代。

每次计算 
∣
𝑧
𝑛
∣
∣z
n
	​

∣，如果 
∣
𝑧
𝑛
∣
>
2
∣z
n
	​

∣>2，则认为 发散，停止迭代。

如果在最大迭代次数内都没有超过 2，就认为它“可能收敛”（属于集合内）。

3. 代码中的体现（Mandelbrot/Julia 通用）

比如你上面写的：

for i in range(200):
    zs = zs*zs + c   # 迭代公式
    mask = torch.abs(zs) < 4.0  # 判定是否发散（这里用了 4 而不是 2）
    ns += mask


这里 torch.abs(zs) < 4.0 就是判定条件：

如果模长小于 4，就认为还没发散；

超过 4，就认为一定会发散（因为超过 2 就保证发散，取 4 更宽松）。

✅ 总结

数学结论：
∣
𝑧
𝑛
∣
>
2
  
⟹
  
∣z
n
	​

∣>2⟹ 发散。

程序实现：设定 escape radius（通常 2 或 4）+ 最大迭代次数。

如果在迭代过程中超过 escape radius → 发散；否则视为收敛或边界。