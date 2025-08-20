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
or _ in range(iterations):
        # 对当前所有线段并行变换，长度缩为 1/√2
        Vp = (V @ R_plus.T)  * 1.0 / math.sqrt(2.0)   # 左转段
        Vm = (V @ R_minus.T) * 1.0 / math.sqrt(2.0)   # 右转段

        # 替换顺序：每条旧段 → [左转段, 右转段]
        V = torch.stack([Vp, Vm], dim=1).reshape(-1, 2)  # (2N, 2)

    # 从位移向量得到顶点坐标：前缀和（并行内核）
    P = torch.cat([torch.zeros(1, 2, device=device), V.cumsum(dim=0)], dim=0)  # (2^it + 1, 2)
    return P[:,0], P[:,1]  # x, y有没有别的简单的方法，要体现并行
ChatGPT 说：
已思考 54s

当然有！给你两种更简单且确实用到并行/向量化的做法：

方法一：复数 + Kronecker 积（最简洁）

把每条线段看作复数位移 
𝑧
z，每轮替换就是把 
𝑧
z 同时乘以两个常数

𝛼
=
1
2
𝑒
+
𝑖
𝜋
/
4
α=
2
	​

1
	​

e
+iπ/4
、
𝛽
=
1
2
𝑒
−
𝑖
𝜋
/
4
β=
2
	​

1
	​

e
−iπ/4
 并交替排列。
用 torch.kron 一行就能把“所有线段”批量替换，最后 cumsum 得到顶点。

import torch, math

def levy_c_curve_kron(iters=15, device='cuda'):
    # 初始只有一条水平线段：位移 1+0j
    z = torch.tensor([1+0j], device=device, dtype=torch.complex64)

    alpha = (1/math.sqrt(2)) * torch.exp(1j*torch.pi/4)   # 左转 +45°
    beta  = (1/math.sqrt(2)) * torch.exp(-1j*torch.pi/4)  # 右转 -45°
    base  = torch.tensor([alpha, beta], device=device, dtype=torch.complex64)

    for _ in range(iters):
        z = torch.kron(z, base)   # 并行“每段→两段”的替换：长度翻倍

    p = torch.cat([torch.zeros(1, device=device, dtype=z.dtype), torch.cumsum(z, dim=0)])  # 顶点
    x, y = p.real.float(), p.imag.float()
    return x, y


为何并行？ kron、exp、cumsum 都是大张量/向量化运算，GPU/多核可一次性计算海量元素，比逐字符海龟法高效得多。

方法二：IFS “混沌游戏”批量采样（超并行）

Lévy C 曲线的 IFS（迭代函数系统）可写为（复数形式）：

𝑤
1
(
𝑧
)
=
1
2
𝑒
+
𝑖
𝜋
/
4
𝑧
,
𝑤
2
(
𝑧
)
=
1
−
1
2
𝑒
−
𝑖
𝜋
/
4
𝑧
.
w
1
	​

(z)=
2
	​

1
	​

e
+iπ/4
z,w
2
	​

(z)=1−
2
	​

1
	​

e
−iπ/4
z.

我们一次生成 很多点（比如十万），每一步对全部点并行地随机选择 
𝑤
1
w
1
	​

 或 
𝑤
2
w
2
	​

 应用，多次迭代后点云就收敛到曲线的吸引子。

import torch, math

def levy_c_curve_chaos(n_points=200_000, n_iters=20, device='cuda'):
    a = (1/math.sqrt(2)) * torch.exp(1j*torch.pi/4)
    b = (1/math.sqrt(2)) * torch.exp(-1j*torch.pi/4)

    z = torch.zeros(n_points, dtype=torch.complex64, device=device)
    # 预热几步丢弃可更稳定，这里直接迭代
    for _ in range(n_iters):
        choice = torch.randint(0, 2, (n_points,), device=device)  # 0/1 随机
        # 选择 w1 或 w2（按元素并行）
        z = torch.where(choice.bool(), a*z, 1 - b*z)

    return z.real.float(), z.imag.float()  # 这是点云采样（散点），不是按路径顺序的折线


为何并行？ 每一步都是对长度为 n_points 的向量做逐元素选择和复乘，全部在 GPU 上批量进行；n_points 越大并行越明显。