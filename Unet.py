import os, random, time
from glob import glob
from collections import Counter
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import matplotlib.pyplot as plt

# -----------------------------
# 0. 环境 & 设备
# -----------------------------
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.benchmark = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
amp_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

# -----------------------------
# 1) 数据集：稳配对 + 可多分类
# -----------------------------
def stem(fname: str) -> str:
    name = fname
    if name.startswith("case_"): name = name[len("case_"):]
    if name.startswith("seg_"):  name = name[len("seg_"):]
    base, ext = os.path.splitext(name)
    while ext.lower() in [".png", ".nii", ".gz"]:
        name = base
        base, ext = os.path.splitext(name)
    return name

class OasisSegDataset(Dataset):
    """
    使用共同 stem（去掉前缀 + 连环扩展名）做配对。
    例如：case_402_slice_0.nii.png  <->  seg_402_slice_0.png
    """
    def __init__(self, img_dir, msk_dir, size=(128, 128), augment=False):
        self.img_dir, self.msk_dir = img_dir, msk_dir
        self.size, self.augment = size, augment

        img_files = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(".png")])
        msk_files = sorted([f for f in os.listdir(msk_dir) if f.lower().endswith(".png")])
        msk_by_stem = {stem(f): f for f in msk_files}

        self.pairs = []
        dropped = 0
        for f in img_files:
            s = stem(f)
            if s in msk_by_stem:
                self.pairs.append((f, msk_by_stem[s]))
            else:
                dropped += 1
        if dropped > 0:
            print(f"⚠️ 有 {dropped} 张图像未找到对应 mask，已跳过。样本数={len(self.pairs)}")

        self.tf_img = T.Compose([
            T.Resize(size, interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),                                   # [0,1]
            # MR 图像可以按需归一化到 0-1；若需要标准化，可在此添加
        ])
        self.tf_msk = T.Compose([
            T.Resize(size, interpolation=T.InterpolationMode.NEAREST),
        ])
        self.aug_ops = (T.RandomChoice([
            T.RandomHorizontalFlip(p=1.0),
            T.RandomVerticalFlip(p=1.0),
            T.RandomRotation(degrees=10),
        ], p=[1/3,1/3,1/3]) if augment else None)

    def __len__(self): return len(self.pairs)

    def __getitem__(self, idx):
        img_name, msk_name = self.pairs[idx]
        img = Image.open(os.path.join(self.img_dir, img_name)).convert("L")
        msk = Image.open(os.path.join(self.msk_dir, msk_name))

        if self.augment:
            seed = np.random.randint(0, 10_000)
            random.seed(seed); torch.manual_seed(seed)
            tform = random.choice(self.aug_ops.transforms)
            img = tform(img); msk = tform(msk)

        img = self.tf_img(img)                              # (1,H,W), float
        msk = self.tf_msk(msk)                              # PIL -> PIL resized
        msk = torch.from_numpy(np.array(msk, dtype=np.int64))  # (H,W) long (灰度标签)
        return img, msk

def probe_class_values(msk_dir, k=200):
    """从掩码目录抽样统计像素值，自动推断类别集合（忽略背景=0）。"""
    paths = sorted(glob(os.path.join(msk_dir, "*.png")))
    if not paths:
        return [0, 1]  # 兜底成二类
    pick = paths if len(paths) <= k else list(np.random.choice(paths, k, replace=False))
    counter = Counter()
    for p in pick:
        arr = np.array(Image.open(p))
        # 采样部分像素以提速
        flat = arr.flatten()
        if flat.size > 10000:
            flat = np.random.choice(flat, 10000, replace=False)
        counter.update(flat.tolist())
    values = sorted(v for v in counter.keys())
    # 常见 OASIS 标注是少数离散值；若值数量太多，则退化为二类（非0=前景）
    if len(values) > 10:
        return [0, 1]
    return values if 0 in values else [0] + values

class ToCategorical(Dataset):
    """
    将灰度 mask 离散到给定类值集合 -> 类索引（0..C-1）。
    class_values: 例如 [0, 63, 127, 191, 255]，其中 0 视作背景。
    """
    def __init__(self, base: Dataset, class_values):
        self.base = base
        self.class_values = np.array(sorted(class_values), dtype=np.float32)

    def __len__(self): return len(self.base)

    def __getitem__(self, idx):
        img, msk = self.base[idx]
        m = msk.numpy().astype(np.float32)
        if len(self.class_values) == 2 and set(self.class_values)=={0.0,1.0}:
            # 标准二值：非0前景；保持 0 背景
            m = (m > 0).astype(np.int64)
            return img, torch.from_numpy(m)
        # 多类：映射到最近的类值索引
        cv = self.class_values.reshape(1, -1)               # (1,C)
        dist = np.abs(m.reshape(-1,1) - cv)                 # (HW,C)
        cls  = dist.argmin(axis=1).astype(np.int64).reshape(m.shape)
        return img, torch.from_numpy(cls)

def one_hot(labels: torch.Tensor, num_classes: int):
    return F.one_hot(labels.long(), num_classes=num_classes).permute(0,3,1,2).float()

# -----------------------------
# 2) UNet
# -----------------------------
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.block(x)

class UNet(nn.Module):
    def __init__(self, in_ch=1, num_classes=2, base=64):
        super().__init__()
        self.down1 = DoubleConv(in_ch, base);   self.pool1 = nn.MaxPool2d(2)
        self.down2 = DoubleConv(base, base*2);  self.pool2 = nn.MaxPool2d(2)
        self.down3 = DoubleConv(base*2, base*4);self.pool3 = nn.MaxPool2d(2)
        self.down4 = DoubleConv(base*4, base*8);self.pool4 = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(base*8, base*16)

        self.up4 = nn.ConvTranspose2d(base*16, base*8, 2, stride=2)
        self.dec4 = DoubleConv(base*16, base*8)
        self.up3 = nn.ConvTranspose2d(base*8, base*4, 2, stride=2)
        self.dec3 = DoubleConv(base*8, base*4)
        self.up2 = nn.ConvTranspose2d(base*4, base*2, 2, stride=2)
        self.dec2 = DoubleConv(base*4, base*2)
        self.up1 = nn.ConvTranspose2d(base*2, base, 2, stride=2)
        self.dec1 = DoubleConv(base*2, base)

        self.out_conv = nn.Conv2d(base, num_classes, 1)

    def forward(self, x):
        d1 = self.down1(x); p1 = self.pool1(d1)
        d2 = self.down2(p1); p2 = self.pool2(d2)
        d3 = self.down3(p2); p3 = self.pool3(d3)
        d4 = self.down4(p3); p4 = self.pool4(d4)

        bn = self.bottleneck(p4)
        u4 = self.dec4(torch.cat([self.up4(bn), d4], 1))
        u3 = self.dec3(torch.cat([self.up3(u4), d3], 1))
        u2 = self.dec2(torch.cat([self.up2(u3), d2], 1))
        u1 = self.dec1(torch.cat([self.up1(u2), d1], 1))
        return self.out_conv(u1)  # logits (N,C,H,W)

# -----------------------------
# 3) Loss & Metric (one-hot)
# -----------------------------
class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0, ignore_bg=True):
        super().__init__(); self.smooth = smooth; self.ignore_bg = ignore_bg
    def forward(self, logits, targets_onehot):
        probs = torch.softmax(logits, dim=1)
        if self.ignore_bg and probs.size(1) > 1:
            probs, targets_onehot = probs[:,1:], targets_onehot[:,1:]
        dims = (0,2,3)
        inter = torch.sum(probs * targets_onehot, dims)
        denom = torch.sum(probs, dims) + torch.sum(targets_onehot, dims)
        dice = (2*inter + self.smooth) / (denom + self.smooth)
        return 1 - dice.mean()

@torch.no_grad()
def dice_per_class(logits, targets, num_classes, ignore_bg=True, eps=1e-7):
    probs = torch.softmax(logits, dim=1)
    preds = probs.argmax(1)
    pred_oh = one_hot(preds, num_classes).to(logits.dtype)
    tgt_oh  = one_hot(targets, num_classes).to(logits.dtype)
    if ignore_bg and num_classes > 1:
        pred_oh, tgt_oh = pred_oh[:,1:], tgt_oh[:,1:]
    inter = torch.sum(pred_oh * tgt_oh, (0,2,3))
    denom = torch.sum(pred_oh, (0,2,3)) + torch.sum(tgt_oh, (0,2,3))
    return ((2*inter + eps) / (denom + eps)).cpu().numpy()  # shape: (C or C-1,)

# -----------------------------
# 4) 训练 / 评估
# -----------------------------
def train_one_epoch(model, loader, opt, ce_loss, dice_loss, num_classes):
    model.train(); total=0.0; nb=0
    for imgs, msks in loader:
        imgs, msks = imgs.to(device), msks.to(device)
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=device.type=="cuda"):
            logits = model(imgs)
            ce  = ce_loss(logits, msks)
            dl  = dice_loss(logits, one_hot(msks, num_classes).to(logits.dtype))
            loss = ce + dl
        loss.backward(); opt.step()
        total += loss.item(); nb += 1
    return total/max(1,nb)

@torch.no_grad()
def evaluate(model, loader, num_classes, ignore_bg=True):
    model.eval(); all_dices=[]
    for imgs, msks in loader:
        imgs, msks = imgs.to(device), msks.to(device)
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=device.type=="cuda"):
            logits = model(imgs)
        all_dices.append(dice_per_class(logits, msks, num_classes, ignore_bg))
    all_dices = np.stack(all_dices, 0)           # (B, C' )
    mean_per_class = all_dices.mean(0)           # (C' )
    mean_dice = float(mean_per_class.mean())
    min_dice  = float(mean_per_class.min())
    return mean_dice, mean_per_class, min_dice

# -----------------------------
# 5) 可视化
# -----------------------------
@torch.no_grad()
def visualize_batch(model, loader, num_classes, out_path="vis.png", max_samples=6):
    model.eval()
    imgs, msks = next(iter(loader))
    imgs, msks = imgs.to(device), msks.to(device)
    preds = torch.softmax(model(imgs), 1).argmax(1)
    imgs, msks, preds = imgs.cpu().numpy(), msks.cpu().numpy(), preds.cpu().numpy()

    n = min(max_samples, imgs.shape[0])
    plt.figure(figsize=(10, 3*n))
    for i in range(n):
        plt.subplot(n,3,3*i+1); plt.imshow(imgs[i,0], cmap="gray"); plt.title("Image"); plt.axis("off")
        plt.subplot(n,3,3*i+2); plt.imshow(msks[i], cmap="tab20");  plt.title("GT");    plt.axis("off")
        plt.subplot(n,3,3*i+3); plt.imshow(preds[i], cmap="tab20"); plt.title("Pred");  plt.axis("off")
    plt.tight_layout(); plt.savefig(out_path, dpi=150); plt.close()

# -----------------------------
# 6) 主程序
# -----------------------------
def main():
    # 修改为你的本地路径
    base_dir = "/Users/ljh/Desktop/lab2/keras_png_slices_data"
    train_img = os.path.join(base_dir, "keras_png_slices_train")
    val_img   = os.path.join(base_dir, "keras_png_slices_validate")
    test_img  = os.path.join(base_dir, "keras_png_slices_test")
    train_msk = os.path.join(base_dir, "keras_png_slices_seg_train")
    val_msk   = os.path.join(base_dir, "keras_png_slices_seg_validate")
    test_msk  = os.path.join(base_dir, "keras_png_slices_seg_test")

    # 自动推断类别值（抽样统计）；如果你想手动指定，可写成 [0,63,127,191,255]
    class_values = probe_class_values(train_msk, k=200)
    # 若类别离散值太多（>10），会自动退化为二类 {0,1}
    print(f"📌 使用的类值集合: {class_values}")

    # 数据集 & DataLoader
    train_set = ToCategorical(OasisSegDataset(train_img, train_msk, (128,128), augment=True), class_values)
    val_set   = ToCategorical(OasisSegDataset(val_img,   val_msk,   (128,128), augment=False), class_values)
    test_set  = ToCategorical(OasisSegDataset(test_img,  test_msk,  (128,128), augment=False), class_values)

    # num_classes 与 one-hot 输出
    num_classes = len(class_values) if class_values != [0,1] else 2
    print(f"✅ num_classes = {num_classes}（分类 one-hot 输出）")

    use_pin_memory = (device.type == "cuda")
    train_loader = DataLoader(train_set, batch_size=64, shuffle=True,  num_workers=8, pin_memory=use_pin_memory)
    val_loader   = DataLoader(val_set,   batch_size=64, shuffle=False, num_workers=2, pin_memory=use_pin_memory)
    test_loader  = DataLoader(test_set,  batch_size=64, shuffle=False, num_workers=2, pin_memory=use_pin_memory)

    # 模型 & 优化器 & Loss
    model = UNet(in_ch=1, num_classes=num_classes, base=64).to(device).to(memory_format=torch.channels_last)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    ce_loss   = nn.CrossEntropyLoss(label_smoothing=0.05)     # 分类（整数标签）
    dice_loss = DiceLoss(smooth=1.0, ignore_bg=True)          # one-hot dice，忽略背景

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=20)

    best_min, EPOCHS = -1.0, 10
    for epoch in range(EPOCHS):
        for imgs, msks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            t0 = time.time()
            tr = train_one_epoch(model, train_loader, opt, ce_loss, dice_loss, num_classes)
            val_mean, val_per_class, val_min = evaluate(model, val_loader, num_classes, ignore_bg=True)
            scheduler.step()

        print(f"Epoch {epoch+1:02d}/{EPOCHS} | train_loss {tr:.4f} | "
              f"val_mean_DSC {val_mean:.4f} | val_min_DSC {val_min:.4f} | "
              f"per-class {np.round(val_per_class,4)} | time {time.time()-t0:.1f}s",
              flush=True)

        # 用 "最小类别 Dice" 作为主指标保存
        if val_min > best_min:
            best_min = val_min
            torch.save(model.state_dict(), "unet_oasis_best.pt")
            print(f"  ✅ 保存最优模型（by DSCmin），val_min_DSC={best_min:.4f}", flush=True)

        # 每 5 轮可视化一次验证集
        if (epoch+1) % 5 == 0:
            visualize_batch(model, val_loader, num_classes, out_path=f"unet_vis_val_ep{epoch+1}.png", max_samples=6)

    # ------------- 测试集推理 & 可视化（演示用） -------------
    print("🚀 测试集推理与评估...")
    model.load_state_dict(torch.load("unet_oasis_best.pt", map_location=device))
    test_mean, test_per_class, test_min = evaluate(model, test_loader, num_classes, ignore_bg=True)
    print(f"TEST  mean_DSC={test_mean:.4f} | min_DSC={test_min:.4f} | per-class={np.round(test_per_class,4)}")
    visualize_batch(model, test_loader, num_classes, out_path="unet_vis_test.png", max_samples=8)
    print("✅ 已保存可视化：unet_vis_test.png；最优权重：unet_oasis_best.pt")

if __name__ == "__main__":
    main()
