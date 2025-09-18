import os, random, math, time
from glob import glob
from PIL import Image
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import matplotlib.pyplot as plt

# -----------------------------
# 0. 可重复 & 环境设置
# -----------------------------
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.benchmark = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
amp_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32  # A100建议bfloat16

# -----------------------------
# 1. 数据集
# -----------------------------
class OasisSegDataset(Dataset):
    """
    使用共同 stem（去掉前缀 case_/seg_ 和连环扩展名）进行配对。
    例如：
      case_402_slice_0.nii.png <-> seg_402_slice_0.png
    """
    def __init__(self, img_dir, msk_dir, size=(64, 64), augment=False):
        self.img_dir = img_dir
        self.msk_dir = msk_dir
        self.size = size
        self.augment = augment

        def stem(fname):
            name = fname
            if name.startswith("case_"): name = name[len("case_"):]
            if name.startswith("seg_"):  name = name[len("seg_"):]
            base, ext = os.path.splitext(name)
            while ext.lower() in [".png", ".nii", ".gz"]:
                name = base
                base, ext = os.path.splitext(name)
            return name

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
            print(f"⚠️ 有 {dropped} 张图像未找到对应的 mask，已跳过。")

        self.tf_img = T.Compose([
            T.Resize(size, interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
        ])
        self.tf_msk = T.Compose([
            T.Resize(size, interpolation=T.InterpolationMode.NEAREST),
        ])
        self.aug_ops = T.RandomChoice([
            T.RandomHorizontalFlip(p=1.0),
            T.RandomVerticalFlip(p=1.0),
            T.RandomRotation(degrees=10),
        ], p=[1/3,1/3,1/3]) if augment else None

    def __len__(self): return len(self.pairs)

    def __getitem__(self, idx):
        img_name, msk_name = self.pairs[idx]
        img = Image.open(os.path.join(self.img_dir, img_name)).convert("L")
        msk = Image.open(os.path.join(self.msk_dir, msk_name))

        if self.augment:
            seed = np.random.randint(0, 10_000)
            random.seed(seed); torch.manual_seed(seed)
            if isinstance(self.aug_ops, T.RandomChoice):
                tform = random.choice(self.aug_ops.transforms)
                img = tform(img); msk = tform(msk)

        img = self.tf_img(img)
        msk = self.tf_msk(msk)
        msk = torch.from_numpy(np.array(msk, dtype=np.int64))
        return img, msk


class DiscretizeLabelWrapper(Dataset):
    """把 mask 灰度值离散化。默认 binary：非0=前景"""
    def __init__(self, base_dataset: Dataset, mode="binary", class_values=None):
        assert mode in ("binary", "values")
        self.base = base_dataset
        self.mode = mode
        self.class_values = None
        if mode == "values":
            assert class_values is not None and len(class_values) >= 2
            self.class_values = np.array(sorted(class_values), dtype=np.float32)

    def __len__(self): return len(self.base)

    def __getitem__(self, idx):
        img, msk = self.base[idx]
        m = msk.numpy().astype(np.float32)

        if self.mode == "binary":
            m = (m > 0).astype(np.int64)
        else:
            m_flat = m.reshape(-1, 1)
            cv = self.class_values.reshape(1, -1)
            dist = np.abs(m_flat - cv)
            cls = dist.argmin(axis=1).astype(np.int64)
            m = cls.reshape(msk.shape)

        return img, torch.from_numpy(m)


def one_hot(labels: torch.Tensor, num_classes: int):
    return F.one_hot(labels.long(), num_classes=num_classes).permute(0,3,1,2).float()

# -----------------------------
# 2. UNet 模型
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
        self.down1 = DoubleConv(in_ch, base); self.pool1 = nn.MaxPool2d(2)
        self.down2 = DoubleConv(base, base*2); self.pool2 = nn.MaxPool2d(2)
        self.down3 = DoubleConv(base*2, base*4); self.pool3 = nn.MaxPool2d(2)
        self.down4 = DoubleConv(base*4, base*8); self.pool4 = nn.MaxPool2d(2)
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
        return self.out_conv(u1)

# -----------------------------
# 3. 损失 & 指标
# -----------------------------
class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0, ignore_bg=False):
        super().__init__(); self.smooth = smooth; self.ignore_bg = ignore_bg
    def forward(self, logits, targets_onehot):
        probs = torch.softmax(logits, dim=1)
        dims = (0,2,3)
        if self.ignore_bg and probs.size(1) > 1:
            probs, targets_onehot = probs[:,1:], targets_onehot[:,1:]
        inter = torch.sum(probs * targets_onehot, dims)
        denom = torch.sum(probs, dims) + torch.sum(targets_onehot, dims)
        dice = (2*inter + self.smooth) / (denom + self.smooth)
        return 1 - dice.mean()

def dice_per_class(logits, targets, num_classes, ignore_bg=False, eps=1e-7):
    with torch.no_grad():
        probs = torch.softmax(logits, dim=1)
        preds = probs.argmax(1)
        pred_oh = one_hot(preds, num_classes).to(logits.dtype)
        tgt_oh  = one_hot(targets, num_classes).to(logits.dtype)
        if ignore_bg and num_classes > 1:
            pred_oh, tgt_oh = pred_oh[:,1:], tgt_oh[:,1:]
        inter = torch.sum(pred_oh * tgt_oh, (0,2,3))
        denom = torch.sum(pred_oh, (0,2,3)) + torch.sum(tgt_oh, (0,2,3))
        return ((2*inter + eps) / (denom + eps)).cpu().numpy()

# -----------------------------
# 4. 训练/评估循环
# -----------------------------
def train_one_epoch(model, loader, optimizer, ce_loss, dice_loss, num_classes, scaler=None):
    model.train(); total, n_batches = 0.0, 0
    for imgs, msks in loader:
        imgs, msks = imgs.to(device), msks.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=device.type=="cuda"):
            logits = model(imgs)
            ce = ce_loss(logits, msks)
            msk_oh = one_hot(msks, num_classes).to(logits.dtype)
            dl = dice_loss(logits, msk_oh)
            loss = ce + dl
        loss.backward(); optimizer.step()
        total += loss.item(); n_batches += 1
    return total / max(1,n_batches)

@torch.no_grad()
def evaluate(model, loader, num_classes, ignore_bg=True):
    model.eval(); dices_all = []
    for imgs, msks in loader:
        imgs, msks = imgs.to(device), msks.to(device)
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=device.type=="cuda"):
            logits = model(imgs)
        dices_all.append(dice_per_class(logits, msks, num_classes, ignore_bg))
    dices_all = np.stack(dices_all, 0)
    return dices_all.mean(), dices_all.mean(0)

# -----------------------------
# 5. 可视化
# -----------------------------
@torch.no_grad()
def visualize_batch(model, loader, num_classes, out_path="unet_vis.png", max_samples=6):
    model.eval()
    imgs, msks = next(iter(loader))
    imgs, msks = imgs.to(device), msks.to(device)
    preds = torch.softmax(model(imgs), 1).argmax(1)
    imgs, msks, preds = imgs.cpu().numpy(), msks.cpu().numpy(), preds.cpu().numpy()
    n = min(max_samples, imgs.shape[0])
    plt.figure(figsize=(9, 3*n))
    for i in range(n):
        plt.subplot(n,3,3*i+1); plt.imshow(imgs[i,0], cmap="gray"); plt.title("Image"); plt.axis("off")
        plt.subplot(n,3,3*i+2); plt.imshow(msks[i], cmap="tab20"); plt.title("GT"); plt.axis("off")
        plt.subplot(n,3,3*i+3); plt.imshow(preds[i], cmap="tab20"); plt.title("Pred"); plt.axis("off")
    plt.tight_layout(); plt.savefig(out_path, dpi=150); plt.close()

# -----------------------------
# 6. 主函数
# -----------------------------
def main():
    base_dir = "/Users/ljh/Desktop/lab2/keras_png_slices_data"
    train_img = os.path.join(base_dir, "keras_png_slices_train")
    val_img   = os.path.join(base_dir, "keras_png_slices_validate")
    test_img  = os.path.join(base_dir, "keras_png_slices_test")
    train_msk = os.path.join(base_dir, "keras_png_slices_seg_train")
    val_msk   = os.path.join(base_dir, "keras_png_slices_seg_validate")
    test_msk  = os.path.join(base_dir, "keras_png_slices_seg_test")

    # 数据集
    train_set = DiscretizeLabelWrapper(OasisSegDataset(train_img, train_msk, (64,64), augment=True), mode="binary")
    val_set   = DiscretizeLabelWrapper(OasisSegDataset(val_img,   val_msk,   (64,64)), mode="binary")
    test_set  = DiscretizeLabelWrapper(OasisSegDataset(test_img,  test_msk,  (64,64)), mode="binary")
    num_classes = 2  # 二分类

    use_pin_memory = (device.type == "cuda")
    train_loader = DataLoader(train_set, 64, True,  num_workers=1, pin_memory=use_pin_memory)
    val_loader   = DataLoader(val_set,   64, False, num_workers=1, pin_memory=use_pin_memory)
    test_loader  = DataLoader(test_set,  64, False, num_workers=1, pin_memory=use_pin_memory)

    # 模型
    model = UNet(in_ch=1, num_classes=num_classes, base=64).to(device).to(memory_format=torch.channels_last)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    ce_loss, dice_loss = nn.CrossEntropyLoss(label_smoothing=0.05), DiceLoss(smooth=1.0, ignore_bg=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)

    best_val, EPOCHS = -1, 30
    for epoch in range(EPOCHS):
        t0 = time.time()
        tr_loss = train_one_epoch(model, train_loader, optimizer, ce_loss, dice_loss, num_classes)
        val_mean, val_per_class = evaluate(model, val_loader, num_classes, True)
        scheduler.step()
        print(f"Epoch {epoch+1:02d}/{EPOCHS} | train_loss {tr_loss:.4f} | val_mean_DSC {val_mean:.4f} | per-class {np.round(val_per_class,4)} | time {time.time()-t0:.1f}s")
        if val_mean > best_val:
            best_val = val_mean; torch.save(model.state_dict(), "unet_oasis_best.pt")
            print(f"  ✅ 保存最优模型，val_mean_DSC={best_val:.4f}")
        if (epoch+1) % 5 == 0:
            visualize_batch(model, val_loader, num_classes, out_path=f"unet_vis_val_ep{epoch+1}.png")

    print("🚀 测试集评估...")
    model.load_state_dict(torch.load("unet_oasis_best.pt", map_location=device))
    test_mean, test_per_class = evaluate(model, test_loader, num_classes, True)
    print(f"TEST Mean DSC: {test_mean:.4f} | per-class: {np.round(test_per_class,4)}")
    visualize_batch(model, test_loader, num_classes, "unet_vis_test.png", 8)
    print("✅ 可视化保存 unet_vis_test.png")

if __name__ == "__main__":
    main()
