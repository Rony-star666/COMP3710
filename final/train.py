# train.py
import os, argparse, time
import torch
import torch.nn as nn
from torch.optim import Adam
from sklearn.metrics import accuracy_score, roc_auc_score
import matplotlib.pyplot as plt
from tqdm import tqdm

from modules import build_model
from dataset import get_loaders, set_seed

def train_one_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0, 0, 0

    # 🔹 使用 tqdm 包装 DataLoader
    for imgs, labels in tqdm(train_loader, desc="Training", ncols=100):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss, y_true, y_prob = 0.0, [], []
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)
        loss = criterion(logits, labels)
        running_loss += loss.item() * imgs.size(0)
        probs = torch.softmax(logits, dim=1)[:,1].cpu().numpy()
        y_prob.extend(probs.tolist())
        y_true.extend(labels.cpu().numpy().tolist())

    epoch_loss = running_loss / len(loader.dataset)
    acc = accuracy_score(y_true, (torch.tensor(y_prob)>0.5).numpy())
    try:
        auc = roc_auc_score(y_true, y_prob)
    except Exception:
        auc = 0.0
    return epoch_loss, acc, auc

def plot_curves(history, outdir):
    os.makedirs(outdir, exist_ok=True)
    # loss
    plt.figure()
    plt.plot(history["train_loss"], label="train_loss")
    plt.plot(history["val_loss"], label="val_loss")
    plt.xlabel("epoch"); plt.ylabel("loss"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(outdir, "loss_curve.png")); plt.close()
    # acc
    plt.figure()
    plt.plot(history["train_acc"], label="train_acc")
    plt.plot(history["val_acc"], label="val_acc")
    plt.xlabel("epoch"); plt.ylabel("acc"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(outdir, "acc_curve.png")); plt.close()
    # auc
    plt.figure()
    plt.plot(history["train_auc"], label="train_auc")
    plt.plot(history["val_auc"], label="val_auc")
    plt.xlabel("epoch"); plt.ylabel("auc"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(outdir, "auc_curve.png")); plt.close()

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # 数据加载
    train_loader, val_loader, test_loader, class_names = get_loaders(
        data_root=args.data_root,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.workers,
        gray=(args.in_channels == 1)
    )

    # 模型、损失、优化器
    model = build_model(in_channels=args.in_channels, height=args.img_size, width=args.img_size)


    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_acc = 0
    os.makedirs(args.outdir, exist_ok=True)

    for epoch in range(args.epochs):
        print(f"\nEpoch [{epoch+1}/{args.epochs}]")

        # ---- 训练 ----
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        print(f"Train Loss: {tr_loss:.4f}, Train Acc: {tr_acc:.4f}")

        # ---- 验证 ----
        val_loss, val_acc, val_auc = evaluate(model, val_loader, criterion, device)

        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val AUC: {val_auc:.4f}")


        # ---- 保存最优模型 ----
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(args.outdir, "best_model.pt"))
            print("✅ Saved best model")

    # ---- 测试集评估 ----
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"\nTest Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, default="ADNI/AD_NC")
    ap.add_argument("--outdir", type=str, default="runs/adni_resnet")
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-5)
    ap.add_argument("--in_channels", type=int, default=1, help="ADNI 多为灰度：1")
    ap.add_argument("--backbone", type=str, default="resnet18", choices=["resnet18","resnet50"])
    ap.add_argument("--no_pretrain", action="store_true")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    main(args)
