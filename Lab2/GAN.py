import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
from tqdm import tqdm

# -----------------------------
# 0. 环境 & 超参数
# -----------------------------
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_path = "./keras_png_slices_data"

noise_dim = 100
image_size = 256
batch_size = 16
num_epochs = 50
lr = 0.0002
beta1 = 0.5
save_interval = 5  # 每隔多少 epoch 保存一次模型和图像

# -----------------------------
# 1. 数据集
# -----------------------------
class MyDataset(Dataset):
    def __init__(self, root, transform=None):
        self.image_dir = os.path.join(root, "keras_png_slices_train")
        self.image_files = sorted(os.listdir(self.image_dir))
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.image_files[index])
        image = Image.open(img_path).convert("L")
        if self.transform:
            image = self.transform(image)
        return image

transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

dataset = MyDataset(data_path, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# -----------------------------
# 2. 模型
# -----------------------------
class Generator(nn.Module):
    def __init__(self, noise_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(noise_dim, 1024, 4, 1, 0, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(True),

            nn.ConvTranspose2d(1024, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),

            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),

            nn.ConvTranspose2d(32, 1, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, z):
        return self.model(z)


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(32, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(512, 1024, 4, 2, 1, bias=False),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(1024, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, img):
        return self.model(img).view(-1, 1).squeeze(1)

# -----------------------------
# 3. 初始化
# -----------------------------
def weights_init(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
        nn.init.normal_(m.weight.data, 0.0, 0.02)

generator = Generator(noise_dim).to(device)
discriminator = Discriminator().to(device)
generator.apply(weights_init)
discriminator.apply(weights_init)

criterion = nn.BCELoss()
optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, 0.999))

fixed_noise = torch.randn(64, noise_dim, 1, 1, device=device)
os.makedirs("GAN_images", exist_ok=True)

# -----------------------------
# 4. 训练循环
# -----------------------------
for epoch in range(1, num_epochs + 1):
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{num_epochs}")
    for i, real_images in enumerate(pbar):
        real_images = real_images.to(device)
        batch_size = real_images.size(0)

        real_labels = torch.ones(batch_size, device=device)
        fake_labels = torch.zeros(batch_size, device=device)

        # --- Train Discriminator ---
        optimizer_D.zero_grad()
        outputs = discriminator(real_images)
        d_loss_real = criterion(outputs, real_labels)

        z = torch.randn(batch_size, noise_dim, 1, 1, device=device)
        fake_images = generator(z)
        outputs = discriminator(fake_images.detach())
        d_loss_fake = criterion(outputs, fake_labels)

        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        optimizer_D.step()

        # --- Train Generator ---
        optimizer_G.zero_grad()
        outputs = discriminator(fake_images)
        g_loss = criterion(outputs, real_labels)
        g_loss.backward()
        optimizer_G.step()

        pbar.set_postfix(D_loss=d_loss.item(), G_loss=g_loss.item())

    # --- Save samples & models ---
    if epoch % save_interval == 0 or epoch == num_epochs:
        with torch.no_grad():
            fake_images = generator(fixed_noise).detach().cpu()
            save_image(fake_images, f"GAN_images/epoch_{epoch}.png", normalize=True, nrow=8)

        torch.save(generator.state_dict(), f"Generator_epoch{epoch}.pth")
        torch.save(discriminator.state_dict(), f"Discriminator_epoch{epoch}.pth")
        print(f"✅ Epoch {epoch} completed. Models & samples saved.")

print("🎉 Training completed!")
