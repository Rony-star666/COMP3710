import os, random
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_transforms(img_size=224, gray=True, aug=True):
    t = [transforms.Resize((img_size, img_size)), transforms.ToTensor()]
    if gray:
        t.append(transforms.Normalize(mean=[0.5], std=[0.5]))
        return transforms.Compose([transforms.Grayscale(num_output_channels=1)] + t)
    else:
        t.append(transforms.Normalize(mean=[0.485,0.456,0.406],
                                      std=[0.229,0.224,0.225]))
        return transforms.Compose(t)


def get_loaders(data_root="ADNI/AD_NC", img_size=224, batch_size=32, num_workers=4, gray=True):
    raise NotImplementedError("loaders not implemented yet")
