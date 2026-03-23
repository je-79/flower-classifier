# src/data/flower_dataset.py
import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image

class FlowerDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples   = samples    # list of (PIL Image, label)
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img, label = self.samples[idx]
        img = np.array(img)         # albumentations expects numpy HWC
        if self.transform:
            img = self.transform(image=img)["image"]
        return img, torch.tensor(label, dtype=torch.long)
import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image

class FlowerDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples   = samples    # list of (PIL Image, label)
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img, label = self.samples[idx]
        img = np.array(img)         # albumentations expects numpy HWC
        if self.transform:
            img = self.transform(image=img)["image"]
        return img, torch.tensor(label, dtype=torch.long)


def mixup_batch(images, labels, num_classes=102, alpha=0.2):
    """Apply MixUp to a batch. Call after collating, before forward pass."""
    lam = np.random.beta(alpha, alpha)
    batch_size = images.size(0)
    index = torch.randperm(batch_size, device=images.device)
    mixed_images = lam * images + (1 - lam) * images[index]
    labels_a, labels_b = labels, labels[index]
    return mixed_images, labels_a, labels_b, lam


def mixup_criterion(criterion, pred, labels_a, labels_b, lam):
    return lam * criterion(pred, labels_a) + (1 - lam) * criterion(pred, labels_b)


def get_dataloaders(train, val, test, batch_size=32,
                    train_tf=None, val_tf=None, num_workers=4):
    from torch.utils.data import DataLoader
    train_ds = FlowerDataset(train, train_tf)
    val_ds   = FlowerDataset(val,   val_tf)
    test_ds  = FlowerDataset(test,  val_tf)   # same transforms as val
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                   num_workers=num_workers, pin_memory=True),
        DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                   num_workers=num_workers, pin_memory=True),
        DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                   num_workers=num_workers, pin_memory=True),
    )
