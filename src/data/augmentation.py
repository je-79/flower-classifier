# src/data/augmentation.py
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np

# ImageNet normalisation stats (EfficientNetV2-S pretrained on ImageNet)
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]
IMG_SIZE = 300   # EfficientNetV2-S native resolution

def get_train_transforms():
    return A.Compose([
        A.RandomResizedCrop(IMG_SIZE, IMG_SIZE, scale=(0.7, 1.0)),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=15, p=0.5),
        A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05, p=0.8),
        A.ToGray(p=0.1),
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        A.Normalize(mean=MEAN, std=STD),
        ToTensorV2()
    ])

def get_val_transforms():
    return A.Compose([
        A.Resize(320, 320),
        A.CenterCrop(IMG_SIZE, IMG_SIZE),
        A.Normalize(mean=MEAN, std=STD),
        ToTensorV2()
    ])
