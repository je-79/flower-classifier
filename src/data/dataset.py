import tensorflow_datasets as tfds
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

def build_splits(data_dir="./data/tfds", seed=42):
    print("Downloading Oxford Flowers 102 (first run only)...")
    splits, info = tfds.load(
        "oxford_flowers102",
        split=["train", "validation", "test"],
        as_supervised=True,
        with_info=True,
        data_dir=data_dir
    )
    print(f"Classes : {info.features['label'].num_classes}")

    all_data = []
    for split in splits:
        for img_t, lbl_t in split:
            all_data.append((
                Image.fromarray(img_t.numpy()).convert("RGB"),
                int(lbl_t.numpy())
            ))

    labels_all = [s[1] for s in all_data]
    train_d, temp_d, _, temp_l = train_test_split(
        all_data, labels_all,
        test_size=0.30, stratify=labels_all, random_state=seed)
    val_d, test_d, _, _ = train_test_split(
        temp_d, temp_l,
        test_size=0.50, stratify=temp_l, random_state=seed)

    print(f"Train: {len(train_d)} | Val: {len(val_d)} | Test: {len(test_d)}")
    return train_d, val_d, test_d
