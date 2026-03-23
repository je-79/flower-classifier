# notebooks/eda.py  — run with: python notebooks/eda.py
import sys
sys.path.insert(0, ".")
from src.data.dataset import build_splits
import matplotlib.pyplot as plt
from collections import Counter

train, val, test = build_splits()

# Class distribution
labels = [s[1] for s in train]
counts = Counter(labels)
plt.figure(figsize=(18, 3))
plt.bar(counts.keys(), counts.values(), width=1.0)
plt.xlabel("Class ID (0–101)")
plt.ylabel("Image count")
plt.title("Oxford Flowers 102 — training set class distribution")
plt.tight_layout()
plt.savefig("outputs/logs/class_distribution.png", dpi=120)
print(f"Min class count: {min(counts.values())}")
print(f"Max class count: {max(counts.values())}")
print("Saved: outputs/logs/class_distribution.png")
