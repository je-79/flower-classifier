# src/models/efficientnet.py
import torch
import torch.nn as nn
import timm

NUM_CLASSES = 102

def build_model(pretrained=True, dropout=0.4):
    """
    EfficientNetV2-S from timm.
    Returns model with custom head for 102-class flower classification.
    """
    model = timm.create_model(
        "tf_efficientnetv2_s",
        pretrained=pretrained,
        num_classes=0,          # remove original head
        global_pool="avg"
    )
    in_features = model.num_features   # 1280 for EfficientNetV2-S

    model.classifier = nn.Sequential(
        nn.BatchNorm1d(in_features),
        nn.Dropout(p=dropout),
        nn.Linear(in_features, NUM_CLASSES)
    )
    return model


def freeze_backbone(model):
    """Stage 1: freeze everything except the classifier head."""
    for name, param in model.named_parameters():
        if "classifier" not in name:
            param.requires_grad = False
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Stage 1 — trainable params: {trainable:,}")
    return model


def unfreeze_backbone(model, lr_head=1e-3):
    """
    Stage 2: unfreeze all. Return layer-wise LR groups.
    Deeper layers get 10x lower LR (layer-wise LR decay).
    """
    for param in model.parameters():
        param.requires_grad = True

    # Split into 3 groups: head | top blocks | deep blocks
    head_params  = list(model.classifier.parameters())
    top_params   = list(model.blocks[-2:].parameters()) if hasattr(model, 'blocks') else []
    deep_params  = [p for p in model.parameters()
                    if id(p) not in {id(x) for x in head_params + top_params}]

    param_groups = [
        {"params": head_params,  "lr": lr_head},
        {"params": top_params,   "lr": lr_head * 0.1},
        {"params": deep_params,  "lr": lr_head * 0.01},
    ]
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Stage 2 — trainable params: {total:,}")
    return param_groups
