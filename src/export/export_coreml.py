
import torch
import timm
import torch.nn as nn
import coremltools as ct

def load_model(ckpt_path, device="cpu"):
    model = timm.create_model("tf_efficientnetv2_s",
                               pretrained=False, num_classes=0, global_pool="avg")
    model.classifier = nn.Sequential(
        nn.BatchNorm1d(model.num_features),
        nn.Dropout(0.4),
        nn.Linear(model.num_features, 102)
    )
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model




model = load_model("outputs/checkpoints/best_model.pt")  # reuse function above
model.eval()
traced = torch.jit.trace(model, torch.randn(1, 3, 300, 300))

mlmodel = ct.convert(
    traced,
    inputs=[ct.ImageType(name="image", shape=(1, 3, 300, 300),
                         bias=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
                         scale=1.0/255.0)],
    convert_to="mlprogram",   # modern CoreML format
    compute_precision=ct.precision.FLOAT16
)
mlmodel.save("outputs/exports/FlowerClassifier.mlpackage")
print("CoreML export complete: outputs/exports/FlowerClassifier.mlpackage")




import torch, coremltools as ct
import torch.nn as nn
import timm

model = load_model("outputs/checkpoints/best_model.pt")  # reuse function above
model.eval()
traced = torch.jit.trace(model, torch.randn(1, 3, 300, 300))

mlmodel = ct.convert(
    traced,
    inputs=[ct.ImageType(name="image", shape=(1, 3, 300, 300),
                         bias=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
                         scale=1.0/255.0)],
    convert_to="mlprogram",   # modern CoreML format
    compute_precision=ct.precision.FLOAT16
)
mlmodel.save("outputs/exports/FlowerClassifier.mlpackage")
print("CoreML export complete: outputs/exports/FlowerClassifier.mlpackage")
