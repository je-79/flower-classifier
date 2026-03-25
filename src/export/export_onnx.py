import torch
import timm
import torch.nn as nn

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

model = load_model("outputs/checkpoints/best_model.pt")
dummy = torch.randn(1, 3, 300, 300)

torch.onnx.export(
    model, dummy,
    "outputs/exports/flower_classifier.onnx",
    input_names=["image"],
    output_names=["logits"],
    dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
    opset_version=17
)
print("ONNX export complete: outputs/exports/flower_classifier.onnx")
