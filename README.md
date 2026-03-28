# 🌸 Flower Classifier — 102 Species

End-to-end ML project: train, export, and serve a flower classification model across multiple platforms.

## Model
- Architecture: EfficientNetV2-S
- Dataset: Oxford Flowers 102 (102 species)
- Training: PyTorch 

## Serving
| Platform | URL | Use case |
|----------|-----|----------|
| REST API | [Render](https://flower-classifier-tyb5.onrender.com) | Cataloguing |
| Web demo | [HF Spaces](https://huggingface.co/spaces/je-01/flower-classifier) | Field ID / Store |

## Project structure
```
flower_classifier/
├── configs/          # label map, training config
├── notebooks/        # EDA
├── outputs/          # exports (ONNX, CoreML)
├── src/
│   ├── data/         # dataset, augmentation
│   ├── models/       # EfficientNetV2-S
│   ├── export/       # ONNX, CoreML, HF Hub
│   └── serving/      # FastAPI, Gradio
└── requirements.txt
```

## Exports
- ONNX — REST API via FastAPI on Render
- CoreML — iOS on-device inference (M3 Neural Engine)

## Model weights
Hosted on HF Hub: [je-01/flower-classifier-efficientnetv2s](https://huggingface.co/je-01/flower-classifier-efficientnetv2s)
