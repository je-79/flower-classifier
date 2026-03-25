from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import onnxruntime as ort
import numpy as np
from PIL import Image
import json, io, time

app = FastAPI(title="Flower Classifier API")

# Load model + labels once at startup
session   = ort.InferenceSession("outputs/exports/flower_classifier.onnx",
                providers=["CPUExecutionProvider"])
with open("configs/label_map.json") as f:
    LABEL_MAP = json.load(f)

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def preprocess(img: Image.Image) -> np.ndarray:
    img = img.resize((300, 300)).convert("RGB")
    x   = np.array(img, dtype=np.float32) / 255.0
    x   = (x - MEAN) / STD
    return x.transpose(2, 0, 1)[None]   # NCHW

@app.get("/")
def root():
    return {"status": "ok", "model": "EfficientNetV2-S · Oxford Flowers 102"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    t0  = time.perf_counter()
    img = Image.open(io.BytesIO(await file.read()))
    x   = preprocess(img)
    logits   = session.run(["logits"], {"image": x})[0][0]
    probs    = np.exp(logits) / np.exp(logits).sum()
    top5_idx = probs.argsort()[::-1][:5]
    latency  = round((time.perf_counter() - t0) * 1000, 1)

    results = [
        {"rank": i+1,
         "class_id": int(idx),
         "flower": LABEL_MAP[str(idx)],
         "confidence": round(float(probs[idx]), 4)}
        for i, idx in enumerate(top5_idx)
    ]
    confident = probs[top5_idx[0]] >= 0.60
    return JSONResponse({
        "predictions" : results,
        "confident"   : confident,
        "latency_ms"  : latency,
        "warning"     : None if confident else "Low confidence — result may be uncertain"
    })
