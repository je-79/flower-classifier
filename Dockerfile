FROM python:3.11-slim
WORKDIR /app
RUN mkdir -p outputs/exports configs src/serving
COPY requirements.txt .
RUN pip install --no-cache-dir fastapi uvicorn onnxruntime pillow numpy python-multipart
COPY outputs/exports/flower_classifier.onnx outputs/exports/
COPY outputs/exports/flower_classifier.onnx.data outputs/exports/
COPY configs/label_map.json configs/
COPY src/serving/api.py src/serving/
CMD ["uvicorn", "src.serving.api:app", "--host", "0.0.0.0", "--port", "8000"]
