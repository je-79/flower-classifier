import os
import json
from huggingface_hub import HfApi, create_repo
import tensorflow_datasets as tfds

HF_USERNAME = "je-01"
REPO_NAME   = "flower-classifier-efficientnetv2s"
REPO_ID     = f"{HF_USERNAME}/{REPO_NAME}"
TOKEN       = os.environ["HF_TOKEN"]

api = HfApi()

# 1. Create private repo
create_repo(REPO_ID, token=TOKEN, private=True, exist_ok=True)
print(f"Repo ready: https://huggingface.co/{REPO_ID}")

# 2. Upload best checkpoint
print("Uploading model weights...")
api.upload_file(
    path_or_fileobj="outputs/checkpoints/best_model.pt",
    path_in_repo="weights/best_model.pt",
    repo_id=REPO_ID,
    token=TOKEN
)

# 3. Build and upload label / index map (flower names)
# Oxford Flowers 102 label map
import tensorflow_datasets as tfds
print("Building label map...")
builder = tfds.builder("oxford_flowers102",
                        data_dir="./data/tfds")
info       = builder.info
names      = info.features["label"].names
label_map  = {str(i): name for i, name in enumerate(names)}

with open("configs/label_map.json", "w") as f:
    json.dump(label_map, f, indent=2)

api.upload_file(
    path_or_fileobj="configs/label_map.json",
    path_in_repo="configs/label_map.json",
    repo_id=REPO_ID,
    token=TOKEN
)

print("Done — model and label map uploaded to HF Hub")
print(f"https://huggingface.co/{REPO_ID}")
