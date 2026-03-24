# configs/train_config.py

CFG = {
    # Data
    "data_dir"      : "./data/tfds",
    "num_classes"   : 102,
    "img_size"      : 300,
    "batch_size"    : 32,
    "num_workers"   : 4,
    "seed"          : 42,

    # Model
    "backbone"      : "tf_efficientnetv2_s",
    "dropout"       : 0.4,
    "pretrained"    : True,

    # Training stages
    "stage1_epochs" : 10,
    "stage2_epochs" : 40,
    "total_epochs"  : 50,

    # Optimiser
    "lr_head"       : 1e-3,
    "weight_decay"  : 1e-4,
    "label_smoothing": 0.1,

    # Scheduler
    "T_0"           : 10,
    "T_mult"        : 2,

    # MixUp
    "mixup_alpha"   : 0.2,
    "mixup_prob"    : 0.5,

    # Checkpointing
    "ckpt_dir"      : "./outputs/checkpoints",
    "ckpt_every_n"  : 5,            # save to Drive every N epochs
    "early_stop_patience": 7,

    # W&B
    "wandb_project" : "flower-classifier",
    "wandb_entity"  : None,         # your W&B username (or leave None)

    # Device — auto-detects M3 MPS or Kaggle CUDA
    "device"        : "mps" if __import__("torch").backends.mps.is_available()
                      else ("cuda" if __import__("torch").cuda.is_available()
                      else "cpu"),
}
