"""
Train on a Custom Dataset
============================
Train FlashOCR on your own text recognition dataset.

Data format (labels.tsv):
    image_001.jpg\thello
    image_002.jpg\tworld
"""
from flashocr import Trainer

trainer = Trainer(
    model_size="m",
    train_images="data/train",
    train_labels="data/train/labels.tsv",
    val_images="data/val",
    val_labels="data/val/labels.tsv",
    charset="0123456789abcdefghijklmnopqrstuvwxyz",
    epochs=100,
    batch_size=64,
    device="cuda",
)
trainer.train()

print(f"Best model saved to: {trainer.best_model_path}")
print(f"Best accuracy: {trainer.best_accuracy:.4f}")
