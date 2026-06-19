"""
LoRA Fine-Tuning
============================
Fine-tune a pretrained FlashOCR model with LoRA for your custom charset.
Only ~5% of parameters are trained, much faster convergence.
"""
from flashocr import Trainer

trainer = Trainer(
    model_size="m",
    train_images="data/custom/train",
    train_labels="data/custom/train/labels.tsv",
    val_images="data/custom/val",
    val_labels="data/custom/val/labels.tsv",
    charset="0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ",
    epochs=50,
    batch_size=64,
    device="cuda",
    pretrained=True,
    use_lora=True,
    lora_rank=8,
    lora_alpha=16.0,
    lora_variant="dora",
)
trainer.train()

print(f"LoRA model saved to: {trainer.best_model_path}")
print(f"Trainable params: {trainer.trainable_params:,} / {trainer.total_params:,}")
