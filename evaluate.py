import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from transformers import ViTForImageClassification, ViTImageProcessor
from datasets import load_dataset
import os

# --------------------------------------------------
# AYARLAR
# --------------------------------------------------
DATASET_PATH = r"/Users/elifbeyzabeyaz/Desktop/sektorkampuste/flowers"
MODEL_DIR = "./flowers_vit_model_cikti"

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# --------------------------------------------------
# VERİ SETİ
# --------------------------------------------------
ds = load_dataset("imagefolder", data_dir=DATASET_PATH, split="train")
ds = ds.train_test_split(test_size=0.2, seed=42)

labels = ds["train"].features["label"].names

# --------------------------------------------------
# MODEL ve PROCESSOR
# --------------------------------------------------
processor = ViTImageProcessor.from_pretrained(MODEL_DIR)
model = ViTForImageClassification.from_pretrained(MODEL_DIR)
model.to(device)
model.eval()

# --------------------------------------------------
# TAHMİN
# --------------------------------------------------
y_true = []
y_pred = []

with torch.no_grad():
    for example in ds["test"]:
        image = example["image"].convert("RGB")
        inputs = processor(images=image, return_tensors="pt").to(device)

        outputs = model(**inputs)
        pred = torch.argmax(outputs.logits, dim=1).item()

        y_true.append(example["label"])
        y_pred.append(pred)

# --------------------------------------------------
# METRİKLER
# --------------------------------------------------
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average="weighted")
recall = recall_score(y_true, y_pred, average="weighted")
f1 = f1_score(y_true, y_pred, average="weighted")

print("\n📊 Model Performans Sonuçları\n")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-score : {f1:.4f}")

print("\n📋 Detaylı Sınıflandırma Raporu:\n")
print(classification_report(y_true, y_pred, target_names=labels))