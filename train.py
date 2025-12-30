import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset
from sklearn.metrics import accuracy_score
from transformers import (
    ViTImageProcessor, 
    ViTForImageClassification, 
    TrainingArguments, 
    Trainer, 
    DefaultDataCollator
)

# --- 1. AYARLAR ---
# Kopyaladığın tam yolu buraya tırnak içinde yapıştırdık
DATASET_PATH = r"/Users/elifbeyzabeyaz/Desktop/sektorkampuste/flowers" 
MODEL_NAME = "google/vit-base-patch16-224"
OUTPUT_DIR = "./flowers_vit_model_cikti"
EPOCHS = 10
BATCH_SIZE = 16 

# --- 2. GPU (MPS) KONTROLÜ ---
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("\nMac GPU (MPS) Tespit Edildi. Eğitim GPU üzerinde yapılacak.")
else:
    device = torch.device("cpu")
    print("\nMPS bulunamadı, işlemler CPU üzerinden devam edecek.")

# --- 3. VERİ SETİNİ YÜKLEME (KRİTİK DÜZELTME) ---
print(f"📂 Veri seti okunuyor: {DATASET_PATH}")

# 'split="train"' parametresi EmptyDatasetError hatasını çözer.
try:
    ds = load_dataset("imagefolder", data_dir=DATASET_PATH, split="train")
    # Veriyi %80 Eğitim, %20 Test olarak bölüyoruz
    ds = ds.train_test_split(test_size=0.2, seed=42)
except Exception as e:
    print(f"❌ HATA: Veri yüklenemedi. Klasör yapısını kontrol edin.\nDetay: {e}")
    exit()

labels = ds['train'].features['label'].names
label2id = {label: str(i) for i, label in enumerate(labels)}
id2label = {str(i): label for i, label in enumerate(labels)}

print(f"Sınıflar: {labels}")

# --- 4. ÖN İŞLEME ---
processor = ViTImageProcessor.from_pretrained(MODEL_NAME)

def transform(example_batch):
    # 'imagefolder' ile yüklenen verilerde resim sütunu 'image' adını alır.
    inputs = processor([x.convert("RGB") for x in example_batch['image']], return_tensors='pt')
    inputs['labels'] = example_batch['label']
    return inputs

prepared_ds = ds.with_transform(transform)

# --- 5. MODELİ HAZIRLAMA ---
model = ViTForImageClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(labels),
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True
)
model.to(device)

# --- 6. METRİK ---
def compute_metrics(p):
    return {"accuracy": accuracy_score(p.label_ids, np.argmax(p.predictions, axis=1))}

# --- 7. EĞİTİM AYARLARI ---
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    learning_rate=2e-5,
    weight_decay=0.01,
    remove_unused_columns=False, # Transform kullandığımız için False kalmalı
    eval_strategy="epoch",       
    save_strategy="epoch",
    load_best_model_at_end=True,
    fp16=False,                  # Mac MPS'de kararlılık için False (Önemli)
    dataloader_num_workers=0,    # Mac'te MPS çakışmasını önlemek için 0
    logging_steps=10,
    report_to="none"
)

# --- 8. TRAINER ---
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=prepared_ds["train"],
    eval_dataset=prepared_ds["test"],
    data_collator=DefaultDataCollator(),
    processing_class=processor, 
    compute_metrics=compute_metrics,
)

print("\n Eğitim Başlıyor...")
trainer.train()

# --- 9. KAYIT ---
trainer.save_model(OUTPUT_DIR)
processor.save_pretrained(OUTPUT_DIR)
print(f"\nModel başarıyla kaydedildi: {OUTPUT_DIR}")

# --- 10. NİHAİ DEĞERLENDİRME ---
print("\nTest Seti Üzerindeki Nihai Değerlendirme Sonuçları")
metrics = trainer.evaluate()

for key, value in metrics.items():
    print(f"{key}: {value:.4f}")