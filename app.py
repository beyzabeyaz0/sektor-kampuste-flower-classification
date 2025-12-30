import streamlit as st
import torch
from transformers import ViTForImageClassification, ViTImageProcessor
from PIL import Image
import os
import pandas as pd

# --------------------------------------------------
# SAYFA AYARLARI
# --------------------------------------------------
st.set_page_config(
    page_title="Çiçek Sınıflandırma Web Arayüzü",
    layout="centered"
)

st.title("🌸 Çiçek Sınıflandırma Sistemi")
st.write(
    "Bilgisayarınızdan bir çiçek fotoğrafı yükleyin. "
    "Eğitilmiş Vision Transformer (ViT) modeli kullanarak çiçek türünü tahmin edelim!"
)

st.divider()

# --------------------------------------------------
# SINIF İSİMLERİ (EĞİTİM SIRASIYLA)
# --------------------------------------------------
class_names = ["daisy", "dandelion", "rose", "sunflower", "tulip"]

# --------------------------------------------------
# MODEL YOLU
# --------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "flowers_vit_model_cikti")

# --------------------------------------------------
# MODEL ve PROCESSOR
# --------------------------------------------------
@st.cache_resource
def load_model():
    model = ViTForImageClassification.from_pretrained(MODEL_DIR)
    processor = ViTImageProcessor.from_pretrained(MODEL_DIR)
    model.eval()
    return model, processor

model, processor = load_model()

# --------------------------------------------------
# GÖRÜNTÜ YÜKLEME ALANI
# --------------------------------------------------
uploaded_file = st.file_uploader(
    "📤 Görüntü Yükle",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    # --------------------------------------------------
    # GÖRSEL OKUMA ve GÖSTERME
    # --------------------------------------------------
    image = Image.open(uploaded_file).convert("RGB")

    st.subheader("🖼️ Yüklenen Görüntü")
    st.image(image, use_container_width=True)

    st.divider()

    # --------------------------------------------------
    # TAHMİN BUTONU
    # --------------------------------------------------
    if st.button("🔍 Tahmin Et"):
        with st.spinner("Görüntü işleniyor ve sınıflandırılıyor..."):

            # -----------------------------
            # OTOMATİK ÖN İŞLEME
            # (resize + crop + normalize)
            # -----------------------------
            inputs = processor(
                images=image,
                return_tensors="pt"
            )

            # -----------------------------
            # MODEL TAHMİNİ
            # -----------------------------
            with torch.no_grad():
                outputs = model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=1)
                confidence, predicted_class = torch.max(probs, dim=1)

        predicted_label = class_names[predicted_class.item()]
        confidence_score = confidence.item() * 100

        # --------------------------------------------------
        # SONUÇLAR
        # --------------------------------------------------
        st.success(f"✅ Tahmin Edilen Sınıf: **{predicted_label}**")
        st.info(f"🔢 Güven Oranı: **%{confidence_score:.2f}**")

        st.divider()

        # --------------------------------------------------
        # OLASILIK GÖSTERİMİ
        # --------------------------------------------------
        st.subheader("📊 Sınıf Olasılıkları")

        prob_df = pd.DataFrame({
            "Sınıf": class_names,
            "Olasılık (%)": probs.squeeze().numpy() * 100
        })

        st.dataframe(prob_df, use_container_width=True)
        st.bar_chart(prob_df.set_index("Sınıf"))

else:
    st.warning("👆 Lütfen bir görüntü yükleyin.")
