
import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
from huggingface_hub import hf_hub_download

# Konfigurasi aplikasi
st.set_page_config(
    page_title="Orange Disease CNN",
    page_icon="🍊",
    layout="centered"
)

st.title("🍊 Orange Disease Classification")
st.write("Aplikasi deteksi penyakit jeruk menggunakan Convolutional Neural Network (CNN).")

# Nama repository dan model
REPO_ID = "AnjaniNessa19/orange-disease-cnn"
MODEL_NAME = "best_orange_cnn.keras"

CLASS_NAMES = [
    "Black Spot",
    "Citrus Canker",
    "Fresh Orange",
    "Greening Citrus"
]

# Memuat model dari Hugging Face
@st.cache_resource
def load_model():
    model_path = hf_hub_download(
        repo_id=REPO_ID,
        filename=MODEL_NAME
    )
    return tf.keras.models.load_model(model_path)

# Upload gambar
uploaded_file = st.file_uploader(
    "Upload gambar daun atau buah jeruk",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Gambar yang diunggah",
        use_container_width=True
    )

    if st.button("🔍 Klasifikasi Sekarang"):
        try:
            with st.spinner("Sedang menganalisis gambar..."):
                model = load_model()

                image_resized = image.resize((224, 224))
                image_array = np.array(
                    image_resized,
                    dtype=np.float32
                ) / 255.0

                image_array = np.expand_dims(
                    image_array,
                    axis=0
                )

                prediction = model.predict(
                    image_array,
                    verbose=0
                )[0]

                predicted_index = np.argmax(prediction)
                predicted_class = CLASS_NAMES[predicted_index]
                confidence = float(prediction[predicted_index]) * 100

            st.success("Klasifikasi berhasil!")

            st.subheader("Hasil Prediksi")
            st.write(f"**Kelas:** {predicted_class}")
            st.write(f"**Tingkat keyakinan:** {confidence:.2f}%")

            st.subheader("Probabilitas Setiap Kelas")

            for name, score in zip(CLASS_NAMES, prediction):
                st.write(f"{name}: {score * 100:.2f}%")
                st.progress(float(score))

        except Exception as e:
            st.error(f"Terjadi kesalahan: {e}")
