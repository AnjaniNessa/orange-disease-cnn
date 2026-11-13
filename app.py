
import os
import gradio as gr
import tensorflow as tf
import numpy as np

from PIL import Image
from huggingface_hub import hf_hub_download

# Mengambil model dari Hugging Face
REPO_ID = "AnjaniNessa19/orange-disease-cnn"
MODEL_NAME = "best_orange_cnn.keras"

model_path = hf_hub_download(
    repo_id=REPO_ID,
    filename=MODEL_NAME
)

# Memuat model CNN
model = tf.keras.models.load_model(model_path)

# Urutan kelas sesuai model
class_names = [
    "Black Spot",
    "Citrus Canker",
    "Fresh Orange",
    "Greening Citrus"
]

# Fungsi prediksi
def predict_orange(image):

    if image is None:
        return {}, "Silakan unggah gambar jeruk."

    image = image.convert("RGB")
    image = image.resize((224, 224))

    array = np.array(
        image,
        dtype=np.float32
    ) / 255.0

    array = np.expand_dims(array, axis=0)

    prediction = model.predict(array, verbose=0)[0]

    results = {
        class_names[i]: float(prediction[i])
        for i in range(len(class_names))
    }

    best = int(np.argmax(prediction))

    return results, (
        f"Prediksi: {class_names[best]}\n"
        f"Keyakinan model: {prediction[best]*100:.2f}%"
    )

# Tampilan Gradio
app = gr.Interface(
    fn=predict_orange,
    inputs=gr.Image(
        type="pil",
        label="Upload Foto Jeruk"
    ),
    outputs=[
        gr.Label(
            num_top_classes=4,
            label="Hasil Prediksi"
        ),
        gr.Textbox(
            label="Informasi Hasil"
        )
    ],
    title="🍊 ORANGE DISEASE CLASSIFICATION CNN",
    description=(
        "Aplikasi klasifikasi penyakit jeruk "
        "menggunakan Convolutional Neural Network (CNN)."
    )
)

if __name__ == "__main__":
    app.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860))
    )

