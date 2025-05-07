import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import cv2

# Memuat model yang sudah dilatih
model = tf.keras.models.load_model("ResNet50000.h5")

# Label kelas
class_labels = ['Reject', 'Ripe', 'Unripe']

# Fungsi untuk memprediksi gambar dengan model
def predict_image(image, model):
    # Ubah gambar ke format BGR sesuai kebutuhan ResNet50
    img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, (224, 224))  # Ubah ukuran ke 224x224
    img_array = np.expand_dims(img, axis=0)  # Tambah dimensi batch
    img_array = tf.keras.applications.resnet50.preprocess_input(img_array)  # Normalisasi
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class = class_labels[predicted_class_index]
    return predicted_class

# Fungsi klasifikasi gambar
def classify_image(uploaded_file):
    image = Image.open(uploaded_file)
    st.image(image, caption="Gambar yang Diunggah", use_container_width=True)
    img_array = np.array(image)
    class_label = predict_image(img_array, model)
    st.success(f"Hasil Prediksi: {class_label}")

# Tampilan utama aplikasi
def main():
    st.markdown("<h1 style='text-align: center; color: green;'>🍅 Klasifikasi Kematangan Tomat</h1>", unsafe_allow_html=True)
    st.markdown("Upload gambar tomat untuk mengetahui statusnya: **(Ripe)Matang**, **(Unripe)Mentah**, atau **(Reject)Rusak**.")
    
    uploaded_file = st.file_uploader("Unggah Gambar Tomat", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        classify_image(uploaded_file)
    
if __name__ == "__main__":
    main()
