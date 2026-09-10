import streamlit as st
import numpy as np
import onnxruntime as ort
from PIL import Image

# Səhifə konfiqurasiyası
st.set_page_config(
    page_title="First Break Picking",
    page_icon="⚡",
    layout="centered"
)

st.title("⚡ First Break Picking (ONNX)")
st.write("Seysmik şəkil faylını yükləyin və modelin ilk gəliş dalğalarını necə aşkar etdiyini görün.")

# ONNX modelini yaddaşa yükləyirik (cache vasitəsilə sürətləndiririk)
@st.cache_resource
def load_model():
    session = ort.InferenceSession("unet_model.onnx")
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    return session, input_name, output_name

try:
    session, input_name, output_name = load_model()
except Exception as e:
    st.error(f"Model yüklənərkən xəta baş verdi: {e}")
    st.stop()

# Giriş ölçüləri (Modelinizin gözlədiyi ölçüyə uyğun dəyişin)
IMG_HEIGHT = 256
IMG_WIDTH = 256

# Fayl yükləmə interfeysi
uploaded_file = st.file_uploader("Seysmik kəsim şəklini seçin...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Şəkli açırıq
    image = Image.open(uploaded_file)
    
    # 2 Sütunlu vizuallaşdırma
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Giriş Şəkli")
        st.image(image, use_container_width=True)
    
    # Ön emal (Pre-processing)
    img_resized = image.convert("RGB").resize((IMG_WIDTH, IMG_HEIGHT))
    img_np = np.array(img_resized, dtype=np.float32) / 255.0
    img_np = np.transpose(img_np, (2, 0, 1))  # (H, W, C) -> (C, H, W)
    input_tensor = np.expand_dims(img_np, axis=0)
    
    # Modellə təxmin
    with st.spinner("Model təxmin edir..."):
        outputs = session.run([output_name], {input_name: input_tensor})
        mask = outputs[0][0]
        if mask.ndim == 3 and mask.shape[0] == 1:
            mask = mask.squeeze(0)
        mask_binary = (mask > 0.5).astype(np.uint8) * 255
        result_img = Image.fromarray(mask_binary)
    
    with col2:
        st.subheader("Təxmin (Maska)")
        st.image(result_img, use_container_width=True)
