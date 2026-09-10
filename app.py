import streamlit as st
import numpy as np
import onnxruntime as ort
import torch
import matplotlib.pyplot as plt

st.set_page_config(page_title="First Break Picking", layout="wide")
st.title("⚡ First Break Picking (UNet ONNX)")

@st.cache_resource
def load_session():
    session = ort.InferenceSession("unet_model.onnx")
    input_meta = session.get_inputs()[0]
    output_meta = session.get_outputs()[0]
    return session, input_meta, output_meta

try:
    session, input_meta, output_meta = load_session()
    input_name = input_meta.name
    output_name = output_meta.name
    expected_shape = input_meta.shape
    
    st.success(f"Model uğurla yükləndi | Input: `{input_name}` | Gözlənilən Shape: `{expected_shape}`")
except Exception as e:
    st.error(f"Model yüklənmə xətası: {e}")
    st.stop()

uploaded_file = st.file_uploader(".pt Faylı Yükləyin", type=["pt"])

if uploaded_file is not None:
    data_dict = torch.load(uploaded_file, map_location="cpu")
    
    if isinstance(data_dict, dict):
        shot = data_dict["data"][0].numpy() if data_dict["data"].ndim == 3 else data_dict["data"].numpy()
        mask = data_dict.get("mask", None)
        if mask is not None and mask.ndim == 3:
            mask = mask[0].numpy()
    else:
        shot = data_dict[0].numpy()
        mask = None

    # Model (1578, 751) gözləyir. Əgər gələn matris (751, 1578)-dirsə, .T edərək (1578, 751) edirik
    if shot.shape == (751, 1578):
        shot_input = shot.T
        mask_input = mask.T if mask is not None else None
    else:
        shot_input = shot
        mask_input = mask

    # Percentile 98 normallaşdırılması
    vlim = np.percentile(np.abs(shot_input), 98)
    if vlim == 0:
        vlim = 1.0
    shot_norm = np.clip(shot_input / vlim, -1.0, 1.0).astype(np.float32)

    # Modelin kanal sayını avtomatik aşkar etmək (1 və ya 2 kanal)
    expected_channels = expected_shape[1] if len(expected_shape) > 1 and isinstance(expected_shape[1], int) else 1

    if expected_channels == 2:
        if mask_input is not None:
            m_input = mask_input.astype(np.float32)
        else:
            m_input = np.zeros_like(shot_norm, dtype=np.float32)
        combined_tensor = np.stack([shot_norm, m_input], axis=0)
    else:
        combined_tensor = np.expand_dims(shot_norm, axis=0)

    # Batch dimension: (1, C, 1578, 751)
    input_tensor = np.expand_dims(combined_tensor, axis=0).astype(np.float32)

    st.write(f"Göndərilən Tensor Shape: `{input_tensor.shape}`, Data Type: `{input_tensor.dtype}`")

    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(shot_norm, cmap="seismic", aspect="auto")
        ax.set_title("Input Seismogram")
        ax.axis("off")
        st.pyplot(fig)
        plt.close(fig)

    if st.button("İnferensiyanı Başlat"):
        try:
            outputs = session.run([output_name], {input_name: input_tensor})
            pred = outputs[0][0]
            if pred.ndim == 3:
                pred = pred.squeeze(0)

            with col2:
                fig_pred, ax_pred = plt.subplots(figsize=(6, 6))
                ax_pred.imshow(pred, cmap="tab10", aspect="auto")
                ax_pred.set_title("Prediction Mask")
                ax_pred.axis("off")
                st.pyplot(fig_pred)
                plt.close(fig_pred)

        except Exception as err:
            st.error(f"ONNX Inference Error: {err}")
