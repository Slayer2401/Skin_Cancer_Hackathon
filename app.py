import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import os

# --- 1. CONFIGURATION (Matches your V2 Model) ---
CLASS_NAMES = {
    0: 'Actinic Keratoses (akiec)',
    1: 'Basal Cell Carcinoma (bcc)',
    2: 'Benign Keratosis (bkl)',
    3: 'Dermatofibroma (df)',
    4: 'Healthy Skin',           # The new class you added!
    5: 'Melanocytic Nevi (nv)',
    6: 'Melanoma (mel)',
    7: 'Vascular Lesions (vasc)'
}

# --- 2. LOAD THE MODEL ---
@st.cache_resource
def load_model():
    # Clear session to avoid conflicts
    tf.keras.backend.clear_session()
    
    # Path to the model file
    model_path = "skin_cancer_model_v2_perfect.h5"
    
    if not os.path.exists(model_path):
        st.error(f"❌ ERROR: File '{model_path}' not found.")
        st.warning("👉 Please download 'skin_cancer_model_v2_perfect.h5' from Google Drive and put it in this folder.")
        return None

    try:
        # Load the model (Custom Head + EfficientNet)
        # We use 'compile=False' to avoid safety warnings, then compile manually
        model = tf.keras.models.load_model(model_path, compile=False)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None

model = load_model()

# --- 3. UI DESIGN ---
st.set_page_config(page_title="DermaAI Scanner", page_icon="🔬")
st.markdown("<h1 style='text-align: center; color: #d33;'>🔬 DermaAI: Skin Lesion Analysis</h1>", unsafe_allow_html=True)
st.markdown("### Upload a dermoscopic image for AI risk assessment.")

# --- 4. PREDICTION ENGINE ---
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(image, caption='Uploaded Scan', width=300)
    
    if model is not None:
        with st.spinner('Analyzing tissue structure...'):
            # Preprocess
            img_array = np.array(image.convert('RGB'))
            img_array = cv2.resize(img_array, (224, 224))
            img_array = img_array.astype('float32')
            # EfficientNet expects 0-255, so we don't divide by 255 here if using standard preprocessing
            # But your training script used standard loading, so let's keep it simple:
            img_input = np.expand_dims(img_array, axis=0)

            # Predict
            preds = model.predict(img_input)
            probs = preds[0]
            
            # Get Top Prediction
            class_idx = np.argmax(probs)
            confidence = probs[class_idx] * 100
            label = CLASS_NAMES.get(class_idx, "Unknown")
            
            # --- RESULTS LOGIC ---
            status_color = "green"
            msg = "Benign."
            
            # 1. Healthy Skin
            if class_idx == 4:
                status_color = "green"
                label = "Healthy Skin"
                msg = "No abnormalities detected."
            
            # 2. Cancer Flags (Melanoma & BCC)
            elif class_idx == 6: # Melanoma
                status_color = "red"
                msg = "High-Risk Malignancy Detected."
            elif class_idx == 1: # BCC
                status_color = "red"
                msg = "Basal Cell Carcinoma Detected."
                
            # 3. Low Confidence Fallback
            if confidence < 35:
                status_color = "gray"
                label = "Inconclusive"
                msg = "Image quality low or unclear features."

            # --- DISPLAY REPORT ---
            with col2:
                if status_color == "red":
                    st.error(f"**DIAGNOSIS:** {label}")
                    st.error(f"⚠️ {msg}")
                elif status_color == "gray":
                    st.warning(f"**RESULT:** {label}")
                    st.info(msg)
                else:
                    st.success(f"**RESULT:** {label}")
                    st.success(f"✅ {msg}")
                
                st.metric("Confidence Score", f"{confidence:.2f}%")
                
                # Show Chart
                st.bar_chart(dict(zip(CLASS_NAMES.values(), probs)))