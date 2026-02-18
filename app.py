import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import os

# --- 1. CONFIGURATION (8 CLASSES - MATCHING YOUR V2 MODEL) ---
CLASS_NAMES = {
    0: 'Actinic Keratoses (akiec)',
    1: 'Basal Cell Carcinoma (bcc)',
    2: 'Benign Keratosis (bkl)',
    3: 'Dermatofibroma (df)',
    4: 'Healthy Skin',           # The Safety Class
    5: 'Melanocytic Nevi (nv)',
    6: 'Melanoma (mel)',
    7: 'Vascular Lesions (vasc)'
}

# --- 2. QUALITY GATES (Blur & Skin Filter) ---
def check_image_quality(image):
    """
    GATE 0: Blur Check (Laplacian Var < 80)
    GATE 1: Skin Check (YCbCr > 15%)
    """
    img_array = np.array(image.convert('RGB'))
    
    # A. BLUR CHECK
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
    if blur_score < 80: 
        return False, f"Image is too blurry (Score: {blur_score:.1f}). Please retake."

    # B. SKIN CHECK (YCbCr)
    img_ycbcr = cv2.cvtColor(img_array, cv2.COLOR_RGB2YCrCb)
    lower_skin = np.array([0, 133, 77], dtype=np.uint8)
    upper_skin = np.array([255, 173, 127], dtype=np.uint8)
    skin_mask = cv2.inRange(img_ycbcr, lower_skin, upper_skin)
    skin_pct = (cv2.countNonZero(skin_mask) / (img_array.shape[0] * img_array.shape[1])) * 100

    if skin_pct < 15: 
        return False, f"No skin detected ({skin_pct:.1f}%). Upload a clear skin photo."

    return True, "Passed"

# --- 3. MODEL LOADER ---
@st.cache_resource
def load_learner():
    tf.keras.backend.clear_session()
    # ENSURE THIS MATCHES YOUR FILENAME
    model_path = "skin_cancer_model_v2_perfect.h5" 
    
    if not os.path.exists(model_path):
        st.error(f"❌ File '{model_path}' not found.")
        return None

    try:
        model = tf.keras.models.load_model(model_path, compile=False)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model
    except Exception as e:
        st.error(f"❌ Error: {e}")
        return None

model = load_learner()

# --- 4. UI HEADER ---
st.set_page_config(page_title="DermaAI", page_icon="🔬")
st.markdown("<h1 style='text-align: center; color: #d33;'>🔬 DermaAI: Intelligent Diagnostics</h1>", unsafe_allow_html=True)

# --- 5. MAIN APP ---
uploaded_file = st.file_uploader("Upload Scan", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(image, caption='Patient Scan', width=300)
    
    # RUN GATES
    is_valid, msg = check_image_quality(image)
    
    if not is_valid:
        with col2:
            st.error("🚫 IMAGE REJECTED")
            st.warning(msg)
    
    elif model is not None:
        with st.spinner('Analyzing...'):
            # PREPROCESS
            img_array = np.array(image.convert('RGB'))
            img_array = cv2.resize(img_array, (224, 224))
            img_array = img_array.astype('float32')
            img_input = np.expand_dims(img_array, axis=0)

            # PREDICT
            preds = model.predict(img_input)
            probs = preds[0]
            class_idx = np.argmax(probs)
            confidence = probs[class_idx] * 100
            label = CLASS_NAMES.get(class_idx, "Unknown")
            
            # --- SAFETY LOGIC ENGINE ---
            status = "gray"
            display_msg = "Inconclusive."

            # 1. HEALTHY SKIN (Green)
            if class_idx == 4:
                status = "green"
                label = "Healthy Skin"
                display_msg = "No abnormalities detected."

            # 2. HIGH CONFIDENCE CANCER (Red) -> >65% Sure
            elif class_idx in [0, 1, 6] and confidence > 65:
                status = "red"
                if class_idx == 6: label = "High-Risk Melanoma"
                elif class_idx == 1: label = "Basal Cell Carcinoma"
                else: label = "Actinic Keratosis"
                display_msg = "⚠️ Malignant features detected. Immediate consultation recommended."

            # 3. HIGH CONFIDENCE BENIGN (Green) -> >50% Sure
            elif class_idx in [2, 3, 5, 7] and confidence > 50:
                status = "green"
                display_msg = "Benign Lesion. Monitor for changes."

            # 4. THE GRAY ZONE (Infections/Rashes) -> Low Confidence
            else:
                status = "gray"
                label = "Inconclusive / Suspicious"
                display_msg = "Analysis uncertain. This could be an infection, rash, or early lesion. Please consult a doctor."

            # DISPLAY
            with col2:
                if status == "red":
                    st.error(f"**DIAGNOSIS:** {label}")
                    st.error(display_msg)
                elif status == "green":
                    st.success(f"**RESULT:** {label}")
                    st.success(f"✅ {display_msg}")
                else:
                    st.warning(f"**RESULT:** {label}")
                    st.info(display_msg)
                    st.caption("ℹ️ Note: Low confidence. AI excludes fungal/rash conditions.")
                
                st.metric("AI Confidence", f"{confidence:.1f}%")
                st.bar_chart(dict(zip(CLASS_NAMES.values(), probs)))