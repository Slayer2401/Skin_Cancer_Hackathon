import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import os

# --- 1. CONFIGURATION (MATCHING V2 MODEL) ---
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

# --- 2. ADVANCED QUALITY GATES (The "Smart Filter") ---

def check_image_quality(image):
    """
    GATE 0: Checks if image is blurry.
    GATE 1: Checks if image contains skin.
    Returns: (Pass/Fail, Message)
    """
    # Convert to OpenCV format
    img_array = np.array(image.convert('RGB'))
    
    # --- GATE 0: BLUR DETECTION (Laplacian Variance) ---
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    if blur_score < 80: # Threshold for "Too Blurry"
        return False, f"Image is too blurry (Score: {blur_score:.1f}). Please retake with better focus."

    # --- GATE 1: SKIN DETECTION (YCbCr Method) ---
    # We use YCbCr because it works well on ALL skin tones (dark/light)
    img_ycbcr = cv2.cvtColor(img_array, cv2.COLOR_RGB2YCrCb)
    
    # Skin Color Bounds (Medical Standard)
    lower_skin = np.array([0, 133, 77], dtype=np.uint8)
    upper_skin = np.array([255, 173, 127], dtype=np.uint8)
    
    # Masking
    skin_mask = cv2.inRange(img_ycbcr, lower_skin, upper_skin)
    skin_pixels = cv2.countNonZero(skin_mask)
    total_pixels = img_array.shape[0] * img_array.shape[1]
    skin_pct = (skin_pixels / total_pixels) * 100

    if skin_pct < 15: # Less than 15% skin? Reject it.
        return False, f"No skin detected (Only {skin_pct:.1f}% skin tones found). Please upload a close-up of the lesion."

    return True, "Quality Check Passed."

# --- 3. MODEL LOADER ---
@st.cache_resource
def load_learner():
    tf.keras.backend.clear_session()
    model_path = "skin_cancer_model_v2_perfect.h5"
    
    if not os.path.exists(model_path):
        st.error(f"❌ ERROR: Model file '{model_path}' not found.")
        st.warning("👉 Move 'skin_cancer_model_v2_perfect.h5' to this folder.")
        return None

    try:
        model = tf.keras.models.load_model(model_path, compile=False)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None

model = load_learner()

# --- 4. UI DESIGN ---
st.set_page_config(page_title="DermaAI Pro", page_icon="🔬")
st.markdown("<h1 style='text-align: center; color: #d33;'>🔬 DermaAI: Intelligent Diagnostics</h1>", unsafe_allow_html=True)
st.markdown("### 🛡️ Medical-Grade Skin Lesion Analysis")
st.caption("Powered by EfficientNetB0 • YCbCr Skin Detection • Noise Filtering")

# --- 5. MAIN APPLICATION ---
uploaded_file = st.file_uploader("Upload Dermoscopic Scan", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(image, caption='Patient Scan', width=300)
    
    # --- RUN QUALITY GATES FIRST ---
    is_valid, msg = check_image_quality(image)
    
    if not is_valid:
        with col2:
            st.error("🚫 IMAGE REJECTED")
            st.warning(msg)
            st.info("The AI model was NOT run to prevent false diagnosis.")
    
    # --- IF QUALITY IS GOOD, RUN AI ---
    elif model is not None:
        with st.spinner('Analyzing tissue structure...'):
            # Preprocess
            img_array = np.array(image.convert('RGB'))
            img_array = cv2.resize(img_array, (224, 224))
            img_array = img_array.astype('float32')
            img_input = np.expand_dims(img_array, axis=0)

            # Predict
            preds = model.predict(img_input)
            probs = preds[0]
            
            # Get Results
            class_idx = np.argmax(probs)
            confidence = probs[class_idx] * 100
            label = CLASS_NAMES.get(class_idx, "Unknown")
            
            # --- LOGIC ENGINE ---
            status_color = "green"
            display_msg = "Benign."
            
            # 1. Healthy Skin Check
            if class_idx == 4:
                status_color = "green"
                label = "Healthy Skin"
                display_msg = "No pathological features detected."
            
            # 2. Cancer Flags
            elif class_idx == 6: # Melanoma
                status_color = "red"
                display_msg = "⚠️ High-Risk Malignancy Detected."
            elif class_idx == 1: # BCC
                status_color = "red"
                display_msg = "⚠️ Basal Cell Carcinoma Detected."
                
            # 3. Low Confidence Fallback
            if confidence < 40:
                status_color = "gray"
                label = "Inconclusive"
                display_msg = "Low confidence. Please consult a dermatologist."

            # --- DISPLAY REPORT ---
            with col2:
                if status_color == "red":
                    st.error(f"**DIAGNOSIS:** {label}")
                    st.error(display_msg)
                elif status_color == "gray":
                    st.warning(f"**RESULT:** {label}")
                    st.info(display_msg)
                else:
                    st.success(f"**RESULT:** {label}")
                    st.success(f"✅ {display_msg}")
                
                st.metric("AI Confidence", f"{confidence:.1f}%")
                
                with st.expander("Show Probability Distribution"):
                    st.bar_chart(dict(zip(CLASS_NAMES.values(), probs)))