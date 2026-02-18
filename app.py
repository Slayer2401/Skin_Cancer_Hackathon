import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import os

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="DermaAI Diagnostics",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. LOAD EXTERNAL CSS ---
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Load the style.css file
if os.path.exists("style.css"):
    local_css("style.css")
else:
    st.warning("⚠️ Style file not found. Please ensure 'style.css' is in the same folder.")

# --- 3. LOGIC & CONFIGURATION ---
CLASS_NAMES = {
    0: 'Actinic Keratoses (akiec)',
    1: 'Basal Cell Carcinoma (bcc)',
    2: 'Benign Keratosis (bkl)',
    3: 'Dermatofibroma (df)',
    4: 'Healthy Skin',
    5: 'Melanocytic Nevi (nv)',
    6: 'Melanoma (mel)',
    7: 'Vascular Lesions (vasc)'
}

def check_image_quality(image):
    img_array = np.array(image.convert('RGB'))
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
    if blur_score < 50: 
        return False, f"Blur Score: {blur_score:.1f} (Too Low)"
    
    img_ycbcr = cv2.cvtColor(img_array, cv2.COLOR_RGB2YCrCb)
    lower_skin = np.array([0, 133, 77], dtype=np.uint8)
    upper_skin = np.array([255, 173, 127], dtype=np.uint8)
    skin_mask = cv2.inRange(img_ycbcr, lower_skin, upper_skin)
    skin_pct = (cv2.countNonZero(skin_mask) / (img_array.shape[0] * img_array.shape[1])) * 100
    
    if skin_pct < 15:
        return False, f"Skin Detect: {skin_pct:.1f}% (No Skin Found)"
        
    return True, "Passed"

@st.cache_resource
def load_learner():
    tf.keras.backend.clear_session()
    model_path = "skin_cancer_model_v2_perfect.h5"
    if not os.path.exists(model_path): return None
    try:
        model = tf.keras.models.load_model(model_path, compile=False)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model
    except: return None

model = load_learner()

# --- 4. SIDEBAR NAVIGATION ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3063/3063176.png", width=80)
    st.markdown("## DermaAI Pro")
    st.markdown("---")
    st.markdown("### 📋 Patient Details")
    st.text_input("Patient ID", placeholder="E.g. PT-1024")
    st.date_input("Consultation Date")
    st.markdown("---")
    st.info("🔒 Secure Mode Active\nNo data is saved to cloud.")

# --- 5. MAIN CONTENT ---
col1, col2 = st.columns([1, 2])

with col1:
    st.markdown("### 📤 Upload Scan")
    uploaded_file = st.file_uploader("Drop dermoscopic image here", type=["jpg", "png", "jpeg"])
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Current Scan", use_column_width=True)

with col2:
    st.markdown("### 📊 Diagnostic Analysis")
    
    if not uploaded_file:
        st.info("👈 Please upload a patient scan to begin analysis.")
    else:
        # Quality Gate
        is_valid, msg = check_image_quality(image)
        
        if not is_valid:
            st.markdown(f"""
            <div class="result-card-gray">
                <h3>🚫 Image Rejected</h3>
                <p>{msg}</p>
                <p><b>Action:</b> Please retake the photo with better lighting and focus.</p>
            </div>
            """, unsafe_allow_html=True)
            
        elif model is not None:
            with st.spinner('Processing neural network layers...'):
                # Predict
                img_array = np.array(image.convert('RGB'))
                img_array = cv2.resize(img_array, (224, 224))
                img_array = img_array.astype('float32')
                img_input = np.expand_dims(img_array, axis=0)
                
                preds = model.predict(img_input)
                probs = preds[0]
                class_idx = np.argmax(probs)
                confidence = probs[class_idx] * 100
                label = CLASS_NAMES.get(class_idx, "Unknown")

                # Logic Engine
                if class_idx == 4: # Healthy
                    card_class = "result-card-green"
                    icon = "✅"
                    status = "Healthy Skin"
                    desc = "No pathological abnormalities detected. Routine monitoring recommended."
                elif class_idx in [0, 1, 6] and confidence > 65: # Cancer
                    card_class = "result-card-red"
                    icon = "🚨"
                    status = f"High Risk: {label}"
                    desc = "Malignant features detected. Immediate biopsy and specialist referral required."
                elif class_idx in [2, 3, 5, 7] and confidence > 50: # Benign
                    card_class = "result-card-green"
                    icon = "🟢"
                    status = f"Benign: {label}"
                    desc = "Lesion appears benign. Monitor for asymmetry or color changes."
                else: # Gray Zone
                    card_class = "result-card-gray"
                    icon = "⚠️"
                    status = "Inconclusive Analysis"
                    desc = "Confidence low. Possible infection, rash, or noise. Clinical correlation needed."

                # DISPLAY RESULT CARD
                st.markdown(f"""
                <div class="{card_class}">
                    <h2>{icon} {status}</h2>
                    <p><b>AI Confidence:</b> {confidence:.1f}%</p>
                    <p>{desc}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # METRICS ROW
                m1, m2, m3 = st.columns(3)
                m1.metric("Confidence", f"{confidence:.1f}%", delta_color="off")
                m2.metric("Scan Quality", "Pass", "High")
                m3.metric("Model Version", "V2.1 (EfficientNet)")
                
                # CHART
                st.markdown("#### Probability Distribution")
                st.bar_chart(dict(zip(CLASS_NAMES.values(), probs)))