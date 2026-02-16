import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

# --- CONFIGURATION ---
# This matches the alphabetical order of the HAM10000 folders
CLASS_NAMES = {
    0: 'Actinic Keratoses (akiec)',
    1: 'Basal Cell Carcinoma (bcc)', 
    2: 'Benign Keratosis (bkl)',
    3: 'Dermatofibroma (df)',
    4: 'Melanoma (mel)',
    5: 'Melanocytic Nevi (nv)',
    6: 'Vascular Lesions (vasc)'
}

# --- 1. LOAD MODEL ---
@st.cache_resource
def load_learner():
    try:
        # This is the file you will download from Colab later
        model = tf.keras.models.load_model('skin_cancer_model.h5')
        return model
    except Exception as e:
        return None

model = load_learner()

# --- 2. PREPROCESSING (Digital Hair Removal) ---
def preprocess_image(image):
    # Convert to numpy array
    img_array = np.array(image.convert('RGB'))
    
    # Hair Removal (DullRazor Algorithm)
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    kernel = cv2.getStructuringElement(1, (17, 17))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    _, threshold = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
    final_image = cv2.inpaint(img_array, threshold, 1, cv2.INPAINT_TELEA)
    
    # Resize to 224x224 (Model Requirement)
    final_image = cv2.resize(final_image, (224, 224))
    
    # Normalize pixel values to 0-1
    final_image = final_image.astype('float32') / 255.0
    
    # Add batch dimension (1, 224, 224, 3)
    return np.expand_dims(final_image, axis=0)

# --- 3. UI DESIGN ---
st.set_page_config(page_title="DermaAI Scanner", page_icon="🔬")

st.markdown("<h1 style='text-align: center; color: #d33;'>🔬 DermaAI: Skin Lesion Classifier</h1>", unsafe_allow_html=True)
st.markdown("### Upload a dermoscopic image for AI risk assessment.")

# Sidebar
st.sidebar.title("About")
st.sidebar.info("This model uses **EfficientNetB0** trained on the HAM10000 dataset. It includes digital hair removal preprocessing.")
st.sidebar.warning("⚠️ **Disclaimer:** This tool is for educational purposes only. It is NOT a substitute for professional medical advice.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display Image
    image = Image.open(uploaded_file)
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(image, caption='Analyzed Image', use_column_width=True)
    
    if model is None:
        st.error("⚠️ Model not found! Please download 'skin_cancer_model.h5' and place it in this folder.")
    else:
        # Make Prediction
        with st.spinner('Scanning for patterns...'):
            processed_img = preprocess_image(image)
            predictions = model.predict(processed_img)
            
            # Get results
            class_idx = np.argmax(predictions[0])
            confidence = np.max(predictions[0]) * 100
            result_label = CLASS_NAMES.get(class_idx, "Unknown")

        # Display Results
        with col2:
            st.markdown("### 🔍 Analysis Report")
            
            # Critical Alert for high risk classes
            high_risk = ['Melanoma (mel)', 'Basal Cell Carcinoma (bcc)', 'Actinic Keratoses (akiec)']
            
            if any(risk in result_label for risk in high_risk):
                st.error(f"**PREDICTION:** {result_label}")
                st.warning("⚠️ High-Risk markers detected. Please consult a dermatologist.")
            else:
                st.success(f"**PREDICTION:** {result_label}")
                st.info("✅ Lesion appears benign. Monitor for changes.")
                
            st.metric("AI Confidence", f"{confidence:.1f}%")
            
            # Expandable technical details
            with st.expander("See Raw Probabilities"):
                st.bar_chart(predictions[0])