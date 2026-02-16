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
        # Load the model from the local directory
        model = tf.keras.models.load_model('skin_cancer_model.h5')
        return model
    except Exception as e:
        return None

model = load_learner()

# --- 2. PREPROCESSING (Digital Hair Removal) ---
def preprocess_image(image):
    """
    Preprocesses the image to match the training pipeline exactly.
    1. Convert to Array
    2. Digital Hair Removal (DullRazor)
    3. Resize to 224x224
    4. Format for Model (No /255 division!)
    """
    # Convert PIL image to numpy array (RGB)
    img_array = np.array(image.convert('RGB'))
    
    # Ensure it's uint8 for OpenCV processing
    img_array = img_array.astype('uint8')
    
    # --- Hair Removal (DullRazor Algorithm) ---
    # Convert to grayscale to find dark hair on light skin
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    
    # Kernel for the morphological filtering
    kernel = cv2.getStructuringElement(1, (17, 17))
    
    # BlackHat transform to isolate hair
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    
    # Thresholding to create a hair mask
    _, threshold = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
    
    # Inpaint (fill) the masked hair pixels with surrounding skin color
    final_image = cv2.inpaint(img_array, threshold, 1, cv2.INPAINT_TELEA)
    
    # --- Formatting ---
    # Resize to 224x224 (Model Requirement)
    final_image = cv2.resize(final_image, (224, 224))
    
    # CRITICAL: Match the training script!
    # Do NOT divide by 255.0. Just convert to float32.
    final_image = final_image.astype('float32')
    
    # Add batch dimension (1, 224, 224, 3)
    return np.expand_dims(final_image, axis=0)

# --- 3. UI DESIGN ---
st.set_page_config(page_title="DermaAI Scanner", page_icon="🔬")

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        text-align: center; 
        color: #d33;
    }
    .report-box {
        border: 2px solid #eee;
        padding: 20px;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='main-header'>🔬 DermaAI: Skin Lesion Classifier</h1>", unsafe_allow_html=True)
st.markdown("### Upload a dermoscopic image for AI risk assessment.")

# Sidebar
st.sidebar.title("About Project")
st.sidebar.info("This AI model uses **EfficientNetB0** trained on the HAM10000 dataset. It features automatic **Digital Hair Removal** to improve accuracy on real-world images.")
st.sidebar.warning("⚠️ **Disclaimer:** This tool is for educational purposes only. It is NOT a substitute for professional medical advice.")

# File Uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(image, caption='Original Upload', use_column_width=True)
    
    # Check if model is loaded
    if model is None:
        st.error("⚠️ Model not found! Please download 'skin_cancer_model.h5' from Colab and place it in this folder.")
    else:
        # Run Prediction
        with st.spinner('Scanning for patterns...'):
            try:
                processed_img = preprocess_image(image)
                predictions = model.predict(processed_img)
                
                # Get the highest probability class
                class_idx = np.argmax(predictions[0])
                confidence = np.max(predictions[0]) * 100
                result_label = CLASS_NAMES.get(class_idx, "Unknown")
                
                # Success Logic
                with col2:
                    st.markdown("### 🔍 Analysis Report")
                    
                    # High Risk Categories
                    high_risk = ['Melanoma (mel)', 'Basal Cell Carcinoma (bcc)', 'Actinic Keratoses (akiec)']
                    
                    if any(risk in result_label for risk in high_risk):
                        st.error(f"**PREDICTION:** {result_label}")
                        st.warning("⚠️ High-Risk markers detected. Please consult a dermatologist immediately.")
                    else:
                        st.success(f"**PREDICTION:** {result_label}")
                        st.info("✅ Lesion appears benign. Monitor for changes.")
                        
                    st.metric("AI Confidence Score", f"{confidence:.1f}%")
                    
                    # Expandable details
                    with st.expander("See Raw Probabilities"):
                        st.bar_chart(predictions[0])
                        
            except Exception as e:
                st.error(f"Error during prediction: {e}")