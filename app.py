import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import os

# --- CONFIGURATION ---
CLASS_NAMES = {
    0: 'Actinic Keratoses (akiec)',
    1: 'Basal Cell Carcinoma (bcc)', 
    2: 'Benign Keratosis (bkl)',
    3: 'Dermatofibroma (df)',
    4: 'Melanoma (mel)',
    5: 'Melanocytic Nevi (nv)',
    6: 'Vascular Lesions (vasc)'
}

# --- 1. ROBUST MODEL LOADER ---
@st.cache_resource
def load_learner():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, 'skin_cancer_model.h5')
    
    if not os.path.exists(model_path):
        st.error(f"❌ ERROR: File not found at: `{model_path}`")
        return None

    try:
        st.info("🔧 Building model architecture...")
        
        # 1. Re-create Base Model
        base_model = tf.keras.applications.EfficientNetB0(
            include_top=False,
            weights=None, 
            input_shape=(224, 224, 3)
        )
        base_model.trainable = False 

        # 2. Re-create Head
        model = tf.keras.models.Sequential([
            base_model,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(7, activation='softmax')
        ])
        
        # 3. Compile & Load Weights
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        model.load_weights(model_path)
        
        st.success("✅ Model System Online")
        return model

    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None

model = load_learner()

# --- 2. PREPROCESSING ---
def preprocess_image(image):
    img_array = np.array(image.convert('RGB'))
    img_array = img_array.astype('uint8')
    
    # Hair Removal
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    kernel = cv2.getStructuringElement(1, (17, 17))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    _, threshold = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
    final_image = cv2.inpaint(img_array, threshold, 1, cv2.INPAINT_TELEA)
    
    # Resize & Convert
    final_image = cv2.resize(final_image, (224, 224))
    final_image = final_image.astype('float32')
    
    return np.expand_dims(final_image, axis=0)

# --- 3. UI DESIGN ---
st.set_page_config(page_title="DermaAI Scanner", page_icon="🔬")

st.markdown("<h1 style='text-align: center; color: #d33;'>🔬 DermaAI: Skin Lesion Classifier</h1>", unsafe_allow_html=True)

st.sidebar.title("About Project")
st.sidebar.info("Features **EfficientNetB0** + **Digital Hair Removal**.")
st.sidebar.info("🛡️ **High-Sensitivity Mode Active:** This model prioritizes patient safety. Any suspicious features will trigger a medical alert, even if the primary prediction is benign.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    col1, col2 = st.columns(2)
    
    with col1:
        # FIXED: Removed yellow warning box
        st.image(image, caption='Original Upload', width=350)
    
    if model is not None:
        with st.spinner('Scanning for patterns...'):
            processed_img = preprocess_image(image)
            predictions = model.predict(processed_img)
            probs = predictions[0] # Get probabilities
            
            # Get the "Winner"
            class_idx = np.argmax(probs)
            primary_label = CLASS_NAMES.get(class_idx, "Unknown")
            primary_conf = probs[class_idx] * 100
            
            # --- THE SAFETY NET LOGIC ---
            # Indices: 4=Melanoma, 1=BCC, 0=Actinic Keratosis
            # If ANY cancer class is above 20%, override the "Safe" verdict
            cancer_risk_score = probs[4] + probs[1]
            is_high_risk = False
            
            if probs[4] > 0.15: # If Melanoma > 15%
                is_high_risk = True
                warning_msg = "⚠️ High-Risk Melanoma features detected."
                display_label = "Possible Melanoma"
            elif probs[1] > 0.20: # If BCC > 20%
                is_high_risk = True
                warning_msg = "⚠️ Basal Cell Carcinoma features detected."
                display_label = "Possible Carcinoma"
            elif primary_label in ['Melanoma (mel)', 'Basal Cell Carcinoma (bcc)']:
                is_high_risk = True
                warning_msg = f"⚠️ Diagnosis: {primary_label}"
                display_label = primary_label
            else:
                display_label = primary_label

            # --- DISPLAY RESULTS ---
            with col2:
                st.markdown("### 🔍 Analysis Report")
                
                if is_high_risk:
                    st.error(f"**ALERT:** {display_label}")
                    st.warning(f"{warning_msg} Please consult a dermatologist.")
                else:
                    st.success(f"**PREDICTION:** {display_label}")
                    st.info("✅ Lesion appears benign. Monitor for changes.")
                
                # Show the score of the WINNER
                st.metric("Model Confidence (Primary Class)", f"{primary_conf:.1f}%")
                
                with st.expander("See Risk Profile"):
                    # Show bar chart of probabilities
                    st.bar_chart(dict(zip(CLASS_NAMES.values(), probs)))
                    if is_high_risk:
                        st.caption("Note: High-Sensitivity Mode triggered an alert based on sub-threshold probabilities.")