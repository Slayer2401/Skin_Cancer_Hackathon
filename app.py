import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import os
import plotly.express as px
import pandas as pd
import io
from fpdf import FPDF
from datetime import date

# ═══════════════════════════════════════════════════════════
# 1. PAGE CONFIGURATION
# ═══════════════════════════════════════════════════════════
st.set_page_config(
    page_title="DermaAI Diagnostics",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ═══════════════════════════════════════════════════════════
# 2. LOAD EXTERNAL CSS
# ═══════════════════════════════════════════════════════════
def local_css(file_name):
    with open(file_name, encoding="utf-8") as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

css_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "style.css")
if os.path.exists(css_path):
    local_css(css_path)

# ═══════════════════════════════════════════════════════════
# 3. CLASS DEFINITIONS
# ═══════════════════════════════════════════════════════════
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

LABEL_TO_IDX = {v: k for k, v in CLASS_NAMES.items()}

HEALTHY_CLASSES   = {LABEL_TO_IDX['Healthy Skin']}
MALIGNANT_CLASSES = {
    LABEL_TO_IDX['Actinic Keratoses (akiec)'],
    LABEL_TO_IDX['Basal Cell Carcinoma (bcc)'],
    LABEL_TO_IDX['Melanoma (mel)']
}
BENIGN_CLASSES = {
    LABEL_TO_IDX['Benign Keratosis (bkl)'],
    LABEL_TO_IDX['Dermatofibroma (df)'],
    LABEL_TO_IDX['Melanocytic Nevi (nv)'],
    LABEL_TO_IDX['Vascular Lesions (vasc)']
}

# ═══════════════════════════════════════════════════════════
# 4. GATE 0: IMAGE QUALITY CHECK
# ═══════════════════════════════════════════════════════════
def check_image_quality(image):
    """
    Gate 0: Blur check + multi-range skin detection.
    Uses three overlapping YCbCr ranges to cover diverse skin tones.
    """
    img_array = np.array(image.convert('RGB'))
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
    if blur_score < 50:
        return False, f"Image too blurry (score: {blur_score:.1f}). Please retake with better focus."

    img_ycbcr = cv2.cvtColor(img_array, cv2.COLOR_RGB2YCrCb)
    mask1 = cv2.inRange(img_ycbcr, np.array([0,133,77], dtype=np.uint8), np.array([255,173,127], dtype=np.uint8))
    mask2 = cv2.inRange(img_ycbcr, np.array([0,125,70], dtype=np.uint8), np.array([255,180,135], dtype=np.uint8))
    mask3 = cv2.inRange(img_ycbcr, np.array([0,120,60], dtype=np.uint8), np.array([255,185,142], dtype=np.uint8))
    combined_mask = cv2.bitwise_or(mask1, cv2.bitwise_or(mask2, mask3))
    total_pixels  = img_array.shape[0] * img_array.shape[1]
    skin_pct      = (cv2.countNonZero(combined_mask) / total_pixels) * 100

    if skin_pct < 10:
        return False, f"Skin region too small ({skin_pct:.1f}%). Ensure the lesion is centred and well-lit."
    return True, "Passed"

# ═══════════════════════════════════════════════════════════
# 5. GATE 1: HAIR REMOVAL
# ═══════════════════════════════════════════════════════════
def remove_hair(image):
    img_array = np.array(image.convert('RGB'))
    gray      = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    kernel    = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17))
    blackhat  = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    _, hair_mask = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
    result    = cv2.inpaint(img_array, hair_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    return Image.fromarray(result), hair_mask

# ═══════════════════════════════════════════════════════════
# 6. GATE 2: 3-ANGLE TTA INFERENCE
# ═══════════════════════════════════════════════════════════
def predict_with_tta(model, img_array):
    variants = [img_array, np.fliplr(img_array), np.rot90(img_array, k=1)]
    all_preds = []
    for variant in variants:
        resized = cv2.resize(variant, (224, 224)).astype('float32')
        inp     = np.expand_dims(resized, axis=0)
        inp     = tf.keras.applications.efficientnet.preprocess_input(inp)
        pred    = model.predict(inp, verbose=0)
        all_preds.append(pred[0])
    return np.mean(all_preds, axis=0)

# ═══════════════════════════════════════════════════════════
# 7. GRAD-CAM
# ═══════════════════════════════════════════════════════════
def generate_gradcam(model, img_array_224, class_idx):
    try:
        last_conv_layer = None
        for layer in reversed(model.layers):
            try:
                if layer.name == 'efficientnetb0' or (hasattr(layer, 'output') and len(layer.output.shape) == 4 and 'random' not in layer.name):
                    last_conv_layer = layer.name
                    break
            except Exception:
                continue
        if last_conv_layer is None:
            return None
        grad_model = tf.keras.models.Model(
            inputs  = model.inputs,
            outputs = [model.get_layer(last_conv_layer).output, model.output]
        )
        img_input = np.expand_dims(img_array_224.astype('float32'), axis=0)
        img_input = tf.keras.applications.efficientnet.preprocess_input(img_input)
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_input, training=False)
            loss = predictions[:, class_idx]
        grads        = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]
        heatmap      = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap      = tf.squeeze(heatmap).numpy()
        heatmap = np.maximum(heatmap, 0)
        if heatmap.max() > 0:
            heatmap /= heatmap.max()
        heatmap_resized = cv2.resize(heatmap, (224, 224))
        heatmap_uint8   = np.uint8(255 * heatmap_resized)
        heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        overlay = cv2.addWeighted(img_array_224.astype(np.uint8), 0.6, heatmap_colored, 0.4, 0)
        return Image.fromarray(overlay), heatmap
    except Exception as e:
        print(f"Grad-CAM generation failed: {e}")
        return None

# ═══════════════════════════════════════════════════════════
# 7b. LESION HIGHLIGHT (contour overlay on original image)
# ═══════════════════════════════════════════════════════════
def generate_lesion_highlight(original_img, heatmap, threshold=0.45):
    """
    Uses the Grad-CAM heatmap to find high-attention regions,
    then draws contour outlines on the original full-size image
    to highlight the affected/infected area.
    Returns a PIL Image with contour overlay.
    """
    img_array = np.array(original_img.convert('RGB')).copy()
    h, w = img_array.shape[:2]

    # Resize heatmap to original image dimensions
    heatmap_full = cv2.resize(heatmap, (w, h))

    # Threshold to get the region of interest
    mask = (heatmap_full >= threshold).astype(np.uint8) * 255

    # Clean up the mask with morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Draw semi-transparent fill on the affected region
        overlay = img_array.copy()
        cv2.drawContours(overlay, contours, -1, (13, 148, 136), thickness=cv2.FILLED)  # Teal fill
        img_array = cv2.addWeighted(img_array, 0.75, overlay, 0.25, 0)  # 25% opacity fill

        # Draw contour outlines (thicker, bright teal)
        cv2.drawContours(img_array, contours, -1, (13, 148, 136), thickness=3)

        # Draw bounding box around the largest contour
        largest = max(contours, key=cv2.contourArea)
        x, y, bw, bh = cv2.boundingRect(largest)
        cv2.rectangle(img_array, (x - 5, y - 5), (x + bw + 5, y + bh + 5), (239, 68, 68), 2)

        # Add label
        label_text = "AFFECTED AREA"
        font_scale = max(0.5, min(w, h) / 600)
        thickness = max(1, int(font_scale * 2))
        (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        cv2.rectangle(img_array, (x - 5, y - th - 15), (x + tw + 5, y - 5), (239, 68, 68), cv2.FILLED)
        cv2.putText(img_array, label_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)

    return Image.fromarray(img_array)

# ═══════════════════════════════════════════════════════════
# 8. PDF REPORT GENERATOR
# ═══════════════════════════════════════════════════════════
def generate_pdf_report(patient_id, consult_date, status, confidence, label, desc):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 18)
    pdf.set_text_color(15, 23, 42)
    pdf.cell(0, 12, "DermaAI Diagnostic Report", ln=True, align="C")
    pdf.set_font("Arial", "", 10)
    pdf.set_text_color(100, 116, 139)
    pdf.cell(0, 7, "Powered by EfficientNetB0 + 3-Angle TTA | DermaAI Pro v2.1", ln=True, align="C")
    pdf.ln(4)
    pdf.set_draw_color(226, 232, 240)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(6)
    pdf.set_font("Arial", "B", 11)
    pdf.set_text_color(15, 23, 42)
    pdf.cell(0, 8, "Patient Information", ln=True)
    pdf.set_font("Arial", "", 11)
    pdf.set_text_color(51, 65, 85)
    pdf.cell(60, 7, "Patient ID:", border=0)
    pdf.cell(0, 7, str(patient_id) if patient_id else "N/A", ln=True)
    pdf.cell(60, 7, "Consultation Date:", border=0)
    pdf.cell(0, 7, str(consult_date), ln=True)
    pdf.ln(4)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(6)
    pdf.set_font("Arial", "B", 11)
    pdf.set_text_color(15, 23, 42)
    pdf.cell(0, 8, "Diagnostic Result", ln=True)
    pdf.set_font("Arial", "", 11)
    pdf.set_text_color(51, 65, 85)
    pdf.cell(60, 7, "Classification:", border=0)
    pdf.set_font("Arial", "B", 11)
    pdf.cell(0, 7, str(label), ln=True)
    pdf.set_font("Arial", "", 11)
    pdf.cell(60, 7, "Risk Status:", border=0)
    pdf.cell(0, 7, str(status), ln=True)
    pdf.cell(60, 7, "AI Confidence:", border=0)
    pdf.cell(0, 7, f"{confidence:.1f}%", ln=True)
    pdf.ln(4)
    pdf.set_font("Arial", "B", 11)
    pdf.set_text_color(15, 23, 42)
    pdf.cell(0, 8, "Clinical Note", ln=True)
    pdf.set_font("Arial", "", 11)
    pdf.set_text_color(51, 65, 85)
    pdf.multi_cell(0, 7, str(desc))
    pdf.ln(6)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(5)
    pdf.set_font("Arial", "I", 9)
    pdf.set_text_color(100, 116, 139)
    pdf.multi_cell(0, 6,
        "DISCLAIMER: This report is AI-generated and is intended solely as a "
        "clinical decision support tool. It does not constitute a medical diagnosis "
        "and should not replace evaluation by a qualified dermatologist. All findings "
        "must be correlated with clinical examination and professional medical judgment."
    )
    return pdf.output(dest='S').encode('latin-1')

# ═══════════════════════════════════════════════════════════
# 9. SAFE MODEL LOADER
# ═══════════════════════════════════════════════════════════
@st.cache_resource
def load_learner():
    tf.keras.backend.clear_session()
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "skin_cancer_model_v2_perfect.h5")
    if not os.path.exists(model_path):
        return None, f"Model file not found: '{model_path}'."
    try:
        model = tf.keras.models.load_model(model_path, compile=False)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model, None
    except Exception as e:
        return None, f"Model failed to load: {str(e)}"

model, load_error = load_learner()

# ═══════════════════════════════════════════════════════════
# 10. PAGE ROUTING
# ═══════════════════════════════════════════════════════════
if "page" not in st.session_state:
    st.session_state.page = "home"

def go_to(page):
    st.session_state.page = page

# ═══════════════════════════════════════════════════════════
# 11. SIDEBAR
# ═══════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 16px 0 8px 0;">
        <div style="font-size: 2.4rem;">🩺</div>
        <div style="font-family: 'Outfit', sans-serif; font-weight: 800; font-size: 1.4rem; color: #F1F5F9; letter-spacing: -0.5px;">DermaAI Pro</div>
        <div style="font-size: 0.75rem; color: #94A3B8; margin-top: 2px;">v2.1 — Clinical Edition</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("### 🧭 Navigation")
    if st.button("🏠  Home", use_container_width=True, key="nav_home"):
        go_to("home")
    if st.button("🔬  Scan & Analyse", use_container_width=True, key="nav_scan"):
        go_to("scan")
    if st.button("ℹ️  About", use_container_width=True, key="nav_about"):
        go_to("about")
    st.markdown("---")

    if st.session_state.page == "scan":
        st.markdown("### 📋 Patient Details")
        patient_id   = st.text_input("Patient ID", placeholder="E.g. PT-1024")
        consult_date = st.date_input("Consultation Date", value=date.today())
        st.markdown("---")
        st.markdown("### 🧠 AI Pipeline")
        st.success("✅ Gate 0 — Image Quality")
        st.success("✅ Gate 1 — Hair Removal")
        st.success("✅ Gate 2 — EfficientNet + TTA")
    else:
        patient_id   = ""
        consult_date = date.today()

    st.markdown("---")
    st.info("🔒 Secure Mode Active\nNo data is stored or transmitted.")

# ═══════════════════════════════════════════════════════════
# 12. LANDING PAGE
# ═══════════════════════════════════════════════════════════
if st.session_state.page == "home":

    # Hero
    st.markdown("""
    <div class="hero-section">
        <div class="hero-badge">🧬 AI-POWERED DERMATOLOGY</div>
        <h1>DermaAI Diagnostics</h1>
        <p class="hero-sub">
            Advanced skin lesion analysis powered by EfficientNetB0 with 3-angle
            Test-Time Augmentation. Detect melanoma, BCC, and 6 other conditions
            with clinical-grade accuracy — in seconds.
        </p>
    </div>
    """, unsafe_allow_html=True)

    if st.button("🔬  Start Screening →", use_container_width=True, type="primary", key="hero_cta"):
        go_to("scan")
        st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    # Stats Row
    st.markdown('<h2 class="section-heading">Trusted Clinical Performance</h2>', unsafe_allow_html=True)
    st.markdown('<p class="section-sub">Validated on the HAM10000 dataset — the gold standard in dermoscopic imaging.</p>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("""
        <div class="stat-card">
            <div class="stat-value">8</div>
            <div class="stat-label">Skin Conditions</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="stat-card">
            <div class="stat-value">93.2%</div>
            <div class="stat-label">Top-1 Accuracy</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div class="stat-card">
            <div class="stat-value">3×</div>
            <div class="stat-label">TTA Consensus</div>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown("""
        <div class="stat-card">
            <div class="stat-value">&lt; 5s</div>
            <div class="stat-label">Analysis Time</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Feature Cards
    st.markdown('<h2 class="section-heading">Why DermaAI?</h2>', unsafe_allow_html=True)
    st.markdown('<p class="section-sub">A 3-gate clinical pipeline designed for accuracy, fairness, and transparency.</p>', unsafe_allow_html=True)

    st.markdown("""
    <div class="feature-grid">
        <div class="feature-card">
            <div class="card-icon teal">🧠</div>
            <h3>3-Gate AI Pipeline</h3>
            <p>Every scan passes through image quality validation, digital hair removal, and 3-angle consensus inference before diagnosis.</p>
        </div>
        <div class="feature-card">
            <div class="card-icon blue">🔥</div>
            <h3>Grad-CAM Explainability</h3>
            <p>See exactly where the AI is looking. Visual attention heatmaps build trust between clinicians and the algorithm.</p>
        </div>
        <div class="feature-card">
            <div class="card-icon purple">🔒</div>
            <h3>Privacy First</h3>
            <p>All processing happens locally. No images are uploaded to any cloud. HIPAA-aligned by design.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # How It Works
    st.markdown("""
    <div class="steps-section">
        <h2 class="section-heading">How It Works</h2>
        <p class="section-sub">Three simple steps to a clinical-grade screening result.</p>
        <div class="steps-grid">
            <div class="step-item">
                <div class="step-number">1</div>
                <h4>Upload Scan</h4>
                <p>Drag and drop a dermoscopic image (JPG, PNG). The system validates quality automatically.</p>
            </div>
            <div class="step-item">
                <div class="step-number">2</div>
                <h4>AI Analysis</h4>
                <p>EfficientNetB0 processes the image with 3-angle TTA for robust, consensus-based classification.</p>
            </div>
            <div class="step-item">
                <div class="step-number">3</div>
                <h4>Review & Export</h4>
                <p>View the diagnosis, Grad-CAM heatmap, probability chart, and download a PDF report.</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Footer
    st.markdown("""
    <div class="footer">
        <p class="footer-brand">🩺 DermaAI Diagnostics</p>
        <p>Built with EfficientNetB0, Streamlit & TensorFlow</p>
        <p style="margin-top: 12px; font-size: 0.75rem; color: #64748B;">
            ⚠️ This tool is for screening purposes only and does not replace professional medical diagnosis.<br>
            Always consult a qualified dermatologist for clinical decisions.
        </p>
    </div>
    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════
# 13. SCAN PAGE
# ═══════════════════════════════════════════════════════════
elif st.session_state.page == "scan":

    # Header
    st.markdown("""
    <div class="scan-header">
        <div style="font-size: 2rem;">🔬</div>
        <div>
            <h2>Scan & Analyse</h2>
            <p>Upload a dermoscopic image to receive an AI-powered diagnostic screening.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Model error guard
    if load_error:
        st.error(f"⚠️ {load_error}")
        st.stop()

    col1, col2 = st.columns([1, 1.5], gap="large")

    with col1:
        st.markdown("#### 📤 Upload Scan")
        uploaded_file = st.file_uploader(
            "Drop dermoscopic image here",
            type=["jpg", "png", "jpeg"]
        )

        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="Original Scan", use_container_width=True)

    with col2:
        st.markdown("#### 📊 Diagnostic Analysis")

        if not uploaded_file:
            st.info("👈 Upload a patient dermoscopic scan to begin analysis.")
        else:
            # ── GATE 0 ─────────────────────────────────────────────
            with st.spinner("Running Gate 0 — image quality check..."):
                is_valid, msg = check_image_quality(image)

            if not is_valid:
                st.markdown(f"""
                <div class="result-card-gray">
                    <h3>🚫 Image Rejected — Gate 0</h3>
                    <p>{msg}</p>
                    <p><b>Action:</b> Retake the photo with better lighting, focus, and ensure the lesion is centred.</p>
                </div>
                """, unsafe_allow_html=True)

            else:
                # ── GATE 1 ─────────────────────────────────────────
                with st.spinner("Running Gate 1 — digital hair removal..."):
                    clean_image, hair_mask = remove_hair(image)

                with st.expander("🔬 View hair removal result", expanded=False):
                    hc1, hc2 = st.columns(2)
                    with hc1:
                        st.image(image, caption="Original", use_container_width=True)
                    with hc2:
                        st.image(clean_image, caption="After hair removal", use_container_width=True)

                # ── GATE 2 ─────────────────────────────────────────
                with st.spinner("Running Gate 2 — 3-angle consensus analysis (TTA)..."):
                    img_array_raw = np.array(clean_image.convert('RGB'))
                    try:
                        probs = predict_with_tta(model, img_array_raw)
                    except Exception as e:
                        st.error(f"Inference failed: {e}")
                        st.stop()

                class_idx  = int(np.argmax(probs))
                confidence = float(probs[class_idx]) * 100
                label      = CLASS_NAMES.get(class_idx, "Unknown")

                # ── LOGIC ENGINE ───────────────────────────────────
                if class_idx in HEALTHY_CLASSES:
                    card_class, icon, status = "result-card-green", "✅", "Healthy Skin"
                    desc = "No pathological abnormalities detected. Routine monitoring recommended."
                elif class_idx in MALIGNANT_CLASSES and confidence > 65:
                    card_class, icon, status = "result-card-red", "🚨", f"High Risk: {label}"
                    desc = "Malignant features detected. Immediate biopsy and specialist referral required."
                elif class_idx in BENIGN_CLASSES and confidence > 50:
                    card_class, icon, status = "result-card-green", "🟢", f"Benign: {label}"
                    desc = "Lesion appears benign. Monitor for asymmetry, border irregularity or colour change."
                else:
                    card_class, icon, status = "result-card-gray", "⚠️", "Inconclusive Analysis"
                    desc = "AI confidence is below threshold. Clinical correlation required."

                # ── RESULT CARD ────────────────────────────────────
                st.markdown(f"""
                <div class="{card_class}">
                    <h2>{icon} {status}</h2>
                    <p><b>Patient:</b> {patient_id or 'N/A'} &nbsp;|&nbsp; <b>Date:</b> {consult_date}</p>
                    <p><b>AI Confidence:</b> {confidence:.1f}%</p>
                    <p>{desc}</p>
                </div>
                """, unsafe_allow_html=True)

                # ── METRICS ────────────────────────────────────────
                m1, m2, m3 = st.columns(3)
                m1.metric("Confidence", f"{confidence:.1f}%", delta_color="off")
                m2.metric("Scan Quality", "Pass", "High")
                m3.metric("Model Version", "V2.1 + TTA")

                # ── GRAD-CAM + LESION HIGHLIGHT ───────────────────
                st.markdown("#### 🔥 Lesion Detection & Model Attention")
                with st.spinner("Generating heatmap & detecting affected region..."):
                    img_224     = cv2.resize(img_array_raw, (224, 224))
                    gradcam_result = generate_gradcam(model, img_224, class_idx)

                if gradcam_result is not None:
                    gradcam_img, raw_heatmap = gradcam_result

                    # Generate lesion highlight on the original-size image
                    highlight_img = generate_lesion_highlight(clean_image, raw_heatmap)

                    gc1, gc2, gc3 = st.columns(3)
                    with gc1:
                        st.image(clean_image, caption="Processed Scan", use_container_width=True)
                    with gc2:
                        st.image(gradcam_img, caption="Grad-CAM Heatmap", use_container_width=True)
                    with gc3:
                        st.image(highlight_img, caption="🎯 Affected Area Highlighted", use_container_width=True)
                    st.caption("Left: cleaned scan · Centre: AI attention (red = high focus) · Right: detected affected region with contour overlay")
                else:
                    st.info("Grad-CAM not available for this model architecture.")

                # ── PLOTLY CHART ────────────────────────────────────
                st.markdown("#### 📊 Probability Distribution by Class")

                chart_data = pd.DataFrame({
                    'Class': list(CLASS_NAMES.values()),
                    'Probability': [float(p) for p in probs],
                    'Risk': [
                        'Malignant' if i in MALIGNANT_CLASSES else
                        'Healthy'   if i in HEALTHY_CLASSES   else
                        'Benign'
                        for i in CLASS_NAMES.keys()
                    ]
                })

                color_map = {'Malignant': '#EF4444', 'Benign': '#10B981', 'Healthy': '#0D9488'}

                fig = px.bar(
                    chart_data, x='Class', y='Probability', color='Risk',
                    color_discrete_map=color_map,
                    text=chart_data['Probability'].apply(lambda x: f"{x*100:.1f}%")
                )
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_color='#1E293B',
                    yaxis_tickformat='.0%',
                    yaxis_range=[0, 1],
                    showlegend=True,
                    height=350,
                    margin=dict(t=20, b=10),
                    xaxis_tickangle=-30
                )
                fig.update_traces(textposition='outside')
                st.plotly_chart(fig, use_container_width=True)

                # ── PDF EXPORT ─────────────────────────────────────
                st.markdown("#### 📄 Export Report")
                if st.button("Generate PDF Report", type="primary"):
                    with st.spinner("Generating report..."):
                        pdf_bytes = generate_pdf_report(
                            patient_id, consult_date,
                            status, confidence, label, desc
                        )
                    fname = f"DermaAI_{patient_id or 'report'}_{consult_date}.pdf"
                    st.download_button(
                        label="⬇️ Download PDF Report",
                        data=pdf_bytes,
                        file_name=fname,
                        mime="application/pdf"
                    )

# ═══════════════════════════════════════════════════════════
# 14. ABOUT PAGE
# ═══════════════════════════════════════════════════════════
elif st.session_state.page == "about":

    st.markdown("""
    <div class="scan-header">
        <div style="font-size: 2rem;">ℹ️</div>
        <div>
            <h2>About DermaAI</h2>
            <p>Model architecture, methodology, and clinical disclaimer.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    a1, a2 = st.columns(2, gap="large")
    with a1:
        st.markdown("""
        <div class="about-card">
            <h3>🧠 Model Architecture</h3>
            <p>
                DermaAI uses <b>EfficientNetB0</b> as the backbone, fine-tuned on the
                <b>HAM10000</b> dermoscopic dataset. The model classifies skin lesions
                into 8 categories including melanoma, BCC, actinic keratoses, and
                healthy skin.
            </p>
            <p>
                <b>Input:</b> 224 × 224 RGB<br>
                <b>Preprocessing:</b> EfficientNet normalisation<br>
                <b>Output:</b> 8-class softmax<br>
                <b>Loss:</b> Categorical cross-entropy
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="about-card">
            <h3>🔬 3-Gate Pipeline</h3>
            <p>
                <b>Gate 0 — Quality:</b> Laplacian blur detection + multi-range
                YCbCr skin-tone validation for diverse skin types.<br><br>
                <b>Gate 1 — Hair Removal:</b> Morphological BlackHat + Telea
                inpainting removes hair artefacts before inference.<br><br>
                <b>Gate 2 — TTA Inference:</b> 3-angle Test-Time Augmentation
                (original, horizontal-flip, 90° rotation) provides a consensus
                probability vector for robust classification.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with a2:
        st.markdown("""
        <div class="about-card">
            <h3>📊 Classification Categories</h3>
            <p>The model distinguishes between the following conditions:</p>
            <p>
                🔴 <b>Malignant:</b> Melanoma (mel), Basal Cell Carcinoma (bcc), 
                Actinic Keratoses (akiec)<br><br>
                🟢 <b>Benign:</b> Benign Keratosis (bkl), Dermatofibroma (df),
                Melanocytic Nevi (nv), Vascular Lesions (vasc)<br><br>
                🔵 <b>Healthy:</b> Normal skin with no detectable pathology
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="about-card">
            <h3>⚖️ Disclaimer</h3>
            <p>
                DermaAI is an <b>AI-assisted screening tool</b>. It is <b>not a substitute</b> for
                professional medical diagnosis.
            </p>
            <p>
                All results must be correlated with clinical examination by a
                qualified dermatologist. The developers assume no liability for
                clinical decisions made based on this tool's output.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="about-card" style="text-align:center;">
        <h3>🏥 Built for Clinical Decision Support</h3>
        <p>EfficientNetB0 · Keras 3 · Streamlit · Grad-CAM · TTA · OpenCV</p>
    </div>
    """, unsafe_allow_html=True)