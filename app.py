import streamlit as st
import numpy as np
import pickle
from PIL import Image
from datetime import datetime
from tensorflow.keras.applications.resnet50 import preprocess_input

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="AgriSmart AI - Decision Support System",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# CUSTOM STYLING
# =========================================================
st.markdown("""
    <style>
    .main-title {
        font-size: 2.4rem;
        font-weight: 800;
        color: #2E7D32;
        margin-bottom: 0px;
    }
    .subtitle {
        font-size: 1.05rem;
        color: #6b7280;
        margin-top: 0px;
        margin-bottom: 1.2rem;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.3rem;
        color: #2E7D32;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-title">🌱 AgriSmart AI Decision Support System</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Plant Pathology Diagnosis · Soil Chemistry Advisory · Regional Weather Insights</p>', unsafe_allow_html=True)

# =========================================================
# LOAD MODEL (cached so it loads only once)
# =========================================================
@st.cache_resource
def load_disease_model():
    try:
        with open('resnet_plant_disease_model.pkl', 'rb') as f:
            data = pickle.load(f)
        return data['model'], data['class_indices']
    except Exception as e:
        return None, str(e)

disease_model, class_indices = load_disease_model()

# =========================================================
# TREATMENT DATABASE — matches EXACTLY the 19 trained classes
# =========================================================
treatment_db = {
    "Apple___Apple_scab": "Apply protective fungicides such as Captan or Mancozeb during early bloom. Rake and destroy fallen leaves to eliminate overwintering fungal spores.",
    "Apple___Black_rot": "Prune out infected twigs and remove mummified fruit from the tree and ground. Apply copper-based fungicides on a regular schedule through the growing season.",
    "Apple___Cedar_apple_rust": "Remove nearby juniper/cedar trees where feasible, since they host the fungus. Apply protective fungicides (myclobutanil or copper-based) starting at bud break.",
    "Apple___healthy": "No disease detected. Maintain a consistent irrigation schedule and good canopy ventilation to keep the tree in this healthy state.",

    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": "Rotate with non-host crops and use resistant hybrids where available. Apply strobilurin or triazole fungicides if lesions appear before tasseling.",
    "Corn_(maize)___Common_rust_": "Plant rust-resistant hybrids. Apply foliar fungicides containing azoxystrobin or propiconazole at the first sign of orange pustules.",
    "Corn_(maize)___Northern_Leaf_Blight": "Use resistant hybrids and practice deep tillage to bury infected residue. Apply fungicides early if the disease appears before flowering.",
    "Corn_(maize)___healthy": "Crop is in healthy condition. Maintain balanced nitrogen application, especially during the silking stage.",

    "Grape___Black_rot": "Remove mummified berries and infected canes during dormant pruning. Apply protectant fungicides (mancozeb or myclobutanil) starting at early shoot growth.",
    "Grape___Esca_(Black_Measles)": "Prune out and destroy infected wood during dry weather. There is no curative spray — focus on vine stress reduction and sanitation.",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": "Improve canopy airflow through proper leaf pulling. Apply copper-based fungicides at the first sign of leaf spotting.",
    "Grape___healthy": "Vine is healthy. Continue routine canopy management and balanced fertilization.",

    "Potato___Early_blight": "Apply protective fungicides such as chlorothalonil or mancozeb. Maintain adequate plant spacing and practice a 3-year crop rotation.",
    "Potato___Late_blight": "Apply systemic fungicides containing metalaxyl or cymoxanil immediately. Destroy severely infected foliage to prevent spore spread — this disease can devastate a field quickly.",

    "Tomato___Bacterial_spot": "Apply copper-based bactericides mixed with mancozeb. Avoid overhead irrigation and working in fields when plants are wet.",
    "Tomato___Late_blight": "Apply preventive copper-based fungicide sprays immediately and improve field air drainage. Remove and destroy severely affected vines to halt spread.",
    "Tomato___Septoria_leaf_spot": "Remove and destroy lower infected leaves. Apply chlorothalonil-based fungicides on a 7-10 day schedule during humid conditions.",
    "Tomato___Target_Spot": "Improve plant spacing and airflow. Apply protectant fungicides (chlorothalonil or azoxystrobin) at early symptom onset.",
    "Tomato___Tomato_mosaic_virus": "There is no chemical cure — remove and destroy infected plants immediately. Disinfect tools between plants and control aphid/whitefly vectors to limit spread."
}

def get_severity(disease_name: str) -> str:
    """Simple heuristic to badge the severity of a detected condition."""
    if "healthy" in disease_name.lower():
        return "✅ Healthy"
    if "mosaic_virus" in disease_name.lower() or "late_blight" in disease_name.lower() or "esca" in disease_name.lower():
        return "🔴 High Risk"
    return "🟠 Moderate Risk"

# =========================================================
# SESSION STATE — keeps a history of diagnoses in this session
# =========================================================
if "history" not in st.session_state:
    st.session_state.history = []

# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.header("📊 Session Summary")
    st.metric("Diagnoses Run", len(st.session_state.history))
    if disease_model is not None:
        st.success(f"Model loaded ✅ ({len(class_indices)} classes)")
    else:
        st.error("Model failed to load")

    st.markdown("---")
    st.subheader("🕘 Recent History")
    if st.session_state.history:
        for entry in reversed(st.session_state.history[-5:]):
            st.caption(f"{entry['time']} — **{entry['disease']}** ({entry['confidence']:.1f}%)")
    else:
        st.caption("No diagnoses yet this session.")

    st.markdown("---")
    if st.button("🗑️ Clear History"):
        st.session_state.history = []
        st.rerun()

# =========================================================
# NAVIGATION TABS
# =========================================================
tab1, tab2, tab3 = st.tabs(["🍃 Leaf Disease Diagnosis", "🧪 Soil Advisory", "🌤️ Weather Engine"])

# ---------------------------------------------------------
# TAB 1: LEAF DISEASE DIAGNOSIS
# ---------------------------------------------------------
with tab1:
    st.subheader("Plant Leaf Pathogen Identification")
    st.caption("Upload a leaf image to run AI-powered disease classification and get an actionable treatment plan.")

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        uploaded_file = st.file_uploader("Upload Leaf Image", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            img = Image.open(uploaded_file)
            st.image(img, caption="Uploaded Leaf Image", use_container_width=True)

    with col2:
        if uploaded_file is not None:
            if disease_model is None:
                st.error("Model could not be loaded. Check that resnet_plant_disease_model.pkl is in the app folder.")
            else:
                with st.spinner("Analyzing leaf image..."):
                    # --- Preprocessing MUST match training exactly ---
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    img_resized = img.resize((224, 224))
                    img_array = np.array(img_resized).astype("float32")
                    img_array = preprocess_input(img_array)          # same as training (NOT /255.0)
                    img_array = np.expand_dims(img_array, axis=0)

                    preds = disease_model.predict(img_array, verbose=0)
                    pred_idx = np.argmax(preds[0])
                    confidence = float(np.max(preds[0])) * 100

                    labels_map = {v: k for k, v in class_indices.items()}
                    raw_class = labels_map.get(pred_idx, f"Class_{pred_idx}")
                    clean_name = raw_class.replace("___", " - ").replace("_", " ")

                st.markdown("### 📊 Diagnostic Results")
                m1, m2 = st.columns(2)
                m1.metric("Identified Condition", clean_name)
                m2.metric("Confidence", f"{confidence:.2f}%")

                st.progress(min(int(confidence), 100), text=f"Model Confidence: {confidence:.1f}%")
                st.markdown(f"**Severity:** {get_severity(raw_class)}")

                st.markdown("---")
                st.markdown("### 📋 Recommended Action Plan")
                treatment = treatment_db.get(
                    raw_class,
                    "Monitor the crop closely and consult a local agricultural extension officer for targeted treatment guidance."
                )
                st.info(treatment)

                # Save to session history
                st.session_state.history.append({
                    "time": datetime.now().strftime("%H:%M:%S"),
                    "disease": clean_name,
                    "confidence": confidence
                })

                # Downloadable report
                report_text = (
                    f"AgriSmart AI - Diagnosis Report\n"
                    f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
                    f"Detected Condition: {clean_name}\n"
                    f"Confidence: {confidence:.2f}%\n"
                    f"Severity: {get_severity(raw_class)}\n\n"
                    f"Recommended Action Plan:\n{treatment}\n"
                )
                st.download_button(
                    "⬇️ Download Diagnosis Report (.txt)",
                    data=report_text,
                    file_name=f"diagnosis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )
        else:
            st.info("Upload a leaf image on the left to see diagnostic results here.")

# ---------------------------------------------------------
# TAB 2: SOIL ADVISORY
# ---------------------------------------------------------
with tab2:
    st.subheader("Soil Chemistry & Dosing Strategy")
    st.caption("Enter your soil lab test values to receive a customized nutrient advisory.")

    c1, c2, c3, c4 = st.columns(4)
    n_val = c1.number_input("Nitrogen (N) - mg/kg", min_value=0, max_value=300, value=140)
    p_val = c2.number_input("Phosphorus (P) - mg/kg", min_value=0, max_value=300, value=45)
    k_val = c3.number_input("Potassium (K) - mg/kg", min_value=0, max_value=300, value=180)
    ph_val = c4.number_input("Soil pH Level", min_value=0.0, max_value=14.0, value=6.5, step=0.1)

    if st.button("Compute Soil Nutrient Advisory", type="primary"):
        st.markdown("### 🧪 Soil Health Assessment")

        advisories = []
        if n_val < 100:
            advisories.append("• **Nitrogen Deficiency:** Apply Urea (46% N) at the recommended dosage to boost vegetative growth.")
        elif n_val > 200:
            advisories.append("• **Excess Nitrogen:** Reduce nitrogenous fertilizer to avoid root burn and excessive leaf growth.")

        if p_val < 30:
            advisories.append("• **Phosphorus Deficiency:** Apply Single Super Phosphate (SSP) to support root development.")
        if k_val < 150:
            advisories.append("• **Potassium Deficiency:** Apply Muriate of Potash (MOP) to improve disease resistance.")

        if ph_val < 6.0:
            advisories.append("• **Acidic Soil:** Apply agricultural limestone to raise the pH.")
        elif ph_val > 7.5:
            advisories.append("• **Alkaline Soil:** Apply elemental sulfur or gypsum to lower the pH.")

        if not advisories:
            st.success("✅ Soil chemical composition is optimal for general crop cultivation.")
        else:
            for item in advisories:
                st.warning(item)

# ---------------------------------------------------------
# TAB 3: WEATHER ENGINE
# ---------------------------------------------------------
with tab3:
    st.subheader("Regional Weather Advisory")
    st.caption("Demo data shown below. Connect a live weather API (e.g. OpenWeatherMap) to replace these values with real-time data.")

    city = st.text_input("Enter City / Region", value="Sahiwal")

    if st.button("Fetch Advisory"):
        st.markdown(f"#### 📍 Regional Report: **{city}**")

        w1, w2, w3 = st.columns(3)
        w1.metric("Temperature", "31 °C", delta="1.2 °C")
        w2.metric("Humidity", "62%")
        w3.metric("Rain Probability", "15%")

        st.success("🌦️ **Agronomic Insight:** Weather conditions are favorable for spraying. Low risk of immediate rainfall wash-off.")