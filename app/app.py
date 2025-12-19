import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import warnings

warnings.filterwarnings("ignore")

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Cognitive Self-Assessment System",
    page_icon="🧠",
    layout="centered"
)

# --------------------------------------------------
# Custom Styling (Visual Appeal)
# --------------------------------------------------
st.markdown("""
<style>
.main {
    background-color: #f7f9fc;
}
h1, h2, h3 {
    color: #2c3e50;
}
.metric-box {
    padding: 15px;
    border-radius: 10px;
    background-color: #ffffff;
    box-shadow: 0px 4px 10px rgba(0,0,0,0.05);
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# Title
# --------------------------------------------------
st.title("🧠 Cognitive Self-Assessment System")
st.write(
    """
    This application provides a **personalized cognitive and decision-making analysis**
    using Machine Learning and Cognitive Science principles.
    """
)

st.markdown("---")

# --------------------------------------------------
# Load Models (Robust Path Handling)
# --------------------------------------------------
@st.cache_resource
def load_models():
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    cognitive_model = joblib.load(os.path.join(BASE_DIR, "models", "cognitive_kmeans.pkl"))
    cognitive_scaler = joblib.load(os.path.join(BASE_DIR, "models", "cognitive_scaler.pkl"))
    decision_model = joblib.load(os.path.join(BASE_DIR, "models", "decision_rf.pkl"))
    decision_feature_count = joblib.load(os.path.join(BASE_DIR, "models", "decision_feature_count.pkl"))

    return cognitive_model, cognitive_scaler, decision_model, decision_feature_count


cognitive_model, cognitive_scaler, decision_model, decision_feature_count = load_models()

# --------------------------------------------------
# User Inputs
# --------------------------------------------------
st.header("🧠 Mental & Cognitive Inputs")

work_interfere = st.selectbox(
    "Does mental stress interfere with your work?",
    ["Never", "Rarely", "Sometimes", "Often"]
)

mental_consequence = st.selectbox(
    "Do you worry about mental health consequences at work?",
    ["No", "Yes"]
)

treatment = st.selectbox(
    "Have you sought mental health treatment?",
    ["No", "Yes"]
)

benefits = st.selectbox(
    "Does your workplace provide mental health benefits?",
    ["No", "Yes"]
)

care_options = st.selectbox(
    "Are you aware of mental health care options?",
    ["No", "Yes"]
)

st.header("📱 Digital Behavior Inputs")

daily_screen_time = st.slider("Daily Screen Time (minutes)", 30, 900, 300)
sleep_hours = st.slider("Average Sleep Hours", 3, 10, 7)
focus_score = st.slider("Focus Score (1 = Low, 10 = High)", 1, 10, 6)
mood_score = st.slider("Mood Score (1 = Low, 10 = High)", 1, 10, 6)
anxiety_level = st.slider("Anxiety Level (1 = Low, 10 = High)", 1, 10, 4)
digital_wellbeing = st.slider("Digital Wellbeing Score (1 = Low, 10 = High)", 1, 10, 6)

# --------------------------------------------------
# Cognitive Score (User-Driven)
# --------------------------------------------------
def compute_cognitive_score():
    score = 0
    score += (10 - anxiety_level) * 5
    score += focus_score * 5
    score += mood_score * 3
    score += sleep_hours * 4
    score += digital_wellbeing * 3
    return min(int(score), 100)

# --------------------------------------------------
# Encode Cognitive Input
# --------------------------------------------------
def encode_cognitive_input():
    return pd.DataFrame([{
        "work_interfere": ["Never", "Rarely", "Sometimes", "Often"].index(work_interfere),
        "mental_health_consequence": ["No", "Yes"].index(mental_consequence),
        "treatment": ["No", "Yes"].index(treatment),
        "benefits": ["No", "Yes"].index(benefits),
        "care_options": ["No", "Yes"].index(care_options),
        "daily_screen_time_min": daily_screen_time,
        "sleep_hours": sleep_hours,
        "focus_score": focus_score,
        "mood_score": mood_score,
        "anxiety_level": anxiety_level,
        "digital_wellbeing_score": digital_wellbeing
    }])

# --------------------------------------------------
# Decision Vector Builder
# --------------------------------------------------
def build_decision_vector():
    base = np.array([
        daily_screen_time,
        sleep_hours,
        focus_score,
        mood_score,
        anxiety_level,
        digital_wellbeing
    ], dtype=float)

    if len(base) < decision_feature_count:
        base = np.pad(base, (0, decision_feature_count - len(base)))
    else:
        base = base[:decision_feature_count]

    return base.reshape(1, -1)

# --------------------------------------------------
# Run Analysis
# --------------------------------------------------
st.markdown("---")

if st.button("🔍 Analyze My Profile"):

    # Cognitive Score
    cognitive_score = compute_cognitive_score()

    st.subheader("🧠 Cognitive Readiness Score")
    st.progress(cognitive_score / 100)
    st.markdown(f"### **{cognitive_score} / 100**")

    # Cognitive Cluster Prediction
    user_df = encode_cognitive_input()
    scaled_user = cognitive_scaler.transform(user_df)
    cognitive_cluster = cognitive_model.predict(scaled_user)[0]

    cluster_map = {
        0: "Balanced Cognitive State",
        1: "High Cognitive Load & Anxiety",
        2: "Low Focus & Digital Fatigue"
    }

    st.subheader("🧩 Cognitive Profile (ML-Based)")
    st.success(cluster_map.get(cognitive_cluster))

    # Decision Style Prediction
    decision_vector = build_decision_vector()
    decision_style = decision_model.predict(decision_vector)[0]

    st.subheader("🎯 Decision-Making Style")
    st.info(decision_style)

    # Visualization
    st.subheader("📊 Self-Analysis Summary")

    chart_df = pd.DataFrame({
        "Metric": ["Focus", "Mood", "Sleep", "Anxiety", "Wellbeing"],
        "Score": [focus_score, mood_score, sleep_hours, anxiety_level, digital_wellbeing]
    })

    st.bar_chart(chart_df.set_index("Metric"))

    # Explanation
    st.subheader("📌 Why This Result?")

    st.write(
        f"""
        - Your **cognitive readiness score** is derived directly from your inputs,
          reflecting your current mental and behavioral state.
        - Higher **focus, sleep, and wellbeing** improved your score.
        - Elevated **anxiety levels** reduced overall readiness.
        - The ML model classified you into a **{cluster_map.get(cognitive_cluster)}**
          based on patterns learned from real-world data.
        - Your decision-making style (**{decision_style}**) reflects how emotional
          and cognitive factors influence choices.
        """
    )
