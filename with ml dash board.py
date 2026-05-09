import streamlit as st
import pandas as pd
import time
import os
import joblib
from datetime import datetime

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="AI-Powered Live Grid Monitor",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =========================================================
# RESPONSIVE CSS
# =========================================================

st.markdown("""
<style>

/* -------- Main Background -------- */

.stApp {
    background-color: #050816;
    color: white;
}

/* -------- Hide Streamlit Menu -------- */

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

header {
    visibility: hidden;
}

/* -------- Main Title -------- */

.main-title {
    font-size: 3vw;
    font-weight: 800;
    color: white;
}

.ai-text {
    font-size: 1.5vw;
    font-weight: 600;
    color: #00ffcc;
}

/* -------- Cards -------- */

.metric-card {
    background: #111827;
    padding: 20px;
    border-radius: 14px;
    text-align: center;
    border: 1px solid #1f2937;
    margin-bottom: 10px;
}

.metric-title {
    font-size: 1rem;
    color: #9ca3af;
}

.metric-value {
    font-size: 2rem;
    font-weight: bold;
    color: white;
}

/* -------- Feeders -------- */

.normal-box {
    background-color: #064e3b;
    padding: 14px;
    border-radius: 12px;
    text-align: center;
    font-weight: bold;
    color: #4ade80;
    border: 1px solid #166534;
}

.fault-box {
    background-color: #7f1d1d;
    padding: 14px;
    border-radius: 12px;
    text-align: center;
    font-weight: bold;
    color: #fca5a5;
    border: 1px solid #dc2626;
}

/* -------- Scenario -------- */

.scenario-box {
    background-color: #1f2937;
    padding: 12px;
    border-radius: 10px;
    text-align: center;
    font-weight: bold;
    color: white;
}

.active-scenario {
    background-color: #ef4444;
    color: white;
}

/* -------- Warning -------- */

.warning-box {
    background-color: #991b1b;
    border-radius: 14px;
    padding: 18px;
    margin-bottom: 12px;
    border: 2px solid red;
    color: white;
}

.warning-title {
    font-size: 22px;
    font-weight: 900;
    text-align: center;
}

.warning-text {
    text-align: center;
    font-size: 18px;
    font-weight: bold;
}

/* -------- Mobile -------- */

@media screen and (max-width: 768px) {

    .main-title {
        font-size: 7vw;
    }

    .ai-text {
        font-size: 4vw;
    }

    .metric-value {
        font-size: 1.4rem;
    }

    .warning-title {
        font-size: 18px;
    }

    .warning-text {
        font-size: 15px;
    }
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# FILE PATHS
# =========================================================

DATASET_PATH = "AI_Grid_Anomaly_Dataset_for_dash_board.csv"

MODEL_PATH = "xgb_model.pkl"

ENCODER_PATH = "label_encoder.pkl"

# =========================================================
# CHECK FILES
# =========================================================

required_files = [
    DATASET_PATH,
    MODEL_PATH,
    ENCODER_PATH
]

for file in required_files:

    if not os.path.exists(file):

        st.error(f"❌ Missing File: {file}")
        st.stop()

# =========================================================
# LOAD DATA
# =========================================================

@st.cache_data
def load_data():

    return pd.read_csv(DATASET_PATH)

df = load_data()

# =========================================================
# LOAD MODEL
# =========================================================

model = joblib.load(MODEL_PATH)

label_encoder = joblib.load(ENCODER_PATH)

# =========================================================
# SESSION STATES
# =========================================================

if "row_index" not in st.session_state:
    st.session_state.row_index = 0

if "warning_history" not in st.session_state:
    st.session_state.warning_history = []

# =========================================================
# CURRENT ROW
# =========================================================

row = df.iloc[st.session_state.row_index]

# =========================================================
# ML PREDICTION
# =========================================================

input_data = pd.DataFrame([{

    "Voltage": row["Voltage"],
    "Current": row["Current"],
    "Transformer_kW": row["Transformer_kW"]

}])

prediction_encoded = model.predict(input_data)[0]

prediction = label_encoder.inverse_transform(
    [prediction_encoded]
)[0]

# =========================================================
# FEEDER
# =========================================================

faulty_f = "F1"

if pd.notna(row["Fault_Feeder"]):

    faulty_f = str(row["Fault_Feeder"]).strip()

# =========================================================
# MAIN LAYOUT
# =========================================================

left_main, right_main = st.columns([3, 1])

# =========================================================
# LEFT SIDE
# =========================================================

with left_main:

    col1, col2 = st.columns([3, 1])

    with col1:

        st.markdown(
            '<div class="main-title">⚡ AI-Powered Live Grid Monitor</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            f'<div class="ai-text">AI Prediction: {prediction}</div>',
            unsafe_allow_html=True
        )

    with col2:

        st.markdown(
            f"## 🕒 {datetime.now().strftime('%H:%M:%S')}"
        )

        st.caption(
            f"📅 {datetime.now().strftime('%Y-%m-%d')}"
        )

    st.write("")

    # =====================================================
    # METRICS
    # =====================================================

    m1, m2, m3, m4 = st.columns(4)

    with m1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Voltage</div>
            <div class="metric-value">{row['Voltage']:.2f} V</div>
        </div>
        """, unsafe_allow_html=True)

    with m2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Current</div>
            <div class="metric-value">{row['Current']:.2f} A</div>
        </div>
        """, unsafe_allow_html=True)

    with m3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Power</div>
            <div class="metric-value">{row['Transformer_kW']:.2f} kW</div>
        </div>
        """, unsafe_allow_html=True)

    with m4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">PF</div>
            <div class="metric-value">0.88</div>
        </div>
        """, unsafe_allow_html=True)

    # =====================================================
    # FEEDERS
    # =====================================================

    st.subheader("📡 Feeder Line Status")

    fcols = st.columns(4)

    for i in range(1, 5):

        feeder = f"F{i}"

        with fcols[i - 1]:

            if (
                prediction.lower() != "normal"
                and feeder == faulty_f
            ):

                st.markdown(f"""
                <div class="fault-box">
                    🚨 Feeder 0{i}<br>
                    {prediction}
                </div>
                """, unsafe_allow_html=True)

            else:

                st.markdown(f"""
                <div class="normal-box">
                    ✅ Feeder 0{i}<br>
                    Normal
                </div>
                """, unsafe_allow_html=True)

    st.write("")

    # =====================================================
    # SCENARIOS
    # =====================================================

    scenarios = [
        "Normal",
        "Theft",
        "Power_Cut",
        "Ground_Fault",
        "Branch_Touching",
        "Lightning"
    ]

    scols = st.columns(6)

    for idx, sc in enumerate(scenarios):

        active = (
            sc.lower()
            in prediction.lower()
        )

        cls = (
            "scenario-box active-scenario"
            if active
            else "scenario-box"
        )

        scols[idx].markdown(f"""
        <div class="{cls}">
            {sc}
        </div>
        """, unsafe_allow_html=True)

# =========================================================
# RIGHT SIDE
# =========================================================

with right_main:

    st.subheader("🚨 Warning Panel")

    if prediction.lower() != "normal":

        warning_data = {

            "date":
            datetime.now().strftime("%Y-%m-%d"),

            "time":
            datetime.now().strftime("%H:%M:%S"),

            "fault":
            prediction,

            "feeder":
            faulty_f
        }

        if (
            len(st.session_state.warning_history) == 0
            or
            st.session_state.warning_history[0] != warning_data
        ):

            st.session_state.warning_history.insert(
                0,
                warning_data
            )

    if len(st.session_state.warning_history) > 0:

        for warn in st.session_state.warning_history[:10]:

            st.markdown(f"""
            <div class="warning-box">

                <div class="warning-title">
                    ⚠ WARNING DETECTED ⚠
                </div>

                <br>

                <div class="warning-text">
                    📅 {warn['date']}
                </div>

                <div class="warning-text">
                    🕒 {warn['time']}
                </div>

                <br>

                <div class="warning-text">
                    📡 {warn['feeder']}
                </div>

                <div class="warning-text">
                    🚨 {warn['fault']}
                </div>

            </div>
            """, unsafe_allow_html=True)

    else:

        st.success("✅ No Active Warnings")

# =========================================================
# AUTO REFRESH
# =========================================================

time.sleep(3)

st.session_state.row_index = (
    st.session_state.row_index + 1
) % len(df)

st.rerun()
