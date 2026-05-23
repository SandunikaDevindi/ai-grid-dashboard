import streamlit as st
import pandas as pd
import time
import os
import joblib
import pygame

from datetime import datetime
from zoneinfo import ZoneInfo

# ================= PAGE CONFIG =================

st.set_page_config(

    page_title="AI-Powered Live Grid Monitor",

    layout="wide",

    initial_sidebar_state="collapsed"
)

# ================= SOUND INIT =================

pygame.mixer.init()

WARNING_SOUND = "warning_alarm.mp3"

# ================= SRI LANKA TIME =================

sl_zone = ZoneInfo("Asia/Colombo")

# ================= MODEL ACCURACY =================

MODEL_ACCURACY = 99.91

# ================= CSS =================

st.markdown("""

<style>

.stApp{

    background-color:#081120;

    color:white;
}

.block-container{

    padding-top:1rem;

    padding-bottom:1rem;

    padding-left:1.5rem;

    padding-right:1.5rem;

    max-width:100%;
}

h1{

    font-size:clamp(28px,4vw,52px)!important;

    color:white!important;

    font-weight:800!important;
}

h2,h3{

    color:white!important;
}

[data-testid="metric-container"]{

    background:#111827;

    border-radius:16px;

    padding:18px;

    border:1px solid #1f2937;
}

[data-testid="stMetricValue"]{

    font-size:clamp(20px,2vw,42px);
}

.scenario-box{

    padding:14px;

    border-radius:12px;

    text-align:center;

    font-weight:bold;

    color:white;

    font-size:15px;
}

.warning-card{

    background:#ff0000;

    color:white;

    border-radius:20px;

    padding:20px;

    margin-bottom:18px;

    box-shadow:0px 0px 15px rgba(255,0,0,0.4);

    animation: blink 0.5s infinite;
}

@keyframes blink {

    0% {opacity:1;}

    50% {opacity:0.4;}

    100% {opacity:1;}
}

</style>

""", unsafe_allow_html=True)

# ================= FILE PATHS =================

DATASET_PATH = "future_grid_test_dataset.csv"

MODEL_PATH = "rf_model.pkl"

ENCODER_PATH = "label_encoder.pkl"

# ================= CHECK FILES =================

required_files = [

    DATASET_PATH,

    MODEL_PATH,

    ENCODER_PATH
]

for file in required_files:

    if not os.path.exists(file):

        st.error(f"❌ File Not Found: {file}")

        st.stop()

# ================= LOAD DATA =================

@st.cache_data
def load_data():

    return pd.read_csv(DATASET_PATH)

df = load_data()

# ================= LOAD MODEL =================

model = joblib.load(MODEL_PATH)

label_encoder = joblib.load(ENCODER_PATH)

# ================= PREDICT =================

predict_df = df[

    [
        "Voltage",
        "Current",
        "Transformer_kW"
    ]
]

pred_encoded = model.predict(

    predict_df
)

df["Prediction"] = label_encoder.inverse_transform(

    pred_encoded
)

# ================= RANDOM ANOMALY DISPLAY =================

anomaly_df = df[

    df["Prediction"].str.lower() != "normal"
]

row = anomaly_df.sample(1).iloc[0]

prediction = row["Prediction"]

# ================= FEEDER =================

faulty_f = "F1"

if "Fault_Feeder" in df.columns:

    if pd.notna(row["Fault_Feeder"]):

        faulty_f = str(

            row["Fault_Feeder"]

        ).strip().upper()

# ================= SESSION STATES =================

if "alarm_active" not in st.session_state:

    st.session_state.alarm_active = False

if "alarm_start_time" not in st.session_state:

    st.session_state.alarm_start_time = None

if "alarm_muted" not in st.session_state:

    st.session_state.alarm_muted = False

# ================= ALARM SYSTEM =================

current_time = time.time()

if prediction.lower() != "normal":

    if not st.session_state.alarm_active:

        st.session_state.alarm_active = True

        st.session_state.alarm_start_time = current_time

        st.session_state.alarm_muted = False

        if not pygame.mixer.music.get_busy():

            pygame.mixer.music.load(

                WARNING_SOUND
            )

            pygame.mixer.music.play(-1)

    elapsed = (

        current_time
        -
        st.session_state.alarm_start_time
    )

    if elapsed >= 30:

        pygame.mixer.music.stop()

        st.session_state.alarm_active = False

else:

    pygame.mixer.music.stop()

    st.session_state.alarm_active = False

# ================= HEADER =================

left_main, right_main = st.columns([2.7,1.3])

with left_main:

    c1, c2 = st.columns([3,1])

    with c1:

        st.title(

            "⚡ AI-Powered Live Grid Monitor"
        )

        st.subheader(

            f"AI Prediction: {prediction}"
        )

    with c2:

        st.metric(

            "Model Accuracy",

            f"{MODEL_ACCURACY}%"
        )

        sl_time = datetime.now(sl_zone)

        st.markdown(

            f"## 🕒 {sl_time.strftime('%H:%M:%S')}"
        )

        st.caption(

            f"📅 {sl_time.strftime('%Y-%m-%d')}"
        )

    st.divider()

    # ================= METRICS =================

    m1, m2, m3, m4 = st.columns(4)

    m1.metric(

        "Voltage",

        f"{row['Voltage']:.2f} V"
    )

    m2.metric(

        "Current",

        f"{row['Current']:.2f} A"
    )

    m3.metric(

        "Power",

        f"{row['Transformer_kW']:.2f} kW"
    )

    m4.metric(

        "PF",

        "0.88"
    )

    # ================= FEEDER STATUS =================

    st.write("---")

    st.subheader(

        "📡 Feeder Line Status"
    )

    feeder_cols = st.columns(4)

    for i in range(1,5):

        feeder = f"F{i}"

        with feeder_cols[i-1]:

            if (

                prediction.lower() != "normal"
                and
                feeder.upper() == faulty_f.upper()

            ):

                st.error(

                    f"🚨 Feeder 0{i}\n{prediction}"
                )

            else:

                st.success(

                    f"✅ Feeder 0{i}\nNormal"
                )

    # ================= SCENARIOS =================

    st.write("---")

    scenarios = [

        "Normal",

        "Theft",

        "Power_Cut",

        "Ground_Fault",

        "Lightning"
    ]

    s_cols = st.columns(5)

    for idx, s in enumerate(scenarios):

        active = (

            s.lower()
            in
            prediction.lower()
        )

        color = (

            "#ff4b4b"
            if active
            else "#262730"
        )

        s_cols[idx].markdown(

            f"""

            <div class="scenario-box"
            style="background:{color};">

            {s}

            </div>

            """,

            unsafe_allow_html=True
        )

    # ================= ALARM BUTTON =================

    st.write("---")

    if st.session_state.alarm_active:

        if st.button("🔕 Mute Alarm"):

            pygame.mixer.music.stop()

            st.session_state.alarm_active = False

            st.session_state.alarm_muted = True

            st.success("✅ Alarm Muted")

# ================= RIGHT PANEL =================

with right_main:

    st.subheader(

        "🚨 Warning Panel"
    )

    if prediction.lower() != "normal":

        st.markdown(f"""

<div class="warning-card">

<div style="
text-align:center;
font-size:24px;
font-weight:bold;
margin-bottom:20px;
">

⚠ WARNING DETECTED ⚠

</div>

<div style="
text-align:center;
font-size:22px;
font-weight:bold;
margin-bottom:10px;
">

📡 {faulty_f}

</div>

<div style="
text-align:center;
font-size:26px;
font-weight:900;
">

🚨 {prediction}

</div>

</div>

""", unsafe_allow_html=True)

    else:

        st.success(

            "✅ No Active Warnings"
        )

# ================= AUTO REFRESH =================

time.sleep(5)

st.rerun()
