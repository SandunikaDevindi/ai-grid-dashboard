import streamlit as st
import pandas as pd
import time
import os
import joblib
from datetime import datetime

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="AI-Powered Live Grid Monitor",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------------- RESPONSIVE CSS ----------------

st.markdown("""
<style>

/* BACKGROUND */
.stApp{
    background-color:#0b1220;
    color:white;
}

/* MAIN PADDING */
.block-container{
    padding-top:1rem;
    padding-bottom:1rem;
    padding-left:2rem;
    padding-right:2rem;
}

/* TITLE */
h1{
    font-size: clamp(28px, 4vw, 52px) !important;
    color:white !important;
}

/* SUBHEADERS */
h2,h3{
    font-size: clamp(18px, 2vw, 32px) !important;
    color:white !important;
}

/* METRIC BOX */
[data-testid="metric-container"]{
    background:#111827;
    border-radius:16px;
    padding:18px;
    border:1px solid #1f2937;
}

/* METRIC LABEL */
[data-testid="stMetricLabel"]{
    font-size: clamp(12px, 1vw, 18px);
}

/* METRIC VALUE */
[data-testid="stMetricValue"]{
    font-size: clamp(22px, 2vw, 40px);
}

/* ALERT BOX */
.stAlert{
    border-radius:14px;
}

/* WARNING PANEL */
.warning-box{
    color:white;
    padding:20px;
    border-radius:20px;
    margin-bottom:15px;
    font-weight:bold;
    line-height:1.8;
    font-size: clamp(14px, 1vw, 20px);
    box-shadow:0px 0px 10px rgba(255,0,0,0.4);
}

/* SCENARIO BOX */
.scenario-box{
    padding:14px;
    border-radius:12px;
    text-align:center;
    font-weight:bold;
    color:white;
    font-size: clamp(12px, 1vw, 18px);
}

/* MOBILE */
@media (max-width:768px){

    .block-container{
        padding-left:1rem;
        padding-right:1rem;
    }

    h1{
        font-size:28px !important;
    }

    .warning-box{
        font-size:14px;
        padding:14px;
    }
}

/* LARGE SCREEN */
@media (min-width:1800px){

    .block-container{
        padding-left:5rem;
        padding-right:5rem;
    }

    .warning-box{
        font-size:22px;
    }
}

</style>
""", unsafe_allow_html=True)

# ---------------- FILE PATHS ----------------

DATASET_PATH = "AI_Grid_Anomaly_Dataset_for_dash_board.csv"

MODEL_PATH = "xgb_model.pkl"

ENCODER_PATH = "label_encoder.pkl"

HISTORY_FILE = "grid_history_log.xlsx"

STATE_FILE = "grid_state.csv"

# ---------------- CHECK FILES ----------------

required_files = [
    DATASET_PATH,
    MODEL_PATH,
    ENCODER_PATH
]

for file in required_files:

    if not os.path.exists(file):

        st.error(f"❌ File Not Found: {file}")
        st.stop()

# ---------------- LOAD DATA ----------------

@st.cache_data
def load_data():

    return pd.read_csv(DATASET_PATH)

df = load_data()

# ---------------- LOAD MODEL ----------------

model = joblib.load(MODEL_PATH)

label_encoder = joblib.load(ENCODER_PATH)

# ---------------- LOAD HISTORY ----------------

def load_history():

    if os.path.exists(HISTORY_FILE):

        try:

            return pd.read_excel(
                HISTORY_FILE
            ).to_dict("records")

        except:

            return []

    return []

# ---------------- SAVE HISTORY ----------------

def save_history():

    pd.DataFrame(
        st.session_state.history_log
    ).to_excel(
        HISTORY_FILE,
        index=False
    )

# ---------------- LOAD STATE ----------------

def load_state():

    if os.path.exists(STATE_FILE):

        try:

            state_df = pd.read_csv(STATE_FILE)

            return {

                "row_index":
                int(state_df.loc[0, "row_index"]),

                "next_update_time":
                float(state_df.loc[0, "next_update_time"])
            }

        except:

            return None

    return None

# ---------------- SAVE STATE ----------------

def save_state():

    pd.DataFrame([{

        "row_index":
        st.session_state.row_index,

        "next_update_time":
        st.session_state.next_update_time

    }]).to_csv(
        STATE_FILE,
        index=False
    )

# ---------------- SESSION STATES ----------------

saved_state = load_state()

if "history_log" not in st.session_state:

    st.session_state.history_log = load_history()

if "warning_history" not in st.session_state:

    st.session_state.warning_history = []

if "row_index" not in st.session_state:

    if saved_state:

        st.session_state.row_index = saved_state["row_index"]

    else:

        st.session_state.row_index = 0

if "next_update_time" not in st.session_state:

    if saved_state:

        st.session_state.next_update_time = (
            saved_state["next_update_time"]
        )

    else:

        st.session_state.next_update_time = (
            time.time() + (15 * 60)
        )

        save_state()

# ---------------- AUTO UPDATE ----------------

update_interval = 15 * 60

current_time = time.time()

if current_time >= st.session_state.next_update_time:

    st.session_state.row_index = (
        st.session_state.row_index + 1
    ) % len(df)

    st.session_state.next_update_time += (
        update_interval
    )

    save_state()

# ---------------- CURRENT ROW ----------------

row = df.iloc[
    st.session_state.row_index
]

# ---------------- ML PREDICTION ----------------

input_data = pd.DataFrame([{

    "Voltage":
    row["Voltage"],

    "Current":
    row["Current"],

    "Transformer_kW":
    row["Transformer_kW"]

}])

prediction_encoded = model.predict(
    input_data
)[0]

prediction = label_encoder.inverse_transform(
    [prediction_encoded]
)[0]

# ---------------- FEEDER ----------------

faulty_f = ""

if pd.notna(row["Fault_Feeder"]):

    faulty_f = str(
        row["Fault_Feeder"]
    ).strip()

# ---------------- HISTORY ----------------

latest_time = datetime.now().strftime("%H:%M")

exists = False

for item in st.session_state.history_log:

    if item["Logged_Time"] == latest_time:

        exists = True
        break

if not exists:

    history_entry = {

        "Logged_Date":
        datetime.now().strftime("%Y-%m-%d"),

        "Logged_Time":
        latest_time
    }

    for i in range(1, 5):

        feeder = f"F{i}"

        if (
            prediction.lower() != "normal"
            and
            feeder == faulty_f
        ):

            history_entry[feeder] = prediction

        else:

            history_entry[feeder] = "Normal"

    st.session_state.history_log.insert(
        0,
        history_entry
    )

    save_history()

# ---------------- LAYOUT ----------------

left_main, right_main = st.columns([2.8, 1.2])

# ================= LEFT =================

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

        st.markdown(
            f"## 🕒 {datetime.now().strftime('%H:%M:%S')}"
        )

        st.caption(
            f"📅 {datetime.now().strftime('%Y-%m-%d')}"
        )

    st.divider()

    # METRICS

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

    # FEEDERS

    st.write("---")

    st.subheader(
        "📡 Feeder Line Status"
    )

    feeder_cols = st.columns(4)

    for i in range(1, 5):

        feeder = f"F{i}"

        with feeder_cols[i - 1]:

            if (
                prediction.lower() != "normal"
                and
                feeder == faulty_f
            ):

                st.error(
                    f"🚨 Feeder 0{i}\n{prediction}"
                )

            else:

                st.success(
                    f"✅ Feeder 0{i}\nNormal"
                )

    # SCENARIOS

    st.write("---")

    scenarios = [

        "Normal",
        "Theft",
        "Power_Cut",
        "Ground_Fault",
        "Branch_Touching",
        "Lightning"
    ]

    s_cols = st.columns(6)

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

    # COUNTDOWN

    time_left = int(
        st.session_state.next_update_time
        - current_time
    )

    if time_left < 0:

        time_left = 0

    minutes = time_left // 60
    seconds = time_left % 60

    st.caption(
        f"🔄 Next Update in: "
        f"{minutes:02d}m {seconds:02d}s"
    )

    # HISTORY

    st.write("---")

    st.subheader(
        "📜 Event History Log"
    )

    if st.session_state.history_log:

        h_df = pd.DataFrame(
            st.session_state.history_log
        )

        st.dataframe(
            h_df,
            use_container_width=True,
            height=500
        )

# ================= RIGHT =================

with right_main:

    st.subheader("🚨 Warning Panel")

    if prediction.lower() != "normal":

        warning = {

            "date":
            datetime.now().strftime("%Y-%m-%d"),

            "time":
            datetime.now().strftime("%H:%M:%S"),

            "feeder":
            faulty_f,

            "fault":
            prediction
        }

        if (
            len(st.session_state.warning_history) == 0
            or
            st.session_state.warning_history[0]
            != warning
        ):

            st.session_state.warning_history.insert(
                0,
                warning
            )

    if len(st.session_state.warning_history) > 0:

        for warn in st.session_state.warning_history[:10]:

            st.markdown(
                f"""
                <div class="warning-box">

                ⚠ WARNING DETECTED

                <br>

                📅 {warn['date']}
                <br>
                🕒 {warn['time']}
                <br>

                📡 {warn['feeder']}
                <br>
                🚨 {warn['fault']}

                </div>
                """,
                unsafe_allow_html=True
            )

    else:

        st.success(
            "✅ No Active Warnings"
        )

# ---------------- DOWNLOAD ----------------

st.write("---")

history_df = pd.DataFrame(
    st.session_state.history_log
)

excel_file = "grid_history_export.xlsx"

history_df.to_excel(
    excel_file,
    index=False
)

with open(excel_file, "rb") as file:

    st.download_button(
        label="📥 Download History Excel",
        data=file,
        file_name="grid_history_export.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

# ---------------- AUTO REFRESH ----------------

time.sleep(1)

st.rerun()
