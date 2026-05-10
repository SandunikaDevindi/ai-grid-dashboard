import streamlit as st
import pandas as pd
import time
import os
import joblib
from datetime import datetime
from zoneinfo import ZoneInfo

# ================= PAGE CONFIG =================

st.set_page_config(
    page_title="AI-Powered Live Grid Monitor",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ================= SRI LANKA TIME =================

sl_zone = ZoneInfo("Asia/Colombo")

sl_time = datetime.now(sl_zone)

# ================= MODEL ACCURACY =================

MODEL_ACCURACY = 99.87

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
}

</style>
""", unsafe_allow_html=True)

# ================= FILE PATHS =================

DATASET_PATH = "future_grid_test_dataset.csv"

MODEL_PATH = "xgb_model.pkl"

ENCODER_PATH = "label_encoder.pkl"

HISTORY_FILE = "grid_history_log.xlsx"

STATE_FILE = "grid_state.csv"

WARNING_FILE = "warnings.csv"

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

# ================= PREDICT ALL DATA =================

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

# ================= LOAD HISTORY =================

def load_history():

    if os.path.exists(HISTORY_FILE):

        try:

            return pd.read_excel(
                HISTORY_FILE
            ).to_dict("records")

        except:

            return []

    return []

# ================= SAVE HISTORY =================

def save_history():

    pd.DataFrame(
        st.session_state.history_log
    ).to_excel(
        HISTORY_FILE,
        index=False
    )

# ================= LOAD STATE =================

def load_state():

    if os.path.exists(STATE_FILE):

        try:

            state_df = pd.read_csv(STATE_FILE)

            return {

                "row_index":
                int(state_df.loc[0,"row_index"]),

                "next_update_time":
                float(state_df.loc[0,"next_update_time"])
            }

        except:

            return None

    return None

# ================= SAVE STATE =================

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

# ================= SESSION STATES =================

saved_state = load_state()

if "history_log" not in st.session_state:

    st.session_state.history_log = load_history()

if "last_warning_signature" not in st.session_state:

    st.session_state.last_warning_signature = None

# ================= LOAD WARNINGS =================

if os.path.exists(WARNING_FILE):

    try:

        warnings_df = pd.read_csv(WARNING_FILE)

        st.session_state.warning_history = (
            warnings_df.to_dict("records")
        )

    except:

        st.session_state.warning_history = []

else:

    st.session_state.warning_history = []

# ================= REMOVE WARNINGS AFTER 48 HOURS =================

filtered_warnings = []

current_dt = datetime.now(sl_zone)

for warn in st.session_state.warning_history:

    try:

        warn_dt = datetime.strptime(

            f"{warn['date']} {warn['time']}",
            "%Y-%m-%d %H:%M"

        )

        warn_dt = warn_dt.replace(
            tzinfo=sl_zone
        )

        diff_hours = (
            current_dt - warn_dt
        ).total_seconds() / 3600

        if diff_hours <= 48:

            filtered_warnings.append(warn)

    except:

        filtered_warnings.append(warn)

st.session_state.warning_history = filtered_warnings

pd.DataFrame(
    filtered_warnings
).to_csv(
    WARNING_FILE,
    index=False
)

# ================= ROW INDEX =================

if "row_index" not in st.session_state:

    if saved_state:

        st.session_state.row_index = saved_state["row_index"]

    else:

        st.session_state.row_index = 0

# ================= NEXT UPDATE =================

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

# ================= AUTO UPDATE =================

update_interval = 15 * 60

current_time = time.time()

new_update = False

if current_time >= st.session_state.next_update_time:

    st.session_state.row_index = (
        st.session_state.row_index + 1
    ) % len(df)

    st.session_state.next_update_time += (
        update_interval
    )

    save_state()

    new_update = True

# ================= CURRENT ROW =================

row = df.iloc[
    st.session_state.row_index
]

prediction = row["Prediction"]

# ================= FEEDER =================

faulty_f = "F1"

if "Fault_Feeder" in df.columns:

    if pd.notna(row["Fault_Feeder"]):

        faulty_f = str(
            row["Fault_Feeder"]
        ).strip().upper()

# ================= FIXED LOG TIME =================

fixed_time = datetime.fromtimestamp(

    st.session_state.next_update_time
    -
    update_interval,

    sl_zone

)

logged_date = fixed_time.strftime("%Y-%m-%d")

logged_time = fixed_time.strftime("%H:%M")

# ================= HISTORY =================

exists = False

for item in st.session_state.history_log:

    if (

        item["Logged_Date"] == logged_date
        and
        item["Logged_Time"] == logged_time

    ):

        exists = True
        break

if new_update and not exists:

    history_entry = {

        "Logged_Date":
        logged_date,

        "Logged_Time":
        logged_time
    }

    for i in range(1,5):

        feeder = f"F{i}"

        if (

            prediction.lower() != "normal"
            and
            feeder.upper() == faulty_f.upper()

        ):

            history_entry[feeder] = prediction

        else:

            history_entry[feeder] = "Normal"

    st.session_state.history_log.insert(
        0,
        history_entry
    )

    save_history()

# ================= WARNING STORAGE =================

if prediction.lower() != "normal":

    warning_signature = (

        f"{logged_date}_"
        f"{logged_time}_"
        f"{faulty_f}_"
        f"{prediction}"

    )

    warning = {

        "date":
        logged_date,

        "time":
        logged_time,

        "feeder":
        faulty_f,

        "fault":
        prediction
    }

    if (

        st.session_state.last_warning_signature
        !=
        warning_signature

    ):

        st.session_state.warning_history.insert(
            0,
            warning
        )

        warnings_df = pd.DataFrame(
            st.session_state.warning_history
        )

        warnings_df.to_csv(
            WARNING_FILE,
            index=False
        )

        st.session_state.last_warning_signature = (
            warning_signature
        )

# ================= LAYOUT =================

left_main, right_main = st.columns([2.7,1.3])

# ================= LEFT PANEL =================

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

    # ================= COUNTDOWN =================

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

    # ================= FEEDER STATUS HISTORY =================

    st.write("---")

    st.subheader(
        "📜 Feeder Status History"
    )

    if st.session_state.history_log:

        history_df = pd.DataFrame(
            st.session_state.history_log
        )

        st.dataframe(

            history_df[
                [
                    "Logged_Date",
                    "Logged_Time",
                    "F1",
                    "F2",
                    "F3",
                    "F4"
                ]
            ],

            use_container_width=True,
            height=350
        )

    # ================= ANOMALY EVENT HISTORY =================

    st.write("---")

    st.subheader(
        "🚨 Anomaly Event History"
    )

    anomaly_records = []

    for item in st.session_state.history_log:

        for feeder in ["F1", "F2", "F3", "F4"]:

            value = item[feeder]

            if str(value).lower() != "normal":

                anomaly_records.append({

                    "Date":
                    item["Logged_Date"],

                    "Time":
                    item["Logged_Time"],

                    "Feeder":
                    feeder,

                    "Anomaly":
                    value

                })

    if len(anomaly_records) > 0:

        anomaly_df = pd.DataFrame(
            anomaly_records
        )

        st.dataframe(

            anomaly_df,

            use_container_width=True,
            height=300

        )

    else:

        st.success(
            "✅ No anomaly events detected"
        )

# ================= RIGHT PANEL =================

with right_main:

    st.subheader(
        "🚨 Warning Panel"
    )

    if len(st.session_state.warning_history) > 0:

        for warn in st.session_state.warning_history[:20]:

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
display:flex;
justify-content:space-between;
font-size:15px;
margin-bottom:15px;
">

<div>
📅 {warn['date']}
</div>

<div>
🕒 {warn['time']}
</div>

</div>

<div style="
text-align:center;
font-size:22px;
font-weight:bold;
margin-bottom:10px;
">
📡 {warn['feeder']}
</div>

<div style="
text-align:center;
font-size:26px;
font-weight:900;
">
🚨 {warn['fault']}
</div>

</div>

""", unsafe_allow_html=True)

    else:

        st.success(
            "✅ No Active Warnings"
        )

# ================= DOWNLOAD =================

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

# ================= AUTO REFRESH =================

time.sleep(1)

st.rerun()
