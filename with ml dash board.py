import streamlit as st
import pandas as pd
import time
import os
import joblib
from datetime import datetime

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="AI-Powered Live Grid Monitor",
    layout="wide"
)

# ---------------- FILE PATHS ----------------

DATASET_PATH = r"D:\Final research\AI_Grid_Anomaly_Dataset_for_dash_board.csv"

MODEL_PATH = r"D:\Final research\xgb_model.pkl"

ENCODER_PATH = r"D:\Final research\label_encoder.pkl"

HISTORY_FILE = r"D:\Final research\grid_history_log.xlsx"

STATE_FILE = r"D:\Final research\grid_state.csv"

# ---------------- LOAD DATA ----------------

@st.cache_data
def load_data():

    return pd.read_csv(DATASET_PATH)

df = load_data()

# ---------------- LOAD ML MODEL ----------------

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

    history_df = pd.DataFrame(
        st.session_state.history_log
    )

    history_df.to_excel(
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
                int(
                    state_df.loc[
                        0,
                        "row_index"
                    ]
                ),

                "next_update_time":
                float(
                    state_df.loc[
                        0,
                        "next_update_time"
                    ]
                )
            }

        except:

            return None

    return None

# ---------------- SAVE STATE ----------------

def save_state():

    state_df = pd.DataFrame([{

        "row_index":
        st.session_state.row_index,

        "next_update_time":
        st.session_state.next_update_time

    }])

    state_df.to_csv(
        STATE_FILE,
        index=False
    )

# ---------------- LOAD SAVED STATE ----------------

saved_state = load_state()

# ---------------- SESSION STATES ----------------

if "history_log" not in st.session_state:

    st.session_state.history_log = (
        load_history()
    )

if "warning_history" not in st.session_state:

    st.session_state.warning_history = []

# ---------------- ROW INDEX ----------------

if "row_index" not in st.session_state:

    if saved_state:

        st.session_state.row_index = (
            saved_state["row_index"]
        )

    else:

        st.session_state.row_index = 0

# ---------------- NEXT UPDATE ----------------

if "next_update_time" not in st.session_state:

    if (
        saved_state
        and
        "next_update_time" in saved_state
    ):

        st.session_state.next_update_time = (
            saved_state["next_update_time"]
        )

    else:

        st.session_state.next_update_time = (
            time.time() + (15 * 60)
        )

        save_state()

# ---------------- SETTINGS ----------------

update_interval = 15 * 60

# ---------------- UPDATE LOGIC ----------------

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

# ---------------- AI PREDICTION ----------------

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

# ---------------- HISTORY UPDATE ----------------

latest_time = datetime.now().strftime("%H:%M")

history_exists = False

for item in st.session_state.history_log:

    if item["Logged_Time"] == latest_time:

        history_exists = True

        break

if not history_exists:

    history_entry = {}

    history_entry["Logged_Date"] = (
        datetime.now().strftime("%Y-%m-%d")
    )

    history_entry["Logged_Time"] = (
        latest_time
    )

    for i in range(1, 5):

        f_id = f"F{i}"

        if (
            prediction.lower() != "normal"
            and
            f_id == faulty_f
        ):

            history_entry[f"F{i}"] = prediction

        else:

            history_entry[f"F{i}"] = "Normal"

    st.session_state.history_log.insert(
        0,
        history_entry
    )

    save_history()

# ---------------- MAIN LAYOUT ----------------

left_main, right_main = st.columns([2.8, 1.2])

# ================= LEFT SIDE =================

with left_main:

    # HEADER

    c1, c2 = st.columns([3, 1])

    with c1:

        st.title(
            "⚡ AI-Powered Live Grid Monitor"
        )

        st.subheader(
            f"AI Prediction: {prediction}"
        )

    with c2:

        st.markdown(
            f"## 🕒 "
            f"{datetime.now().strftime('%H:%M:%S')}"
        )

        st.caption(
            f"📅 "
            f"{datetime.now().strftime('%Y-%m-%d')}"
        )

    st.divider()

    # ---------------- METRICS ----------------

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

    # ---------------- FEEDER STATUS ----------------

    st.write("---")

    st.subheader(
        "📡 Feeder Line Status (Current)"
    )

    f_cols = st.columns(4)

    for i in range(1, 5):

        f_id = f"F{i}"

        with f_cols[i - 1]:

            if (
                prediction.lower() != "normal"
                and
                f_id == faulty_f
            ):

                st.error(
                    f"🚨 Feeder 0{i}\n"
                    f"{prediction}"
                )

            else:

                st.success(
                    f"✅ Feeder 0{i}\nNormal"
                )

    # ---------------- SCENARIO PANEL ----------------

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
            str(prediction).lower()
        )

        color = (
            "#ff4b4b"
            if active
            else "#262730"
        )

        s_cols[idx].markdown(
            f"""
            <div style="
                background-color:{color};
                color:white;
                padding:12px;
                border-radius:8px;
                text-align:center;
                font-weight:bold;
                font-size:16px;
            ">
                {s}
            </div>
            """,
            unsafe_allow_html=True
        )

    # ---------------- COUNTDOWN ----------------

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

    # ---------------- HISTORY ----------------

    st.write("---")

    st.subheader(
        "📜 Event History Log"
    )

    if st.session_state.history_log:

        h_df = pd.DataFrame(
            st.session_state.history_log
        )

        st.dataframe(
            h_df[
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
            height=500
        )

# ================= RIGHT SIDE =================

with right_main:

    st.subheader("🚨 Warning Panel")

    fault_detected = False

    if (
        pd.notna(prediction)
        and
        str(prediction).lower() != "normal"
    ):

        fault_detected = True

    # ---------------- ADD WARNING ----------------

    if fault_detected:

        warning_data = {

            "time":
            datetime.now().strftime("%H:%M:%S"),

            "date":
            datetime.now().strftime("%Y-%m-%d"),

            "feeder":
            faulty_f,

            "fault":
            prediction
        }

        if (
            len(st.session_state.warning_history) == 0
            or
            st.session_state.warning_history[0]
            != warning_data
        ):

            st.session_state.warning_history.insert(
                0,
                warning_data
            )

    # ---------------- SHOW WARNINGS ----------------

    if len(st.session_state.warning_history) > 0:

        for warn in st.session_state.warning_history[:10]:

            st.markdown(
                f"""
                <div style="
                    background-color:#ff0000;
                    color:white;
                    padding:18px;
                    border-radius:12px;
                    margin-bottom:12px;
                    font-weight:bold;
                    box-shadow:0px 0px 8px rgba(255,0,0,0.5);
                ">

                    <div style="
                        display:flex;
                        justify-content:space-between;
                        font-size:13px;
                        margin-bottom:10px;
                    ">

                        <div>
                            📅 {warn['date']}
                        </div>

                        <div>
                            🕒 {warn['time']}
                        </div>

                    </div>

                    <div style="
                        font-size:22px;
                        font-weight:900;
                        margin-top:8px;
                        margin-bottom:12px;
                        text-align:center;
                    ">

                        ⚠ WARNING DETECTED ⚠

                    </div>

                    <div style="
                        font-size:24px;
                        font-weight:bold;
                        text-align:center;
                    ">

                        📡 {warn['feeder']} - 🚨 {warn['fault']}

                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )

    else:

        st.success("✅ No Active Warnings")

# ---------------- DOWNLOAD EXCEL ----------------

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