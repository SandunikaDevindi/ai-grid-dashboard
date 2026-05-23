import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ================= LOAD DATA =================

file_path = r"D:\Final research\Research_Dataset_50_Events_30_Users_OpenDSS_Augmented.csv"

df = pd.read_csv(file_path)

df['Scenario'] = df['Scenario'].str.strip()

# ================= REMOVE BRANCH TOUCHING =================

df = df[
    df['Scenario'] != 'Branch_Touching'
]

# ================= FEATURES =================

features = [

    'Voltage',
    'Current',
    'Transformer_kW'
]

X = df[features]

# ================= LABEL ENCODING =================

le = LabelEncoder()

y = le.fit_transform(
    df['Scenario']
)

# ================= TRAIN TEST SPLIT =================

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.2,

    stratify=y,

    random_state=42
)

# ================= TRAIN RANDOM FOREST =================

rf_model = RandomForestClassifier(

    n_estimators=100,

    class_weight='balanced',

    random_state=42
)

rf_model.fit(
    X_train,
    y_train
)

print("✅ Random Forest model trained successfully.")

# ================= SAVE MODEL =================

joblib.dump(
    rf_model,
    "rf_model.pkl"
)

print("✅ rf_model.pkl saved successfully.")

# ================= SAVE LABEL ENCODER =================

joblib.dump(
    le,
    "label_encoder.pkl"
)

print("✅ label_encoder.pkl saved successfully.")
