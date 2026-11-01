import pandas as pd
import numpy as np
import joblib
import os

from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier


# =========================================================
# 1. LOAD DATASET
# =========================================================

print("Loading dataset...")

df = pd.read_csv("dataset/dataset.csv")

print("Original rows:", len(df))


# =========================================================
# 2. CLEAN DATA
# =========================================================

df["Disease"] = (
    df["Disease"]
    .astype(str)
    .str.strip()
)

symptom_columns = [
    col for col in df.columns
    if col.startswith("Symptom_")
]

for col in symptom_columns:

    df[col] = (
        df[col]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
    )


# Remove rows without symptoms

df = df[
    df[symptom_columns]
    .replace("", np.nan)
    .notna()
    .any(axis=1)
].copy()


# Remove exact duplicate rows

df = df.drop_duplicates().reset_index(drop=True)

print("Cleaned rows:", len(df))


# =========================================================
# 3. CREATE SYMPTOM VOCABULARY
# =========================================================

all_symptoms = set()

for col in symptom_columns:

    for symptom in df[col]:

        if symptom.strip() != "":
            all_symptoms.add(
                symptom.strip()
            )


all_symptoms = sorted(all_symptoms)

print(
    "Number of unique symptoms:",
    len(all_symptoms)
)


# =========================================================
# 4. MULTI-HOT ENCODING
# =========================================================

X = pd.DataFrame(
    0,
    index=df.index,
    columns=all_symptoms
)


for index, row in df.iterrows():

    for col in symptom_columns:

        symptom = row[col].strip()

        if symptom != "":
            X.loc[index, symptom] = 1


# =========================================================
# 5. ENCODE DISEASE LABELS
# =========================================================

y = df["Disease"]

label_encoder = LabelEncoder()

y_encoded = label_encoder.fit_transform(y)

print(
    "Number of diseases:",
    len(label_encoder.classes_)
)


# =========================================================
# 6. TRAIN FINAL RANDOM FOREST
# =========================================================

print("\nTraining final Random Forest model...")

model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    n_jobs=-1
)

model.fit(
    X,
    y_encoded
)

print("Final model trained successfully.")


# =========================================================
# 7. CREATE MODEL DIRECTORY
# =========================================================

os.makedirs(
    "model",
    exist_ok=True
)


# =========================================================
# 8. SAVE MODEL
# =========================================================

joblib.dump(
    model,
    "model/disease_model.pkl"
)

joblib.dump(
    label_encoder,
    "model/label_encoder.pkl"
)

joblib.dump(
    all_symptoms,
    "model/symptom_features.pkl"
)


# =========================================================
# 9. DISPLAY INFORMATION
# =========================================================

print("\n")
print("=" * 60)
print("FINAL MODEL CREATED SUCCESSFULLY")
print("=" * 60)

print("Model: Random Forest")
print("Trees: 300")
print("Training samples:", len(X))
print("Number of symptoms:", len(all_symptoms))
print("Number of diseases:", len(label_encoder.classes_))

print("\nSaved files:")

print("model/disease_model.pkl")
print("model/label_encoder.pkl")
print("model/symptom_features.pkl")

print("=" * 60)