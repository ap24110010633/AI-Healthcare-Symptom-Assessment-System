import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)


# =========================================================
# 1. LOAD DATASET
# =========================================================

print("Loading dataset...")

df = pd.read_csv("dataset/dataset.csv")

print("Dataset loaded successfully.")
print("Original rows:", len(df))


# =========================================================
# 2. CLEAN DATA
# =========================================================

df["Disease"] = df["Disease"].astype(str).str.strip()

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

print("After removing exact duplicates:", len(df))


# =========================================================
# 3. CREATE SYMPTOM LIST
# =========================================================

all_symptoms = set()

for col in symptom_columns:

    for symptom in df[col]:

        if symptom.strip() != "":
            all_symptoms.add(symptom.strip())


all_symptoms = sorted(all_symptoms)

print("Unique symptoms:", len(all_symptoms))


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
# 5. ENCODE TARGET
# =========================================================

y = df["Disease"]

label_encoder = LabelEncoder()

y_encoded = label_encoder.fit_transform(y)

print("Number of diseases:", len(label_encoder.classes_))


# =========================================================
# 6. CREATE SYMPTOM PATTERN GROUPS
# =========================================================

# Rows having exactly the same symptoms are assigned
# to the same group.

groups = X.astype(str).agg("".join, axis=1)


print("Unique symptom patterns:", groups.nunique())


# =========================================================
# 7. GROUP TRAIN / TEST SPLIT
# =========================================================

gss = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)

train_idx, test_idx = next(
    gss.split(
        X,
        y_encoded,
        groups=groups
    )
)


X_train = X.iloc[train_idx]
X_test = X.iloc[test_idx]

y_train = y_encoded[train_idx]
y_test = y_encoded[test_idx]


print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# =========================================================
# 8. CREATE MODELS
# =========================================================

models = {

    "Decision Tree":
        DecisionTreeClassifier(
            random_state=42
        ),

    "Random Forest":
        RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            n_jobs=-1
        ),

    "Extra Trees":
        ExtraTreesClassifier(
            n_estimators=300,
            random_state=42,
            n_jobs=-1
        ),

    "KNN":
        KNeighborsClassifier(
            n_neighbors=5
        )
}


# =========================================================
# 9. TRAIN AND EVALUATE
# =========================================================

results = []

trained_models = {}


for name, model in models.items():

    print("\nTraining:", name)

    model.fit(
        X_train,
        y_train
    )

    trained_models[name] = model

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    results.append({

        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1

    })


# =========================================================
# 10. DISPLAY RESULTS
# =========================================================

results_df = pd.DataFrame(results)

print("\n")
print("=" * 70)
print("GROUP-BASED MODEL EVALUATION RESULTS")
print("=" * 70)

print(
    results_df.to_string(
        index=False
    )
)


# =========================================================
# 11. SAVE RESULTS
# =========================================================

os.makedirs(
    "reports",
    exist_ok=True
)

results_df.to_csv(
    "reports/model_evaluation_results_grouped.csv",
    index=False
)

print(
    "\nResults saved to:"
    " reports/model_evaluation_results_grouped.csv"
)


# =========================================================
# 12. MODEL COMPARISON GRAPH
# =========================================================

os.makedirs(
    "screenshots",
    exist_ok=True
)

results_plot = results_df.set_index(
    "Model"
)

results_plot[
    [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score"
    ]
].plot(
    kind="bar",
    figsize=(10, 6)
)

plt.title(
    "Machine Learning Model Comparison"
)

plt.ylabel(
    "Score"
)

plt.ylim(
    0,
    1.05
)

plt.xticks(
    rotation=0
)

plt.legend(
    loc="lower right"
)

plt.tight_layout()

plt.savefig(
    "screenshots/model_comparison_grouped.png",
    dpi=300
)

plt.close()


# =========================================================
# 13. FIND BEST MODEL
# =========================================================

best_model_name = results_df.loc[
    results_df["Accuracy"].idxmax(),
    "Model"
]

best_model = trained_models[
    best_model_name
]

best_accuracy = results_df.loc[
    results_df["Accuracy"].idxmax(),
    "Accuracy"
]


print("\n")
print("=" * 70)

print(
    "BEST MODEL:",
    best_model_name
)

print(
    "BEST ACCURACY:",
    best_accuracy
)

print("=" * 70)


# =========================================================
# 14. CONFUSION MATRIX
# =========================================================

y_pred_best = best_model.predict(
    X_test
)

cm = confusion_matrix(
    y_test,
    y_pred_best
)

fig, ax = plt.subplots(
    figsize=(14, 12)
)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm
)

display.plot(
    ax=ax,
    xticks_rotation="vertical",
    colorbar=False
)

plt.title(
    f"Confusion Matrix - {best_model_name}"
)

plt.tight_layout()

plt.savefig(
    "screenshots/confusion_matrix_grouped.png",
    dpi=300
)

plt.close()


print(
    "\nConfusion matrix saved to:"
    " screenshots/confusion_matrix_grouped.png"
)

print("\nEvaluation completed successfully.")