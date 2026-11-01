import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re

from disease_info import disease_info


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="AI Healthcare Symptom Assessment",
    page_icon="🏥",
    layout="centered"
)


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model_files():

    model = joblib.load(
        "model/disease_model.pkl"
    )

    label_encoder = joblib.load(
        "model/label_encoder.pkl"
    )

    symptom_features = joblib.load(
        "model/symptom_features.pkl"
    )

    return model, label_encoder, symptom_features


model, label_encoder, symptom_features = load_model_files()


# =========================================================
# TEXT NORMALIZATION
# =========================================================

def normalize_text(text):

    text = text.lower()

    text = text.replace("_", " ")

    text = re.sub(
        r"[^a-zA-Z0-9\s]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# =========================================================
# SYMPTOM DETECTION
# =========================================================

def extract_symptoms(user_text):

    normalized_input = normalize_text(
        user_text
    )

    detected_symptoms = []

    for symptom in symptom_features:

        normalized_symptom = normalize_text(
            symptom
        )

        if normalized_symptom in normalized_input:

            detected_symptoms.append(
                symptom
            )

    return list(
        dict.fromkeys(
            detected_symptoms
        )
    )


# =========================================================
# CREATE INPUT VECTOR
# =========================================================

def create_input_vector(symptoms):

    input_data = pd.DataFrame(
        0,
        index=[0],
        columns=symptom_features
    )

    for symptom in symptoms:

        if symptom in input_data.columns:

            input_data.loc[
                0,
                symptom
            ] = 1

    return input_data


# =========================================================
# PREDICTION
# =========================================================

def predict_conditions(symptoms):

    input_data = create_input_vector(
        symptoms
    )

    probabilities = model.predict_proba(
        input_data
    )[0]

    top_indices = np.argsort(
        probabilities
    )[::-1][:3]

    predictions = []

    for index in top_indices:

        disease = label_encoder.inverse_transform(
            [index]
        )[0]

        score = float(
            probabilities[index]
        )

        predictions.append(
            (
                disease,
                score
            )
        )

    return predictions


# =========================================================
# HEADER
# =========================================================

st.title(
    "🏥 AI Healthcare Symptom Assessment System"
)

st.write(
    "An educational machine-learning system "
    "that analyzes entered symptoms and provides "
    "possible conditions based on a trained dataset."
)


# =========================================================
# DISCLAIMER
# =========================================================

st.warning(
    """
    ⚠️ IMPORTANT MEDICAL DISCLAIMER

    This application is an academic AI/ML project.
    It does NOT provide medical diagnosis, treatment,
    or professional medical advice.

    Predictions may be incorrect. Always consult a
    qualified healthcare professional for medical concerns.
    """
)


# =========================================================
# PROJECT INFORMATION
# =========================================================

with st.expander("ℹ️ About This Project"):

    st.write(
        """
        This project uses machine learning to demonstrate
        how symptoms can be represented as numerical features
        and used by a classification model to predict possible
        conditions from a predefined dataset.
        """
    )

    st.write(
        "**Machine Learning Model:** Random Forest"
    )

    st.write(
        "**Number of Diseases:** 41"
    )

    st.write(
        "**Number of Symptoms:** 131"
    )


# =========================================================
# SYMPTOM INPUT
# =========================================================

st.subheader(
    "🩺 Enter Your Symptoms"
)

user_input = st.text_area(

    "Describe your symptoms:",

    placeholder=(
        "Example: I have fever, headache, "
        "cough and fatigue"
    ),

    height=130
)


# =========================================================
# ASSESS BUTTON
# =========================================================

assess = st.button(
    "🔍 Assess Symptoms",
    use_container_width=True
)


# =========================================================
# PROCESS INPUT
# =========================================================

if assess:

    if not user_input.strip():

        st.warning(
            "Please enter at least one symptom."
        )

        st.stop()


    # -----------------------------------------------------
    # Detect symptoms
    # -----------------------------------------------------

    detected_symptoms = extract_symptoms(
        user_input
    )


    if len(detected_symptoms) == 0:

        st.error(
            "No recognized symptoms were found."
        )

        st.info(
            "Try using symptoms such as fever, "
            "headache, cough, fatigue, nausea, "
            "vomiting, itching, skin rash, etc."
        )

        st.stop()


    # -----------------------------------------------------
    # Display detected symptoms
    # -----------------------------------------------------

    st.success(
        "Symptoms detected successfully."
    )

    st.subheader(
        "🔎 Detected Symptoms"
    )

    for symptom in detected_symptoms:

        st.write(
            "•",
            normalize_text(symptom)
        )


    # -----------------------------------------------------
    # Minimum symptom warning
    # -----------------------------------------------------

    if len(detected_symptoms) < 2:

        st.warning(
            """
            Only one symptom was recognized.
            The model may not have enough information
            to provide a meaningful assessment.
            """
        )


    # -----------------------------------------------------
    # Predict
    # -----------------------------------------------------

    predictions = predict_conditions(
        detected_symptoms
    )

    top_disease, top_score = predictions[0]


    # -----------------------------------------------------
    # Main prediction
    # -----------------------------------------------------

    st.subheader(
        "🤖 Possible Condition"
    )

    st.success(
        top_disease
    )


    # -----------------------------------------------------
    # Prediction score
    # -----------------------------------------------------

    st.subheader(
        "📊 Model Prediction Score"
    )

    st.progress(
        min(top_score, 1.0)
    )

    st.write(
        f"{top_score * 100:.2f}%"
    )


    # -----------------------------------------------------
    # Confidence warning
    # -----------------------------------------------------

    if top_score < 0.40:

        st.error(
            """
            ⚠️ Low model prediction score.

            The available symptoms may not provide
            enough information for a reliable
            machine-learning assessment.
            """
        )

    elif top_score < 0.70:

        st.warning(
            """
            ⚠️ Moderate model prediction score.

            Consider providing additional relevant
            symptoms. This result should not be
            interpreted as a medical diagnosis.
            """
        )

    else:

        st.info(
            """
            The model found a stronger pattern in
            the available dataset. This is still
            not a medical diagnosis.
            """
        )


    # =====================================================
    # TOP 3 PREDICTIONS
    # =====================================================

    st.subheader(
        "📋 Top 3 Model Predictions"
    )

    for rank, (disease, score) in enumerate(
        predictions,
        start=1
    ):

        st.write(
            f"**{rank}. {disease}** — "
            f"{score * 100:.2f}%"
        )


    # =====================================================
    # EDUCATIONAL INFORMATION
    # =====================================================

    st.subheader(
        "📚 General Educational Information"
    )

    information = disease_info.get(

        top_disease,

        {
            "description":
                "This condition is included "
                "in the machine-learning dataset. "
                "The prediction should not be "
                "treated as a medical diagnosis.",

            "advice":
                "Consult a qualified healthcare "
                "professional for proper evaluation."
        }
    )


    st.write(
        information["description"]
    )


    # =====================================================
    # GENERAL ADVICE
    # =====================================================

    st.subheader(
        "⚕️ General Advice"
    )

    st.write(
        information["advice"]
    )


    # =====================================================
    # FINAL DISCLAIMER
    # =====================================================

    st.error(
        """
        IMPORTANT:

        This system is an educational AI/ML project.
        Model predictions may be incorrect and do not
        constitute medical diagnosis or treatment.

        If symptoms are severe, sudden, worsening,
        or concerning, seek appropriate medical care
        immediately.
        """
    )


# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.caption(
    "AI Healthcare Symptom Assessment System | "
    "Academic Minor Project | Machine Learning"
)