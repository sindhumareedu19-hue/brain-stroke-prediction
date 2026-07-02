import streamlit as st
import joblib
import numpy as np

# -------------------------------
# Load Model and Scaler
# -------------------------------
model = joblib.load("random_forest.pkl")
scaler = joblib.load("scaler.pkl")

# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(
    page_title="Stroke Prediction",
    page_icon="🩺",
    layout="wide"
)

# -------------------------------
# Title
# -------------------------------
st.title("🩺 Stroke Prediction System")
st.write("Predict whether a patient is at risk of stroke using Machine Learning.")

st.markdown("---")

# -------------------------------
# Input Layout
# -------------------------------
col1, col2 = st.columns(2)

with col1:

    gender = st.selectbox(
        "Gender",
        ["Male", "Female", "Other"]
    )

    age = st.number_input(
        "Age",
        min_value=1,
        max_value=100,
        value=30
    )

    hypertension = st.selectbox(
        "Hypertension",
        [0,1]
    )

    heart_disease = st.selectbox(
        "Heart Disease",
        [0,1]
    )

    ever_married = st.selectbox(
        "Ever Married",
        ["No","Yes"]
    )

with col2:

    work_type = st.selectbox(
        "Work Type",
        [
            "Govt_job",
            "Never_worked",
            "Private",
            "Self-employed",
            "children"
        ]
    )

    residence = st.selectbox(
        "Residence Type",
        ["Rural","Urban"]
    )

    glucose = st.number_input(
        "Average Glucose Level",
        min_value=40.0,
        max_value=300.0,
        value=90.0
    )

    bmi = st.number_input(
        "BMI",
        min_value=10.0,
        max_value=70.0,
        value=25.0
    )

    smoking = st.selectbox(
        "Smoking Status",
        [
            "Unknown",
            "formerly smoked",
            "never smoked",
            "smokes"
        ]
    )

# -------------------------------
# Encoding Dictionaries
# -------------------------------

gender_map = {
    "Female":0,
    "Male":1,
    "Other":2
}

married_map = {
    "No":0,
    "Yes":1
}

work_map = {
    "Govt_job":0,
    "Never_worked":1,
    "Private":2,
    "Self-employed":3,
    "children":4
}

residence_map = {
    "Rural":0,
    "Urban":1
}

smoking_map = {
    "Unknown":0,
    "formerly smoked":1,
    "never smoked":2,
    "smokes":3
}

# -------------------------------
# Prediction Button
# -------------------------------

if st.button("Predict Stroke Risk"):

    input_data = np.array([[
        gender_map[gender],
        age,
        hypertension,
        heart_disease,
        married_map[ever_married],
        work_map[work_type],
        residence_map[residence],
        glucose,
        bmi,
        smoking_map[smoking]
    ]])

    scaled_data = scaler.transform(input_data)

    prediction = model.predict(scaled_data)

    probability = model.predict_proba(scaled_data)

    st.markdown("---")

    st.subheader("Prediction Result")

    if prediction[0] == 1:
        st.error("⚠️ High Risk of Stroke")
    else:
        st.success("✅ Low Risk of Stroke")

    st.subheader("Prediction Probability")

    st.write(f"Low Risk : {probability[0][0]*100:.2f}%")
    st.write(f"High Risk: {probability[0][1]*100:.2f}%")

    st.progress(float(probability[0][1]))

    st.markdown("---")

    st.subheader("Patient Information")

    st.write({
        "Gender": gender,
        "Age": age,
        "Hypertension": hypertension,
        "Heart Disease": heart_disease,
        "Ever Married": ever_married,
        "Work Type": work_type,
        "Residence": residence,
        "Average Glucose": glucose,
        "BMI": bmi,
        "Smoking": smoking
    })

# -------------------------------
# Sidebar
# -------------------------------

st.sidebar.title("Stroke Prediction System")

st.sidebar.info(
"""
Machine Learning Models Used

• Logistic Regression

• Random Forest

• XGBoost

• SVM

• KNN

• MLP

Dataset:
Healthcare Stroke Prediction Dataset
"""
)