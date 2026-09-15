from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st


st.set_page_config(page_title="Student GPA Predictor")

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "student_performance_model.pkl"
FEATURES_PATH = BASE_DIR / "student_performance_features.pkl"


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists() or not FEATURES_PATH.exists():
        st.error("Model files are missing from the project.")
        st.stop()
    return joblib.load(MODEL_PATH), joblib.load(FEATURES_PATH)


model, features = load_model()

st.title("Student GPA Predictor")
st.write("Enter the student details and click Predict.")

with st.form("prediction_form"):
    age = st.number_input("Age", min_value=10, max_value=60, value=20, step=1)
    study_hours = st.number_input("Study Hours per Week", 0.0, 100.0, 20.0, 0.5)
    attendance = st.number_input("Attendance Rate (%)", 0.0, 100.0, 85.0, 1.0)
    assignment = st.number_input("Assignment Completion Rate (%)", 0.0, 100.0, 90.0, 1.0)
    previous_gpa = st.number_input("Previous GPA", 0.0, 4.0, 3.0, 0.01)
    test_average = st.number_input("Test Average", 0.0, 100.0, 75.0, 1.0)
    participation = st.number_input("Class Participation", 0.0, 10.0, 8.0, 0.5)
    homework = st.number_input("Homework Hours per Week", 0.0, 100.0, 10.0, 0.5)
    sleep = st.number_input("Sleep Hours per Day", 0.0, 24.0, 7.0, 0.5)
    screen_time = st.number_input("Screen Time Hours per Day", 0.0, 24.0, 4.0, 0.5)
    parental_support = st.selectbox("Parental Support", [0, 1, 2], index=2)
    motivation = st.selectbox("Motivation Level", [1, 2, 3, 4, 5], index=3)
    stress = st.selectbox("Stress Level", [1, 2, 3, 4, 5], index=1)
    tutoring = st.selectbox("Tutoring", [0, 1], index=0)
    extracurricular = st.selectbox("Extracurricular Activities", [0, 1], index=1)
    internet = st.selectbox("Internet Access", [0, 1], index=1)
    predict = st.form_submit_button("Predict GPA")


if predict:
    student_data = pd.DataFrame([{
        "Age": age,
        "StudyHoursPerWeek": study_hours,
        "AttendanceRate": attendance,
        "AssignmentCompletionRate": assignment,
        "PreviousGPA": previous_gpa,
        "TestAverage": test_average,
        "ClassParticipation": participation,
        "HomeworkHoursPerWeek": homework,
        "SleepHoursPerDay": sleep,
        "ScreenTimeHoursPerDay": screen_time,
        "ParentalSupport": parental_support,
        "MotivationLevel": motivation,
        "StressLevel": stress,
        "Tutoring": tutoring,
        "ExtracurricularActivities": extracurricular,
        "InternetAccess": internet,
    }])[features]

    predicted_gpa = float(np.clip(model.predict(student_data)[0], 0.0, 4.0))
    st.success(f"Predicted GPA: {predicted_gpa:.2f} / 4.00")
