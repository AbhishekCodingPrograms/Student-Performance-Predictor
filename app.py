import joblib
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "student_performance_model.pkl"
FEATURES_PATH = BASE_DIR / "student_performance_features.pkl"


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    if not MODEL_PATH.exists():
        st.error(
            "student_performance_model.pkl was not found."
        )
        st.stop()

    if not FEATURES_PATH.exists():
        st.error(
            "student_performance_features.pkl was not found."
        )
        st.stop()

    model = joblib.load(MODEL_PATH)

    features = joblib.load(FEATURES_PATH)

    return model, features


model, features = load_model()


# ============================================================
# HEADER
# ============================================================

st.title("Student Performance Predictor")
st.caption(
    "Estimate student GPA using academic performance, attendance, "
    "study habits and learning conditions."
)


# ============================================================
# PROJECT METRICS
# ============================================================

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric("Test R²", "0.721")
    st.caption("Held-out regression score")

with c2:
    st.metric("Test MAE", "0.176")
    st.caption("Average GPA error")

with c3:
    st.metric("Test RMSE", "0.221")
    st.caption("Larger errors weighted more")

with c4:
    st.metric("±0.30 GPA", "82.33%")
    st.caption("Predictions within tolerance")


st.write("")


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("Student Profile")

    st.caption(
        "Enter the student's academic and learning information."
    )

    age = st.number_input(
        "Age",
        min_value=10,
        max_value=60,
        value=20,
        step=1
    )

    study_hours = st.number_input(
        "Study Hours per Week",
        min_value=0.0,
        max_value=100.0,
        value=20.0,
        step=0.5
    )

    attendance = st.number_input(
        "Attendance Rate (%)",
        min_value=0.0,
        max_value=100.0,
        value=85.0,
        step=1.0
    )

    assignment = st.number_input(
        "Assignment Completion Rate (%)",
        min_value=0.0,
        max_value=100.0,
        value=90.0,
        step=1.0
    )

    previous_gpa = st.number_input(
        "Previous GPA",
        min_value=0.0,
        max_value=4.0,
        value=3.0,
        step=0.01
    )

    test_average = st.number_input(
        "Test Average",
        min_value=0.0,
        max_value=100.0,
        value=75.0,
        step=1.0
    )

    participation = st.number_input(
        "Class Participation",
        min_value=0.0,
        max_value=10.0,
        value=8.0,
        step=0.5
    )

    homework = st.number_input(
        "Homework Hours per Week",
        min_value=0.0,
        max_value=100.0,
        value=10.0,
        step=0.5
    )

    sleep = st.number_input(
        "Sleep Hours per Day",
        min_value=0.0,
        max_value=24.0,
        value=7.0,
        step=0.5
    )

    screen_time = st.number_input(
        "Screen Time Hours per Day",
        min_value=0.0,
        max_value=24.0,
        value=4.0,
        step=0.5
    )

    parental_support = st.selectbox(
        "Parental Support",
        [0, 1, 2],
        index=2
    )

    motivation = st.selectbox(
        "Motivation Level",
        [1, 2, 3, 4, 5],
        index=3
    )

    stress = st.selectbox(
        "Stress Level",
        [1, 2, 3, 4, 5],
        index=1
    )

    tutoring = st.selectbox(
        "Tutoring",
        [0, 1],
        index=0
    )

    extracurricular = st.selectbox(
        "Extracurricular Activities",
        [0, 1],
        index=1
    )

    internet = st.selectbox(
        "Internet Access",
        [0, 1],
        index=1
    )

    predict_button = st.button(
        "Predict Student Performance"
    )


# ============================================================
# INPUT DATA
# ============================================================

student_data = pd.DataFrame([{

    "Age": age,

    "StudyHoursPerWeek":
        study_hours,

    "AttendanceRate":
        attendance,

    "AssignmentCompletionRate":
        assignment,

    "PreviousGPA":
        previous_gpa,

    "TestAverage":
        test_average,

    "ClassParticipation":
        participation,

    "HomeworkHoursPerWeek":
        homework,

    "SleepHoursPerDay":
        sleep,

    "ScreenTimeHoursPerDay":
        screen_time,

    "ParentalSupport":
        parental_support,

    "MotivationLevel":
        motivation,

    "StressLevel":
        stress,

    "Tutoring":
        tutoring,

    "ExtracurricularActivities":
        extracurricular,

    "InternetAccess":
        internet

}])


# Keep exact training feature order
student_data = student_data[features]


# ============================================================
# MAIN AREA
# ============================================================

left, right = st.columns([1.25, 0.75])


with left:

    st.subheader("Prediction Workspace")

    st.write(
        "Review the student profile and run the trained model "
        "to estimate GPA."
    )

    with st.expander("View Student Input"):

        display_data = student_data.T.rename(
            columns={0: "Value"}
        )

        st.dataframe(
            display_data,
            width="stretch"
        )


with right:

    st.subheader("Model")
    st.write("Ridge Regression")
    st.write(
        "The selected model uses the project's original 16 numerical "
        "features and predicts GPA as a continuous value."
    )


# ============================================================
# PREDICTION
# ============================================================

if predict_button:

    predicted_gpa = model.predict(
        student_data
    )[0]

    predicted_gpa = float(
        np.clip(
            predicted_gpa,
            0.0,
            4.0
        )
    )


    # ========================================================
    # RISK LEVEL
    # ========================================================

    if predicted_gpa < 2.0:

        risk = "High Risk"
        risk_class = "risk-high"

    elif predicted_gpa < 2.5:

        risk = "Moderate Risk"
        risk_class = "risk-moderate"

    elif predicted_gpa < 3.0:

        risk = "Low Risk"
        risk_class = "risk-low"

    else:

        risk = "Good Performance"
        risk_class = "risk-good"


    # ========================================================
    # RESULT
    # ========================================================

    st.subheader("Predicted GPA")
    st.markdown(f"## {predicted_gpa:.2f} / 4.00")

    if risk == "Good Performance":
        st.success(risk)
    elif risk == "Low Risk":
        st.info(risk)
    elif risk == "Moderate Risk":
        st.warning(risk)
    else:
        st.error(risk)


    st.write("")


    # ========================================================
    # QUICK SNAPSHOT
    # ========================================================

    r1, r2, r3 = st.columns(3)

    with r1:

        st.metric(
            "Attendance",
            f"{attendance:.0f}%"
        )

    with r2:

        st.metric(
            "Test Average",
            f"{test_average:.0f}"
        )

    with r3:

        st.metric(
            "Study Hours",
            f"{study_hours:.1f}"
        )


    # ========================================================
    # RECOMMENDATIONS
    # ========================================================

    st.subheader(
        "Personalized Recommendations"
    )

    recommendations = []


    if study_hours < 10:

        recommendations.append(
            "Increase weekly study time and follow a consistent study routine."
        )


    if attendance < 75:

        recommendations.append(
            "Improve attendance and reduce missed classes."
        )


    if assignment < 80:

        recommendations.append(
            "Complete assignments consistently and avoid last-minute work."
        )


    if test_average < 60:

        recommendations.append(
            "Focus on revision, practice tests and weak topics."
        )


    if sleep < 6:

        recommendations.append(
            "Improve sleep consistency and aim for at least 6 hours."
        )


    if screen_time > 6:

        recommendations.append(
            "Reduce unnecessary screen time during study periods."
        )


    if stress >= 4:

        recommendations.append(
            "Use a structured study plan and include short breaks."
        )


    if motivation <= 2:

        recommendations.append(
            "Set smaller weekly academic goals to build progress."
        )


    if not recommendations:

        recommendations.append(
            "The current profile is balanced. Maintain these habits consistently."
        )


    for recommendation in recommendations:

        st.write(
            "•",
            recommendation
        )


# ============================================================
# ABOUT
# ============================================================

st.write("")

with st.expander("About the Project"):

    st.write(
        """
        Student Performance Prediction System is a
        regression-based academic support application.

        The model predicts a continuous GPA value using academic,
        behavioral and learning-related features.

        R² is used as the primary regression metric.
        MAE measures average absolute GPA prediction error.
        RMSE gives additional weight to larger errors.

        The ±0.30 GPA percentage is a separate tolerance metric.
        It should not be interpreted as standard classification accuracy.

        This application is intended for academic support and
        should not replace teacher, counselor or institutional judgment.
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.caption(
    "Student Performance Prediction System · Python · Scikit-learn · Streamlit"
)