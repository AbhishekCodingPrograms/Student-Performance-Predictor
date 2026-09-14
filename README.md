# Student Performance Predictor

A Streamlit web application that predicts a student's GPA using academic, attendance, study habit, and behavioral inputs.

## Features
- Simple black-and-white Streamlit dashboard
- GPA prediction using a trained scikit-learn model
- Student profile input form
- Personalized recommendations based on the entered profile

## Files
- `app.py` — Streamlit application
- `requirements.txt` — Python dependencies
- `student_performance_model.pkl` — trained model
- `student_performance_features.pkl` — feature order metadata

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Project structure

```text
student-performance/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── student_performance_model.pkl
├── student_performance_features.pkl
└── __pycache__/
```

## Notes
- The app is designed for academic support and should not replace teacher or counselor judgment.
- The model artifact is a scikit-learn pipeline and is loaded at runtime.
