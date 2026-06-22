
import streamlit as st
import requests
import plotly.graph_objects as go

# Set page layout
st.set_page_config(page_title="Student Depression Predictor", layout="centered")

st.title("🧠 Student Depression Predictor")

# Input Form
with st.form("prediction_form"):
    gender = st.selectbox("Gender", ["Male", "Female"])
    age = st.number_input("Age", min_value=15.0, max_value=100.0, value=22.0)
    academic_pressure = st.slider("Academic Pressure", 0.0, 5.0, 3.0)
    cgpa = st.number_input("CGPA", min_value=0.0, max_value=10.0, value=3.6)
    study_satisfaction = st.slider("Study Satisfaction", 0.0, 5.0, 4.0)
    job_satisfaction = st.slider("Job Satisfaction", 0.0, 5.0, 0.0)
    sleep_duration = st.selectbox("Sleep Duration", ["Less than 5 hours", "5-6 hours", "7-8 hours", "More than 8 hours"])
    dietary_habits = st.selectbox("Dietary Habits", ["Unhealthy", "Moderate", "Healthy"])
    work_study_hours = st.number_input("Work/Study Hours", min_value=0.0, max_value=24.0, value=6.0)
    financial_stress = st.slider("Financial Stress", 0.0, 5.0, 2.0)
    family_history = st.selectbox("Family History of Mental Illness", ["No", "Yes"])
    
    submit = st.form_submit_button("Predict")

if submit:
    # Prepare data
    input_data = {
        "Gender": gender,
        "Age": age,
        "Academic_Pressure": academic_pressure,
        "CGPA": cgpa,
        "Study_Satisfaction": study_satisfaction,
        "Job_Satisfaction": job_satisfaction,
        "Sleep_Duration": sleep_duration,
        "Dietary_Habits": dietary_habits,
        "Work_Study_Hours": work_study_hours,
        "Financial_Stress": financial_stress,
        "Family_History_of_Mental_Illness": family_history
    }

    try:
        # Call FastAPI backend
        response = requests.post("http://127.0.0.1:8000/docs#/default/predict_depression_predict_post", json=input_data)
        
        if response.status_code == 200:
            result = response.json()
            prob = result["probability"] * 100
            
            st.success(f"Status: {result['status']}")
            
            # Display Gauge
            fig = go.Figure(go.Indicator(
                mode="gauge+number", value=prob,
                title={'text': "Risk Probability"},
                gauge={'axis': {'range': [0, 100]}, 'bar': {'color': "red" if prob > 50 else "green"}}
            ))
            st.plotly_chart(fig)
        else:
            st.error(f"Error: {response.text}")
    except Exception as e:
        st.error(f"Could not connect to backend: {e}")

