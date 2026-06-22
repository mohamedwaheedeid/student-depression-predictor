import gradio as gr
import requests
import numpy as np

def predict(gender, age, academic_pressure, cgpa, study_satisfaction, job_satisfaction, 
            sleep_duration, dietary_habits, work_study_hours, financial_stress, family_history):
    
    input_data = {
        "Gender": gender,
        "Age": float(age),
        "Academic_Pressure": float(academic_pressure),
        "CGPA": float(cgpa),
        "Study_Satisfaction": float(study_satisfaction),
        "Job_Satisfaction": float(job_satisfaction),
        "Sleep_Duration": sleep_duration,
        "Dietary_Habits": dietary_habits,
        "Work_Study_Hours": float(work_study_hours),
        "Financial_Stress": float(financial_stress),
        "Family_History_of_Mental_Illness": family_history
    }
    
    try:
        response = requests.post("http://127.0.0.1:8000/predict", json=input_data)
        if response.status_code == 200:
            res = response.json()
            return f"Status: {res['status']} | Probability: {res['probability']*100:.2f}%"
        else:
            return f"Error: {response.text}"
    except Exception as e:
        return f"Could not connect to backend: {e}"

# Build the Interface
demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.Dropdown(["Male", "Female"], label="Gender"),
        gr.Number(value=22, label="Age"),
        gr.Slider(0, 5, value=3, label="Academic Pressure"),
        gr.Number(value=3.6, label="CGPA"),
        gr.Slider(0, 5, value=4, label="Study Satisfaction"),
        gr.Slider(0, 5, value=0, label="Job Satisfaction"),
        gr.Dropdown(["Less than 5 hours", "5-6 hours", "7-8 hours", "More than 8 hours"], label="Sleep Duration"),
        gr.Dropdown(["Unhealthy", "Moderate", "Healthy"], label="Dietary Habits"),
        gr.Number(value=6, label="Work/Study Hours"),
        gr.Slider(0, 5, value=2, label="Financial Stress"),
        gr.Dropdown(["No", "Yes"], label="Family History of Mental Illness")
    ],
    outputs="text",
    title="Student Depression Predictor"
)

if __name__ == "__main__":
    demo.launch(share=True)