import streamlit as  st
import requests

# FastAPI endpoint
API_URL = "http://127.0.0.1:8000/predict"

st.set_page_config(page_title="Diabetes Prediction App",layout="centered")

st.title("Diabetes Prediction App")
st.write("Enter health indicators to predict diabetes status")

# ------------- INPUT FORM -------------
with st.form("prediction_form"):
    HighBP = st.selectbox("High Blood Pressure (1=Yes, 0=No)", [0, 1])
    HighChol = st.selectbox("High Cholesterol (1=Yes, 0=No)", [0, 1])
    CholCheck = st.selectbox("Cholesterol Check (1=Yes, 0=No)", [0, 1])
    BMI = st.number_input("BMI", 10.0, 80.0, 25.0)
    Smoker = st.selectbox("Smoker (1=Yes, 0=No)", [0, 1])
    Stroke = st.selectbox("Stroke (1=Yes, 0=No)", [0, 1])
    HeartDiseaseorAttack = st.selectbox("Heart Disease/Attack (1=Yes, 0=No)", [0, 1])
    PhysActivity = st.selectbox("Physical Activity (1=Yes, 0=No)", [0, 1])
    Fruits = st.selectbox("Fruits Daily (1=Yes, 0=No)", [0, 1])
    Veggies = st.selectbox("Vegetables Daily (1=Yes, 0=No)", [0, 1])
    HvyAlcoholConsump = st.selectbox("Heavy Alcohol Consumption (1=Yes, 0=No)", [0, 1])
    AnyHealthcare = st.selectbox("Any Healthcare Coverage (1=Yes, 0=No)", [0, 1])
    NoDocbcCost = st.selectbox("Couldn't see doctor due to cost (1=Yes, 0=No)", [0, 1])
    GenHlth = st.slider("General Health (1=Excellent → 5=Poor)", 1, 5, 3)
    MentHlth = st.slider("Mental Health Bad Days (0 → 30)", 0, 30, 10)
    PhysHlth = st.slider("Physical Health Bad Days (0 → 30)", 0, 30, 10)
    DiffWalk = st.selectbox("Difficulty Walking (1=Yes, 0=No)", [0, 1])
    Sex = st.selectbox("Sex (1=Male, 0=Female)", [0, 1])
    Age = st.slider("Age Category (1→100)", 1, 100, 5)
    Education = st.slider("Education Level (1→6)", 1, 6, 4)
    Income = st.slider("Income Level (1→8)", 1, 8, 4)

    submit = st.form_submit_button("Predict")

# ------------- PREDICTION -------------

if submit:
    payload = {
    "HighBP": HighBP,
    "HighChol": HighChol,
    "CholCheck": 1,
    "BMI": BMI,
    "Smoker": Smoker,
    "Stroke": Stroke,
    "HeartDiseaseorAttack": HeartDiseaseorAttack,
    "PhysActivity": PhysActivity,
    "Fruits": 1,
    "Veggies": 1,
    "HvyAlcoholConsump": 0,
    "AnyHealthcare": 1,
    "NoDocbcCost": 0,
    "GenHlth": GenHlth,
    "MentHlth": 5,
    "PhysHlth": 5,
    "DiffWalk": 0,
    "Sex": 1,
    "Age": Age,
    "Education": 4,
    "Income": Income
}

    
    
    try :
        
        response = requests.post(API_URL,json=payload)
        response.raise_for_status()
        result = response.json()
        
        st.success(f" Prediction: **{result['predicted_class']}**")
        
        st.subheader("Class Probabilities")
        for k, v in result["probabilities"].items():
            st.write(f"{k}: {v:2f}")
            
    except Exception as e:
        st.error("Server error")
        st.exception(e)
        