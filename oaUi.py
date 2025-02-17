import streamlit as st
import requests
import io
from PIL import Image

# Set Streamlit Page Configurations
st.set_page_config(page_title="Knee Osteoarthritis Research", layout="wide")

# Title and Header
st.title("🔍 Knee Osteoarthritis Research Assistant")
st.markdown("Upload an **X-ray or MRI image**, input clinical data, and predict **OA progression**.")

# Sidebar for Search
st.sidebar.header("🔎 Search Options")
search_query = st.sidebar.text_input("Enter search keywords", "Knee Osteoarthritis")

# Upload File Section
st.subheader("📤 Upload X-ray or MRI Image")
uploaded_file = st.file_uploader("Choose an image file (JPG, PNG)", type=["jpg", "png", "jpeg"])

# Clinical Data Inputs
st.subheader("📋 Clinical Data")
age = st.number_input("Age", min_value=30, max_value=100, value=50)
bmi = st.number_input("BMI", min_value=15.0, max_value=50.0, value=25.0)
sex = st.radio("Sex", ["Male", "Female"])
sex = 1 if sex == "Male" else 0
knee_pain = st.radio("Experiencing Knee Pain?", ["Yes", "No"])
knee_pain = 1 if knee_pain == "Yes" else 0
xray_grade = st.slider("X-ray Grade (OA Severity)", min_value=0, max_value=4, value=2)
prior_injury = st.radio("Prior Knee Injury?", ["Yes", "No"])
prior_injury = 1 if prior_injury == "Yes" else 0

# Predict Button
if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    # Convert file to bytes
    file_bytes = uploaded_file.read()

    if st.button("🩺 Predict Progression"):
        with st.spinner("🔄 Predicting..."):
            response = requests.post(
                "http://127.0.0.1:8000/predict",
                files={"file": (uploaded_file.name, file_bytes, "image/jpeg")},
                data={
                    "age": age,
                    "bmi": bmi,
                    "sex": sex,
                    "knee_pain": knee_pain,
                    "xray_grade": xray_grade,
                    "prior_injury": prior_injury
                }
            )

        if response.status_code == 200:
            result = response.json()
            st.success(f"📊 Prediction: **{result['prediction']}**")
            st.write(f"📈 Likelihood of OA Progression: **{result['confidence']:.2%}**")

            # Display recommendations from AI
            st.subheader("💡 AI Suggestions")
            st.write(result["recommendation"])

        else:
            st.error("⚠️ Failed to get prediction. Try again.")

# Footer
st.sidebar.write("🚀 Developed using FastAPI, Streamlit & PyTorch")
