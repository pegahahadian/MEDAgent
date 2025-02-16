import streamlit as st
import requests
import io
from PIL import Image

# Set Streamlit Page Configurations
st.set_page_config(page_title="Knee Osteoarthritis Research", layout="wide")

# Title and Header
st.title("🔍 Knee Osteoarthritis Research Assistant")
st.markdown("Upload an **X-ray or MRI image** to predict **OA progression**, and fetch related research papers.")

# Sidebar for Search
st.sidebar.header("🔎 Search Options")
search_query = st.sidebar.text_input("Enter search keywords", "Knee Osteoarthritis")

# Upload File Section
st.subheader("📤 Upload X-ray or MRI Image")
uploaded_file = st.file_uploader("Choose an image file (JPG, PNG)", type=["jpg", "png", "jpeg"])

if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    # Convert file to bytes
    file_bytes = uploaded_file.read()

    # Send file to FastAPI backend for prediction
    with st.spinner("🔄 Predicting..."):
        response = requests.post(
            "http://127.0.0.1:8000/predict",
            files={"file": (uploaded_file.name, file_bytes, "image/jpeg")}
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

# Search Button for Research Papers
if st.sidebar.button("Search Research Papers"):
    with st.spinner("🔄 Fetching latest research..."):
        try:
            response = requests.get("http://127.0.0.1:8000/search/knee-osteoarthritis")

            if response.status_code == 200:
                data = response.json()["responses"]

                # Display Arxiv Papers
                if "arxiv" in data:
                    st.subheader("📄 Research Papers from **Arxiv**")
                    st.markdown("\n".join(f"- {paper}" for paper in data["arxiv"].split("\n")))

                # Display PubMed Papers
                if "pubmed" in data:
                    st.subheader("📑 Research Papers from **PubMed**")
                    st.markdown("\n".join(f"- {paper}" for paper in data["pubmed"].split("\n")))

                # Display Web Search Results
                if "web" in data:
                    st.subheader("🌐 Latest News from **Web Search**")
                    st.markdown("\n".join(f"- {news}" for news in data["web"].split("\n")))

                # Display Google Search Results
                if "google" in data:
                    st.subheader("📰 Latest News from **Google**")
                    st.markdown("\n".join(f"- {news}" for news in data["google"].split("\n")))

                # Display Wikipedia Results
                if "wikipedia" in data:
                    st.subheader("📚 Information from **Wikipedia**")
                    st.markdown("\n".join(f"- {info}" for info in data["wikipedia"].split("\n")))

            else:
                st.error("❌ Failed to fetch search results.")

        except Exception as e:
            st.error(f"⚠️ Error: {e}")

# Footer
st.sidebar.write("🚀 Developed using FastAPI, Streamlit & PyTorch")
