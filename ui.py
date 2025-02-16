import streamlit as st
import requests

# Set Streamlit Page Configurations
st.set_page_config(page_title="Knee Osteoarthritis Research", layout="wide")

# Title and Header
st.title("🔍 Knee Osteoarthritis Research Assistant")
st.markdown("This tool fetches the **latest research papers and news articles** on Knee Osteoarthritis using AI-powered multi-agent search.")

# Sidebar for Search
st.sidebar.header("🔎 Search Options")
search_query = st.sidebar.text_input("Enter search keywords", "Knee Osteoarthritis")

# Search Button
if st.sidebar.button("Search"):
    with st.spinner("🔄 Fetching latest research..."):
        try:
            # Call FastAPI
            response = requests.get("http://127.0.0.1:8000/search/knee-osteoarthritis")

            if response.status_code == 200:
                data = response.json()["responses"]  # Updated to match new API format
                
                # Display Arxiv Papers
                if "arxiv" in data:
                    st.subheader("📄 Research Papers from **Arxiv**")
                    st.markdown(data["arxiv"], unsafe_allow_html=True)

                # Display PubMed Papers
                if "pubmed" in data:
                    st.subheader("📑 Research Papers from **PubMed**")
                    st.markdown(data["pubmed"], unsafe_allow_html=True)

                # Display Web Search Results
                if "web" in data:
                    st.subheader("🌐 Latest News from **Web Search**")
                    st.markdown(data["web"], unsafe_allow_html=True)

                # Display Google Search Results
                if "google" in data:
                    st.subheader("📰 Latest News from **Google**")
                    st.markdown(data["google"], unsafe_allow_html=True)

                # Display Wikipedia Results
                if "wikipedia" in data:
                    st.subheader("📚 Information from **Wikipedia**")
                    st.markdown(data["wikipedia"], unsafe_allow_html=True)

                # Display DeepSeek Results
                if "deepseek" in data:
                    st.subheader("🤖 AI Research from **DeepSeek**")
                    st.markdown(data["deepseek"], unsafe_allow_html=True)

            else:
                st.error("❌ Failed to fetch search results. API might be down.")

        except Exception as e:
            st.error(f"⚠️ Error: {e}")

# Footer
st.sidebar.write("🚀 Developed using Phidata AI & Streamlit")
