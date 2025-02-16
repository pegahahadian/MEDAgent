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
            response = requests.get(f"http://127.0.0.1:8000/search/knee-osteoarthritis")

            if response.status_code == 200:
                data = response.json()["response"]

                # Format Sections Separately
                if "---" in data:
                    sections = data.split("---")
                    latest_papers = sections[0].strip()
                    latest_news = sections[1].strip()
                else:
                    latest_papers = data
                    latest_news = ""

                # Display Results
                st.subheader("📄 Latest Research Papers")
                st.markdown(latest_papers, unsafe_allow_html=True)

                st.subheader("📰 Latest News Articles")
                if latest_news:
                    st.markdown(latest_news, unsafe_allow_html=True)
                else:
                    st.info("No recent news articles found.")

            else:
                st.error("❌ Failed to fetch search results. API might be down.")

        except Exception as e:
            st.error(f"⚠️ Error: {e}")

# Footer
st.sidebar.write("🚀 Developed using Phidata AI & Streamlit")
