from fastapi import FastAPI
from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.duckduckgo import DuckDuckGo
from phi.tools.arxiv_toolkit import ArxivToolkit
from phi.tools.googlesearch import GoogleSearch
from phi.tools.wikipedia import WikipediaTools
from phi.tools.pubmed import PubmedTools
from phi.model.deepseek import DeepSeekChat
import os
from dotenv import load_dotenv
import uvicorn

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI()

# Debugging Function to Check Each Agent
def debug_agent(agent, query):
    try:
        print(f"🔹 Running {agent.name}...")
        response = agent.run(query)
        print(f"✅ {agent.name} Response: {response}")
        return response
    except Exception as e:
        print(f"❌ {agent.name} Failed: {e}")
        return f"{agent.name} failed: {e}"

# ✅ Web Search Agent
web_search_agent = Agent(
    name="Web Search Agent",
    role="Search the web for information about Knee Osteoarthritis",
    model=Groq(id="llama-3.3-70b-versatile"),
    tools=[DuckDuckGo()],
    instructions=["Always include sources"],
    show_tool_calls=True,
    markdown=True,
)

# ✅ Arxiv Search Agent
arxiv_search_agent = Agent(
    name="Arxiv Search Agent",
    role="Search Arxiv for papers about Knee Osteoarthritis",
    model=Groq(id="llama-3.3-70b-versatile"),
    tools=[ArxivToolkit()],
    instructions=["Always include sources"],
    show_tool_calls=True,
    markdown=True,
)

# ✅ Google Search Agent
google_search_agent = Agent(
    tools=[GoogleSearch()],
    model=Groq(id="llama-3.3-70b-versatile"),
    description="Search for the latest news and articles in Knee Osteoarthritis.",
    instructions=["Provide 10 latest articles, then filter the top 4."],
    show_tool_calls=True,
    debug_mode=True,
)

# ✅ Wikipedia Search Agent
wikipedia_search_agent = Agent(
    tools=[WikipediaTools()],
    model=Groq(id="llama-3.3-70b-versatile"),
    description="Find important Wikipedia information about Knee Osteoarthritis.",
    instructions=["Provide 10 links, then filter the top 4."],
    show_tool_calls=True,
    debug_mode=True,
)

# ✅ PubMed Search Agent
pubmed_search_agent = Agent(
    tools=[PubmedTools()],
    model=Groq(id="llama-3.3-70b-versatile"),
    description="Search PubMed for latest research on Knee Osteoarthritis.",
    instructions=["Provide 10 latest research papers, then filter the top 4."],
    show_tool_calls=True,
    debug_mode=True,
)

# ✅ DeepSeek Agent (Ensure API Key is Loaded)
deepseek_model = DeepSeekChat(api_key=os.getenv("DEEPSEEK_API_KEY"))

deep_agent = Agent(
    model=deepseek_model,
    name="DeepSeekChat",
    id="deepseek-chat",
    description="AI expert for Knee Osteoarthritis research",
    markdown=True,
)

# ✅ Multi-Agent AI
multi_ai_agent = Agent(
    team=[web_search_agent, arxiv_search_agent, google_search_agent, wikipedia_search_agent, pubmed_search_agent, deep_agent],
    instructions=["Always include sources", "Use tables for data display"],
    show_tool_calls=True,
    markdown=True,
    stream=True,
)

# ✅ Root API Endpoint
@app.get("/")
def read_root():
    return {"message": "Welcome to the Knee Osteoarthritis AI Research API"}

# ✅ Search Endpoint - Calls All Agents Separately
@app.get("/search/knee-osteoarthritis")
def search_knee_osteoarthritis():
    responses = {
        "arxiv": debug_agent(arxiv_search_agent, "Summarize the latest research on Knee Osteoarthritis"),
        "pubmed": debug_agent(pubmed_search_agent, "Find the latest medical research on Knee Osteoarthritis"),
        "web": debug_agent(web_search_agent, "Find the latest news on Knee Osteoarthritis"),
        "google": debug_agent(google_search_agent, "Find the latest news on Knee Osteoarthritis"),
        "wikipedia": debug_agent(wikipedia_search_agent, "Find relevant Wikipedia articles on Knee Osteoarthritis"),
        "deepseek": debug_agent(deep_agent, "Summarize recent AI-based research on Knee Osteoarthritis"),
    }
    return {"responses": responses}

# ✅ Run FastAPI Server
if __name__ == "__main__":
    uvicorn.run("appy:app", host="127.0.0.1", port=8000, reload=True)
