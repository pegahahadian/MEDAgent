from fastapi import FastAPI, HTTPException
from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.duckduckgo import DuckDuckGo
import arxiv
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

# Web search agent
web_seach_agent = Agent(
    name="Web Search Agent",
    role="Search the web for information about Knee Osteoarthritis",
    model=Groq(id="llama-3.3-70b-versatile"),
    tools=[DuckDuckGo()],
    instructions=["Always include sources"],
    show_tool_calls=True,
    markdown=True,
)

# Arxiv search agent
Arxiv_search_agent = Agent(
    name="Arxiv Search Agent",
    role="Search Arxiv for papers about Knee Osteoarthritis",
    model=Groq(id="llama-3.3-70b-versatile"),
    tools=[ArxivToolkit()],
    instructions=["Always include sources"],
    show_tool_calls=True,
    markdown=True,
)

# Google search agent
Google_search_agent = Agent(
    tools=[GoogleSearch()],
    model=Groq(id="llama-3.3-70b-versatile"),
    description="Search for the latest news and articles in Knee Osteoarthritis.",
    instructions=["Provide 10 latest articles, then filter the top 4."],
    show_tool_calls=True,
    debug_mode=True,
)

# Wikipedia search agent
Wikipedia_search_agent = Agent(
    tools=[WikipediaTools()],
    model=Groq(id="llama-3.3-70b-versatile"),
    description="Find important Wikipedia information about Knee Osteoarthritis.",
    instructions=["Provide 10 links, then filter the top 4."],
    show_tool_calls=True,
    debug_mode=True,
)

# PubMed search agent
Pubmed_search_agent = Agent(
    tools=[PubmedTools()],
    model=Groq(id="llama-3.3-70b-versatile"),
    description="Search PubMed for latest research on Knee Osteoarthritis.",
    instructions=["Provide 10 latest research papers, then filter the top 4."],
    show_tool_calls=True,
    debug_mode=True,
)

# DeepSeek agent
Deep_agent = Agent(
    model=DeepSeekChat(),
    name="DeepSeekChat",
    id="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
    description="AI expert for Knee Osteoarthritis research",
    markdown=True,
)

# Multi-agent AI
multi_ai_agent = Agent(
    team=[web_seach_agent, Arxiv_search_agent, Google_search_agent, Wikipedia_search_agent, Pubmed_search_agent, Deep_agent],
    instructions=["Always include sources", "Use tables for data display"],
    show_tool_calls=True,
    markdown=True,
    stream=True,
)

@app.get("/")
def read_root():
    return {"message": "Welcome to the Knee Osteoarthritis AI Research API"}

@app.get("/search/knee-osteoarthritis")
def search_knee_osteoarthritis():
    response = multi_ai_agent.run("Summarize the latest papers and news about Knee Osteoarthritis")
    return {"response": response}

if __name__ == "__main__":
    uvicorn.run("appy:app", host="127.0.0.1", port=8000, reload=True)
