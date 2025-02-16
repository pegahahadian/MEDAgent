from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.duckduckgo import DuckDuckGo
import arxiv
from phi.tools.arxiv_toolkit import ArxivToolkit  # Possible alternative
from phi.tools.googlesearch import GoogleSearch
from phi.tools.wikipedia import WikipediaTools
from pycountry import pycountry
from phi.tools.pubmed import PubmedTools
from phi.model.deepseek import DeepSeekChat


import openai
import os
from dotenv import load_dotenv
load_dotenv()
openai.api_key=os.getenv("OPENAI_API_KEY")

##web search agent
web_seach_agent=Agent(
    name="Web Search Agent",
    role="Search the web for the information about Knee Osteoarthritis",
    #instructions="You are an expert in knee asteoarthritis and professional researcher that can search through the web and find the best and latest news, articles, and papers in knee asteoarthritis.",
    model=Groq(id="llama-3.3-70b-versatile"),
    tools=[DuckDuckGo()],
    instructions=["Always include sources"],
    show_tool_calls=True,
    markdown=True,
)

##Arxiv search agent
Arxiv_seach_agent=Agent(
    name="Arxiv Search Agent",
    role="Search the Arxiv for the Papers about Knee Osteoarthritis",
    model=Groq(id="llama-3.3-70b-versatile"),
    tools=[ArxivToolkit()],
    instructions=["Always include sources"],
    show_tool_calls=True,
    markdown=True,
)

#Google Search
Google_search_agent = Agent(
    tools=[GoogleSearch()],
    model=Groq(id="llama-3.3-70b-versatile"),
    description="You are a domain expert agent that helps users find the latest news and articles in Knee Osteoarthritis.",
    instructions=[
        "Given a keywords by the user in medical- Knee Osteoarthritis domain, respond with 10 latest news and articles items about the topic.",
        "Search for 10 new articles and papers and select the top 4 unique items.",
        "Search in English and in Chinese.",
    ],
    show_tool_calls=True,
    debug_mode=True,
)
Wikipedia_search_agent = Agent(
    tools=[WikipediaTools()],
    model=Groq(id="llama-3.3-70b-versatile"),
    description="You are a domain expert agent that helps users find the Important information in Knee Osteoarthritis.",
    instructions=[
        "Given a keywords by the user in medical- Knee Osteoarthritis domain, respond with 10 important links about the topic.",
        "Search for the latest 10 items and select the top 4 unique items.",
        "Search in English and in Chinese.",
    ],
    show_tool_calls=True,
    debug_mode=True,
)

Pubmed_search_agent = Agent(
    tools=[PubmedTools()],
    model=Groq(id="llama-3.3-70b-versatile"),
    description="You are a domain expert search-agent that helps users find the Important arthicles in Knee Osteoarthritis.",
    instructions=[
        "Given a keywords by the user in medical- Knee Osteoarthritis domain, respond with 10 important latest articles and papers about the topic.",
        "Search for the latest 10 items and select the top 4 unique items.",
        "Search in English and in Chinese.",
    ],
    show_tool_calls=True,
    debug_mode=True,
)
#Deepseek
Deep_agent = Agent(model=DeepSeekChat(),
                   name="DeepSeekChat",
                   id="deepseek-chat",
                   api_key="DEEPSEEK_API_KEY",
                   base_url="https://api.deepseek.com",
                   description="You help user with Knee Osteoarthritis questions and papers/articles in Knee Osteoarthritis domain.",
                   markdown=True)

multi_ai_agent=Agent(
    team=[web_seach_agent, Arxiv_seach_agent, Google_search_agent, Wikipedia_search_agent, Pubmed_search_agent, Deep_agent],
    instructions=["Always include sources", "Use tables to display the data"],
    show_tool_calls=True,
    markdown=True,
    stream=True,
)

multi_ai_agent.print_response("Summarize List of papers and news about Knee Osteoarthritis", stream=True)