from langchain.chat_models import ChatOpenAI
from langchain.tools.tavily_search import TavilySearchResults
from langchain.agents import initialize_agent, AgentType
from src.utils.load_env import load_all_env

load_all_env()

openai_llm = ChatOpenAI(model="gpt-4o", temperature=0)
search_tool = TavilySearchResults(k=3)

agent = initialize_agent(
    tools=[search_tool],
    llm=openai_llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

result = agent.run("Who won the last Tour de France?")
print(result)
