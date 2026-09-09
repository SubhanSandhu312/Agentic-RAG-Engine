from dotenv import load_dotenv
import uuid
import os

from llm_calling import the_call
from Agent_prompts import CRITIC_SYSTEM_PROMPT, PLANNER_SYSTEM_PROMPT
from langchain_core.messages import SystemMessage
from langgraph.graph import StateGraph, START, END, MessagesState
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt import tools_condition

load_dotenv()



llm = ChatOpenAI(
    model="openrouter/free",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)


def planner(state):
    messages = [
        SystemMessage(content=PLANNER_SYSTEM_PROMPT)
    ] + state["messages"]

    response = llm.invoke(messages)
    return {"Planner_messages": [response]}

def retriever(state):
    # query = state["Planner_messages"][-1].content
    for query in state["Planner_messages"]:
        context = the_call(query)
    return {"Retriever_messages": [context]}

def critic_agent(state):
    messages = [
        SystemMessage(content=CRITIC_SYSTEM_PROMPT)
    ] + state["messages"]

    response = llm.invoke(messages)
    return {"Critic_messages": [response]}

