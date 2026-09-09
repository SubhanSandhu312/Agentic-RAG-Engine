from dotenv import load_dotenv
import uuid
import os

from llm_calling import the_call
from Agent_prompts import CRITIC_SYSTEM_PROMPT, PLANNER_SYSTEM_PROMPT, SYNTHESIZER_SYSTEM_PROMPT
from langchain_core.messages import SystemMessage
from langgraph.graph import StateGraph, START, END, MessagesState
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt import tools_condition

load_dotenv()


def router(state):
    if state["Critic_messages"][-1].content == "PASS":
        return "yes"
    return "No"

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

def Synthesizer(state):
    messages = [
        SystemMessage(content=SYNTHESIZER_SYSTEM_PROMPT)
    ] + state["messages"]

    response = llm.invoke(messages)
    return {"Synthesizer_messages": [response]}



graph_builder = StateGraph(MessagesState)

graph_builder.add_node(planner)
graph_builder.add_node(retriever)
graph_builder.add_node(critic_agent)
graph_builder.add_node(Synthesizer)

graph_builder.add_edge(START, "planner")
graph_builder.add_edge("planner", "retriever")
graph_builder.add_edge("retriever", "critic_agent")
graph_builder.add_conditional_edges(
    "critic_agent",
    router,
    {
        "yes": "retriever",
        "No": "Synthesizer"
    }
)
graph_builder.add_edge("Synthesizer", END)
