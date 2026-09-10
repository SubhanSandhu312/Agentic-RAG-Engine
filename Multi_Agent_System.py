from dotenv import load_dotenv
import json
import os
from typing import TypedDict, List

from llm_calling import the_call
from Agent_prompts import CRITIC_SYSTEM_PROMPT, PLANNER_SYSTEM_PROMPT, SYNTHESIZER_SYSTEM_PROMPT
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI

load_dotenv()

# 1. State schema aligned with actual node outputs
class AgentState(TypedDict):
    messages: list
    query: str
    current_query: str
    retrieved_chunks: list
    critic_result: dict
    iteration_count: int
    final_answer: str

llm = ChatOpenAI(
    model="openrouter/free",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

# 2. Node Functions
def planner(state: AgentState):
    iteration = state.get("iteration_count", 0)
    user_query = state.get("query", "")
    critic_feedback = state.get("critic_result", {})
    
    # Supply failure reasoning to the planner if this is a retry loop
    if iteration > 0 and critic_feedback:
        prompt_content = (
            f"Original Query: {user_query}\n"
            f"Previous attempt was insufficient: {critic_feedback.get('reasoning', '')}\n"
            f"Missing context: {critic_feedback.get('missing_info', '')}\n"
            f"Formulate a refined, targeted search query."
        )
    else:
        prompt_content = f"Target technical query: {user_query}"

    messages = [
        SystemMessage(content=PLANNER_SYSTEM_PROMPT),
        HumanMessage(content=prompt_content)
    ]
    response = llm.invoke(messages)
    
    return {
        "current_query": response.content.strip(),
        "iteration_count": iteration + 1
    }

def retriever(state: AgentState):
    query_text = state.get("current_query", state.get("query", ""))
    context = the_call(query_text)
    
    # the_call returns chunk contents or metadata
    chunks = [context] if isinstance(context, str) else context
    return {"retrieved_chunks": chunks}

def critic_agent(state: AgentState):
    chunks_text = "\n\n".join(state.get("retrieved_chunks", []))
    prompt_content = (
        f"Original User Question: {state.get('query', '')}\n\n"
        f"Retrieved Evidence Chunks:\n{chunks_text}"
    )

    messages = [
        SystemMessage(content=CRITIC_SYSTEM_PROMPT),
        HumanMessage(content=prompt_content)
    ]
    response = llm.invoke(messages)
    
    try:
        # Parse JSON output from critic
        raw_content = response.content.strip()
        parsed_result = json.loads(raw_content)
    except Exception:
        parsed_result = {
            "verdict": "FAIL",
            "reasoning": "Output parsing failed.",
            "missing_info": "Unable to verify context validity."
        }

    return {"critic_result": parsed_result}

def Synthesizer(state: AgentState):
    chunks_text = "\n\n".join(state.get("retrieved_chunks", []))
    prompt_content = (
        f"User Inquiry: {state.get('query', '')}\n\n"
        f"Retrieved Context:\n{chunks_text}"
    )

    messages = [
        SystemMessage(content=SYNTHESIZER_SYSTEM_PROMPT),
        HumanMessage(content=prompt_content)
    ]
    response = llm.invoke(messages)
    return {"final_answer": response.content}

# 3. Router with retry limit safeguard
def router(state: AgentState):
    MAX_RETRIES = 3
    verdict = state.get("critic_result", {}).get("verdict", "FAIL")
    
    if verdict == "PASS":
        return "Synthesizer"
    
    if state.get("iteration_count", 0) < MAX_RETRIES:
        return "planner"  # Loop back to planner to rewrite the query
        
    return "Synthesizer"  # Graceful fallback when attempts are exhausted

# 4. Graph Construction
graph_builder = StateGraph(AgentState)

graph_builder.add_node("planner", planner)
graph_builder.add_node("retriever", retriever)
graph_builder.add_node("critic_agent", critic_agent)
graph_builder.add_node("Synthesizer", Synthesizer)

graph_builder.add_edge(START, "planner")
graph_builder.add_edge("planner", "retriever")
graph_builder.add_edge("retriever", "critic_agent")

graph_builder.add_conditional_edges(
    "critic_agent",
    router,
    {
        "planner": "planner",
        "Synthesizer": "Synthesizer"
    }
)
graph_builder.add_edge("Synthesizer", END)

graph = graph_builder.compile()

result = graph.invoke({
    "query": "Why does the Docker build fail?",
    "messages": [],
    "current_query": "",
    "retrieved_chunks": [],
    "critic_result": {},
    "iteration_count": 0,
    "final_answer": ""
})

print(result["final_answer"])