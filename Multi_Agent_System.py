from dotenv import load_dotenv
import json
import os
from typing import TypedDict, List

import mcp_client
from mcp_client import call_tool_json, get_llm_tool_schemas, MCPServerError, DestructiveToolBlockedError
from Agent_prompts import (
    CRITIC_SYSTEM_PROMPT,
    PLANNER_SYSTEM_PROMPT,
    SYNTHESIZER_SYSTEM_PROMPT,
    TOOL_AGENT_SYSTEM_PROMPT,
)
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI

load_dotenv()

# 1. State schema aligned with the project proposal.
#
# `retrieved_chunks` is kept (critic_agent/Synthesizer still join it into
# one context string, unchanged) alongside the new `tool_results`, which
# tracks structured metadata per tool call (which tool, what args, whether
# it succeeded) so the Critic can reason about WHICH tool produced the
# evidence and suggest an alternate one on the next attempt. `plan` is an
# append-only trace of the search goals/sub-queries tried so far.
class AgentState(TypedDict):
    messages: list
    query: str
    current_query: str
    plan: list
    retrieved_chunks: list
    tool_results: list
    critic_result: dict
    iteration_count: int
    final_answer: str

llm = ChatOpenAI(
    model="openrouter/free",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)


def router(state: AgentState):


    MAX_RETRIES = 3

    verdict = state.get(
        "critic_result", {}
    ).get(
        "verdict",
        "FAIL"
    )

    if verdict == "PASS":
        print("→ SYNTHESIZER")
        return "Synthesizer"

    if state.get("iteration_count", 0) < MAX_RETRIES:
        print("→ RETRY PLANNER")
        return "planner"

    return "Synthesizer"

def planner(state: AgentState):

    iteration = state.get("iteration_count", 0)
    user_query = state.get("query", "")
    critic_feedback = state.get("critic_result", {})

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


def _default_args_for(tool_name, query_text):
    """Build a reasonable default argument set for `tool_name` when no LLM
    tool_call is available to draw args from (fallback / critic-suggested
    dispatch paths below). retrieve_file takes a path rather than a query,
    so it's special-cased here purely for argument SHAPE - the actual tool
    execution is still fully dynamic (call_tool_json dispatches by name,
    no per-tool branching there)."""

    if tool_name == "retrieve_file":
        return {"path": query_text}
    return {"query": query_text, "top_k": 5}


def _mcp_unavailable_result(e):
    return {
        "tool_results": [{
            "tool_name": None,
            "args": None,
            "success": False,
            "error": str(e),
        }],
        "retrieved_chunks": [],
        "critic_result": {
            "verdict": "FAIL",
            "reasoning": "The retrieval MCP server is unavailable.",
            "missing_info": str(e),
        },
    }


def tool_agent(state: AgentState):
    """Tool Agent node (replaces the old deterministic `retriever` node).

    Converts the MCP server's dynamically-discovered tool definitions into
    LLM tool schemas and lets the model choose which tool(s) to call
    (search_documents / search_code / retrieve_file), instead of the graph
    hardcoding "search_documents". The chosen tool_calls are dispatched
    dynamically through mcp_client.call_tool_json(name, args) - there is
    no static if/elif branching on tool name for execution.

    Two deterministic paths exist alongside the model's own choice:
      - If the Critic previously suggested a specific alternate tool
        (critic_result["suggested_tool"]), that tool is called directly -
        this is the "fallback from dense search to search_code" behavior.
      - If the model doesn't return any tool_calls at all (e.g. the
        configured model/tier doesn't support function calling), the node
        falls back to the original Step 4 behavior: a plain
        search_documents call. This keeps the graph working end-to-end
        regardless of the underlying model's tool-calling support.
    """

    query_text = state.get("current_query", state.get("query", ""))
    plan_so_far = state.get("plan", [])
    tool_results_so_far = state.get("tool_results", [])
    critic_feedback = state.get("critic_result", {}) or {}

    try:
        tool_schemas = get_llm_tool_schemas()
    except MCPServerError as e:
        return _mcp_unavailable_result(e)

    suggested_tool = critic_feedback.get("suggested_tool") or None
    valid_tool_names = {t["function"]["name"] for t in tool_schemas}

    if suggested_tool in valid_tool_names:
        # Deterministic override: the Critic already identified which tool
        # is likely to do better - no need to ask the model to re-derive
        # that same conclusion.
        tool_calls = [{
            "name": suggested_tool,
            "args": _default_args_for(suggested_tool, query_text),
            "id": "critic-suggested",
        }]
    else:
        prior_attempts = "\n".join(
            f"- tool={r.get('tool_name')} args={r.get('args')} success={r.get('success')}"
            for r in tool_results_so_far
        ) or "None yet."

        prompt_content = (
            f"Current goal / search query: {query_text}\n\n"
            f"Tool calls already made for this question so far:\n{prior_attempts}\n\n"
            "Choose the single best tool call to make progress on this goal."
        )

        messages = [
            SystemMessage(content=TOOL_AGENT_SYSTEM_PROMPT),
            HumanMessage(content=prompt_content),
        ]

        response = None
        try:
            bound_llm = llm.bind_tools(tool_schemas)
            response = bound_llm.invoke(messages)
        except Exception:
            # Some models/tiers (e.g. certain OpenRouter free models) don't
            # support function calling at all and error out of bind_tools
            # or .invoke(); degrade to the deterministic fallback below
            # instead of crashing the graph.
            response = None

        tool_calls = getattr(response, "tool_calls", None) or []

        if not tool_calls:
            tool_calls = [{
                "name": "search_documents",
                "args": _default_args_for("search_documents", query_text),
                "id": "fallback",
            }]

    new_tool_results = list(tool_results_so_far)
    chunk_texts = []

    for call in tool_calls:
        tool_name = call.get("name") if isinstance(call, dict) else getattr(call, "name", None)
        tool_args = call.get("args") if isinstance(call, dict) else getattr(call, "args", {})
        tool_args = tool_args or {}

        try:
            result = call_tool_json(tool_name, tool_args)
        except MCPServerError as e:
            result = {"success": False, "results": [], "error": str(e)}
        except DestructiveToolBlockedError as e:
            # Step 5 preparation: a destructive tool was intercepted before
            # its JSON-RPC call. None of today's tools are destructive, so
            # this branch isn't exercised by mcp_server.py yet, but the
            # graph still degrades gracefully instead of crashing if it is.
            result = {"success": False, "results": [], "error": str(e)}

        new_tool_results.append({
            "tool_name": tool_name,
            "args": tool_args,
            "success": bool(result.get("success")),
            "error": result.get("error"),
        })

        if result.get("success"):
            if result.get("results"):
                for item in result["results"][:2]:
                    text = item.get("text")
                    if text:
                        chunk_texts.append(text)
            elif result.get("content"):
                chunk_texts.append(result["content"])

    return {
        "plan": plan_so_far + [query_text],
        "tool_results": new_tool_results,
        "retrieved_chunks": chunk_texts,
    }

def critic_agent(state: AgentState):

    chunks_text = "\n\n".join(
        state.get("retrieved_chunks", [])
    )

    tool_results = state.get("tool_results", [])
    tool_usage_summary = "\n".join(
        f"- {r.get('tool_name')} (success={r.get('success')})"
        + (f" error={r.get('error')}" if not r.get("success") and r.get("error") else "")
        for r in tool_results
    ) or "No tool calls recorded."

    prompt_content = (
        f"Original User Question: {state.get('query', '')}\n\n"
        f"Tool calls made so far:\n{tool_usage_summary}\n\n"
        f"Retrieved Evidence Chunks:\n{chunks_text}"
    )

    messages = [
        SystemMessage(content=CRITIC_SYSTEM_PROMPT),
        HumanMessage(content=prompt_content)
    ]

    response = llm.invoke(messages)


    try:
        parsed_result = json.loads(response.content.strip())
    except Exception:
        parsed_result = {
            "verdict": "FAIL",
            "reasoning": "Output parsing failed.",
            "missing_info": "Unable to verify context validity."
        }


    return {
        "critic_result": parsed_result
    }
def Synthesizer(state: AgentState):

    chunks_text = "\n\n".join(
        state.get("retrieved_chunks", [])
    )

    prompt_content = (
        f"User Inquiry: {state.get('query', '')}\n\n"
        f"Retrieved Context:\n{chunks_text}"
    )

    messages = [
        SystemMessage(content=SYNTHESIZER_SYSTEM_PROMPT),
        HumanMessage(content=prompt_content)
    ]

    response = llm.invoke(messages)


    return {
        "final_answer": response.content
    }

graph_builder = StateGraph(AgentState)

graph_builder.add_node("planner", planner)
graph_builder.add_node("tool_agent", tool_agent)
graph_builder.add_node("critic_agent", critic_agent)
graph_builder.add_node("Synthesizer", Synthesizer)

graph_builder.add_edge(START, "planner")
graph_builder.add_edge("planner", "tool_agent")
graph_builder.add_edge("tool_agent", "critic_agent")

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
    "plan": [],
    "retrieved_chunks": [],
    "tool_results": [],
    "critic_result": {},
    "iteration_count": 0,
    "final_answer": ""
})

print(result["final_answer"])
