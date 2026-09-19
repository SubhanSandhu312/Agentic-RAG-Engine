from dotenv import load_dotenv
import json
import os
import uuid
from typing import TypedDict, List

import mcp_client
from mcp_client import (
    call_tool_json,
    get_llm_tool_schemas,
    MCPServerError,
    DestructiveToolBlockedError,
    DestructiveActionRejectedError,
    DESTRUCTIVE_TOOLS,
)
from Agent_prompts import (
    CRITIC_SYSTEM_PROMPT,
    PLANNER_SYSTEM_PROMPT,
    SYNTHESIZER_SYSTEM_PROMPT,
    TOOL_AGENT_SYSTEM_PROMPT,
)
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command
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
    # Step 5: tool_agent SELECTS a tool call and stores it here without
    # executing it; tool_executor reads it back and does the actual
    # dispatch/interrupt. Splitting selection from execution into two
    # checkpointed state fields (rather than one node doing both) means
    # that when a destructive call interrupts and the graph later resumes,
    # LangGraph only re-runs the cheap, deterministic "read state and
    # dispatch" step - NOT the LLM tool-selection call - so the human is
    # guaranteed to be approving/rejecting/editing the exact same call that
    # ultimately executes.
    pending_tool_calls: list
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
    dispatch paths below). retrieve_file takes a path rather than a query
    and create_issue takes a title/body rather than a query, so both are
    special-cased here purely for argument SHAPE - the actual tool
    execution is still fully dynamic (call_tool_json dispatches by name,
    no per-tool branching there)."""

    if tool_name == "retrieve_file":
        return {"path": query_text}
    if tool_name == "create_issue":
        title = (query_text or "Untitled issue").strip()[:120]
        return {"title": title, "body": query_text}
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


def _was_recently_rejected(tool_name, tool_args, tool_results_so_far):
    """True if a human already rejected this EXACT (tool_name, args) pair
    earlier in this run. Used to stop the Tool Agent from immediately
    re-proposing a destructive call a human just said no to (Step 5 safety
    requirement: no repeated identical destructive attempts)."""

    for r in tool_results_so_far:
        if (
            r.get("status") == "rejected"
            and r.get("tool_name") == tool_name
            and r.get("args") == tool_args
        ):
            return True
    return False


def tool_agent(state: AgentState):
    """Tool Agent node: SELECTS a tool call, but does not execute it.

    Converts the MCP server's dynamically-discovered tool definitions into
    LLM tool schemas and lets the model choose which tool(s) to call
    (search_documents / search_code / retrieve_file / create_issue),
    instead of the graph hardcoding a tool name. The choice is written to
    `pending_tool_calls` for `tool_executor` (the next node) to dispatch.

    Selection is intentionally split from execution (Step 5): a destructive
    tool call needs to pause for human approval via LangGraph's
    interrupt(), and interrupt() only guarantees "resume returns the exact
    same call" for code that runs AFTER the point state was last
    checkpointed. Since this node calls the LLM (non-deterministic), that
    call must happen and be checkpointed BEFORE any interrupt - so it
    lives here, one node earlier than the actual call_tool_json dispatch.

    Three paths decide `pending_tool_calls`:
      - If the Critic previously suggested a specific alternate tool
        (critic_result["suggested_tool"]), that tool is called directly.
      - Otherwise the model chooses via bind_tools/tool_calls.
      - If the model returns no tool_calls at all (e.g. the configured
        model/tier doesn't support function calling), fall back to the
        original Step 4 behavior: a plain search_documents call.

    A tool call that exactly repeats an already-rejected destructive call
    is swapped for a safe search_documents fallback instead of being
    proposed again (see _was_recently_rejected).
    """

    query_text = state.get("current_query", state.get("query", ""))
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
            + (f" status={r.get('status')}" if r.get("status") else "")
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

    # Normalize each call to a plain dict and swap out any exact repeat of
    # an already-rejected destructive call for a safe fallback.
    normalized_calls = []
    for call in tool_calls:
        name = call.get("name") if isinstance(call, dict) else getattr(call, "name", None)
        args = call.get("args") if isinstance(call, dict) else getattr(call, "args", {})
        args = args or {}
        call_id = call.get("id") if isinstance(call, dict) else getattr(call, "id", None)

        if name in DESTRUCTIVE_TOOLS and _was_recently_rejected(name, args, tool_results_so_far):
            normalized_calls.append({
                "name": "search_documents",
                "args": _default_args_for("search_documents", query_text),
                "id": "post-rejection-fallback",
            })
        else:
            normalized_calls.append({"name": name, "args": args, "id": call_id})

    return {
        "plan": state.get("plan", []) + [query_text],
        "pending_tool_calls": normalized_calls,
    }


def tool_executor(state: AgentState):
    """Executes the tool call(s) `tool_agent` selected (Step 5).

    Reads `pending_tool_calls` from checkpointed state - deterministic, no
    LLM call - and dispatches each one through mcp_client.call_tool_json,
    exactly as the original Step 4 dispatch loop did. The only new
    behavior is what happens for a DESTRUCTIVE_TOOLS name:

      - call_tool_json calls interrupt() internally, which pauses the
        WHOLE GRAPH right here (before the MCP action executes) and
        checkpoints state. Nothing after this point has run yet.
      - On resume, LangGraph re-invokes this node from the top. Since
        `pending_tool_calls` is unchanged (read from checkpointed state,
        not re-derived), the same call is dispatched again - but this time
        the interrupt() call returns the human's decision instead of
        pausing again.
      - Approved -> call_tool_json proceeds (with edited args if the human
        provided them) and the MCP tool actually runs.
      - Rejected -> DestructiveActionRejectedError is caught here and
        turned into a structured, non-crashing tool_result the Critic can
        see (status="rejected"), never executing the tool.
    """

    tool_calls = state.get("pending_tool_calls", []) or []
    tool_results_so_far = state.get("tool_results", [])

    new_tool_results = list(tool_results_so_far)
    chunk_texts = []

    for call in tool_calls:
        tool_name = call.get("name")
        tool_args = call.get("args") or {}
        status = None

        try:
            result = call_tool_json(tool_name, tool_args)
        except MCPServerError as e:
            result = {"success": False, "results": [], "error": str(e)}
        except DestructiveActionRejectedError as e:
            result = {"success": False, "results": [], "error": str(e)}
            status = "rejected"
        except DestructiveToolBlockedError as e:
            # interrupt() itself could not be posed (no checkpointer/thread
            # in this run) - fail safe, the tool never executed.
            result = {"success": False, "results": [], "error": str(e)}
            status = "blocked"

        if status is None:
            status = "approved" if tool_name in DESTRUCTIVE_TOOLS and result.get("success") else None

        entry = {
            "tool_name": tool_name,
            "args": tool_args,
            "success": bool(result.get("success")),
            "error": result.get("error"),
        }
        if status:
            entry["status"] = status
        new_tool_results.append(entry)

        if result.get("success"):
            if result.get("results"):
                for item in result["results"][:2]:
                    text = item.get("text")
                    if text:
                        chunk_texts.append(text)
            elif result.get("content"):
                chunk_texts.append(result["content"])
            elif result.get("issue"):
                chunk_texts.append(json.dumps(result["issue"]))

    return {
        "tool_results": new_tool_results,
        "retrieved_chunks": chunk_texts,
        "pending_tool_calls": [],
    }


def critic_agent(state: AgentState):

    chunks_text = "\n\n".join(
        state.get("retrieved_chunks", [])
    )

    tool_results = state.get("tool_results", [])
    tool_usage_summary = "\n".join(
        f"- {r.get('tool_name')} (success={r.get('success')})"
        + (f" status={r.get('status')}" if r.get("status") else "")
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
graph_builder.add_node("tool_executor", tool_executor)
graph_builder.add_node("critic_agent", critic_agent)
graph_builder.add_node("Synthesizer", Synthesizer)

graph_builder.add_edge(START, "planner")
graph_builder.add_edge("planner", "tool_agent")
graph_builder.add_edge("tool_agent", "tool_executor")
graph_builder.add_edge("tool_executor", "critic_agent")

graph_builder.add_conditional_edges(
    "critic_agent",
    router,
    {
        "planner": "planner",
        "Synthesizer": "Synthesizer"
    }
)
graph_builder.add_edge("Synthesizer", END)

# Step 5: a real checkpointer is REQUIRED for interrupt()/Command(resume=...)
# to actually pause and resume the graph rather than raising a bare
# RuntimeError (see mcp_client._guard_destructive_tool). MemorySaver keeps
# checkpoints in this process's memory only - sufficient for this project
# (a single long-lived process per run) without adding a database.
checkpointer = MemorySaver()
graph = graph_builder.compile(checkpointer=checkpointer)


def _initial_state(query):
    return {
        "query": query,
        "messages": [],
        "current_query": "",
        "plan": [],
        "retrieved_chunks": [],
        "tool_results": [],
        "pending_tool_calls": [],
        "critic_result": {},
        "iteration_count": 0,
        "final_answer": "",
    }


def _print_approval_request(payload):
    print("\n" + "-" * 50)
    print("HUMAN APPROVAL REQUIRED")
    print("-" * 50)
    print(f"\nTool:\n{payload.get('tool')}\n")
    print(f"Arguments:\n{json.dumps(payload.get('args'), indent=2)}\n")
    print(f"Message:\n{payload.get('message')}\n")
    print("-" * 50)


def _prompt_human_decision(payload):
    """Reads a decision from stdin for one interrupt payload. Returns a
    dict shaped exactly like what mcp_client._guard_destructive_tool
    expects back from Command(resume=...): {"approved": bool, ...}."""

    while True:
        choice = input("[y] Approve  [e] Edit  [n] Reject > ").strip().lower()

        if choice == "y":
            return {"approved": True}

        if choice == "n":
            reason = input("Reason for rejection (optional): ").strip()
            return {"approved": False, "reason": reason or "User rejected this action."}

        if choice == "e":
            original_args = payload.get("args") or {}
            edited_args = dict(original_args)
            print("Editing arguments (press Enter to keep the current value):")
            for key, value in original_args.items():
                new_value = input(f"  {key} [{value}]: ")
                if new_value.strip():
                    edited_args[key] = new_value
            return {"approved": True, "args": edited_args}

        print("Please enter 'y' (approve), 'e' (edit), or 'n' (reject).")


def run_query(query, thread_id=None):
    """Runs the graph for `query`, driving it through as many interrupts
    as it takes to reach completion (Step 5 requires supporting more than
    one, e.g. a rejected action followed later by a different destructive
    proposal). The SAME thread_id is reused for every resume so LangGraph
    continues the one checkpointed execution rather than starting a new
    run."""

    thread_id = thread_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    result = graph.invoke(_initial_state(query), config=config)

    while "__interrupt__" in result and result["__interrupt__"]:
        interrupt_obj = result["__interrupt__"][0]
        payload = interrupt_obj.value
        _print_approval_request(payload)
        decision = _prompt_human_decision(payload)
        result = graph.invoke(Command(resume=decision), config=config)

    return result


if __name__ == "__main__":
    final_state = run_query("Why does the Docker build fail?")
    print(final_state.get("final_answer", ""))
