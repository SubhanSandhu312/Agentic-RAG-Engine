"""Standalone verification script for Step 5 (Human-in-the-Loop + controlled
tool autonomy), in the same style as test_mcp_integration.py: a plain
script using assertions, runnable directly with the project's interpreter:

    python test_hil_integration.py

Unlike test_mcp_integration.py (which talks to the real MCP server through
mcp_client.py only), this file drives the ACTUAL LangGraph agent defined in
Multi_Agent_System.py - the real StateGraph, the real MemorySaver
checkpointer, real interrupt()/Command(resume=...) round-trips, and the
real MCP server subprocess for tool execution. The only thing replaced is
the ChatOpenAI LLM object (`Multi_Agent_System.llm`), swapped for a small
deterministic FakeLLM so these tests don't need network access to
OpenRouter and don't depend on any particular model's behavior - the
Planner/Critic/Tool-Agent NODE LOGIC under test is 100% the real,
unmodified project code.

Covers:
  1. A read-only tool call reaches the Critic with NO interrupt.
  2. A destructive tool call (create_issue) genuinely pauses the graph
     BEFORE anything is persisted, and the interrupt payload names the
     tool and arguments.
  3. Approving a paused create_issue resumes the SAME thread, executes the
     action through the real MCP server, and the issue is persisted.
  4. Rejecting a paused create_issue does NOT execute it, is recorded as a
     structured, non-crashing tool_result, and the graph keeps going.
  5. Editing the arguments during approval persists the EDITED values, not
     the original proposal.
  6. A tool-level MCP failure (unknown tool name) becomes a structured
     tool_result instead of crashing the graph.
  Bonus: a rejected destructive call is not immediately re-proposed -
     the very next attempt automatically falls back to a read-only tool.

Every assertion failure raises AssertionError with a clear message; a
clean run prints "ALL HIL INTEGRATION CHECKS PASSED" and exits 0.
"""

import json
import uuid
from pathlib import Path

import Multi_Agent_System as mas
from langgraph.types import Command

ISSUES_FILE = Path(__file__).resolve().parent / "data" / "issues.json"


def check(condition, message):
    if not condition:
        raise AssertionError(f"FAILED: {message}")
    print(f"OK: {message}")


# --- A tiny, deterministic stand-in for ChatOpenAI --------------------------
#
# Real node logic (planner/tool_agent/critic_agent/tool_executor/router) is
# exercised unchanged; only the LLM calls those nodes make are faked, so no
# network access is required and tool selection / critic verdicts are
# fully controlled per test.

class _FakeResponse:
    def __init__(self, tool_calls=None, content=""):
        self.tool_calls = tool_calls or []
        self.content = content


class _FakeBoundLLM:
    def __init__(self, tool_calls):
        self._tool_calls = tool_calls

    def invoke(self, messages):
        return _FakeResponse(tool_calls=self._tool_calls)


class FakeLLM:
    """`next_tool_calls` controls what the Tool Agent's bind_tools().invoke()
    returns. `critic_verdicts` is consumed in order, one per critic_agent
    call within a test (the last entry repeats once exhausted), so a
    single test can script e.g. FAIL-then-PASS across a retry loop."""

    def __init__(self):
        self.next_tool_calls = []
        self.critic_verdicts = [{"verdict": "PASS", "reasoning": "ok", "missing_info": ""}]
        self._critic_calls = 0

    def invoke(self, messages):
        system_content = messages[0].content if messages else ""
        if "Sufficiency Critic" in system_content:
            idx = min(self._critic_calls, len(self.critic_verdicts) - 1)
            verdict = self.critic_verdicts[idx]
            self._critic_calls += 1
            return _FakeResponse(content=json.dumps(verdict))
        # Planner and Synthesizer both just need *some* content back.
        return _FakeResponse(content="fake search query")

    def bind_tools(self, schemas):
        return _FakeBoundLLM(self.next_tool_calls)


def _new_thread():
    return str(uuid.uuid4())


def _config(thread_id):
    return {"configurable": {"thread_id": thread_id}}


def _read_issues():
    if not ISSUES_FILE.exists():
        return None
    with open(ISSUES_FILE, "r", encoding="utf-8") as f:
        return f.read()


def main():
    # Start every test from a clean issues.json so assertions about "no
    # issue was created" / "exactly one issue" are unambiguous.
    if ISSUES_FILE.exists():
        ISSUES_FILE.unlink()

    fake = FakeLLM()
    mas.llm = fake

    # === 1. Read-only tool: no interrupt, result reaches the graph =========
    print("=== 1. Read-only tool (search_documents) does not interrupt ===")
    fake.next_tool_calls = [{
        "name": "search_documents",
        "args": {"query": "docker build fail dependency", "top_k": 3},
        "id": "t1",
    }]
    fake.critic_verdicts = [{"verdict": "PASS", "reasoning": "sufficient", "missing_info": ""}]
    fake._critic_calls = 0

    thread_1 = _new_thread()
    result = mas.graph.invoke(mas._initial_state("Why does the Docker build fail?"), config=_config(thread_1))

    check("__interrupt__" not in result, "search_documents does not pause the graph")
    check(len(result["tool_results"]) == 1, "exactly one tool call was recorded")
    check(result["tool_results"][0]["tool_name"] == "search_documents", "the recorded call is search_documents")
    check(result["tool_results"][0]["success"] is True, "search_documents succeeded")
    check(bool(result["final_answer"]), "the graph reached the Synthesizer and produced a final answer")

    # === 2. Destructive tool: interrupts BEFORE execution ===================
    print("\n=== 2. Destructive tool (create_issue) interrupts before executing ===")
    proposed_args = {"title": "Docker build fails: missing base image tag", "body": "Root cause: ..."}
    fake.next_tool_calls = [{"name": "create_issue", "args": dict(proposed_args), "id": "t2"}]
    fake.critic_verdicts = [{"verdict": "PASS", "reasoning": "ok", "missing_info": ""}]
    fake._critic_calls = 0

    thread_2 = _new_thread()
    result = mas.graph.invoke(mas._initial_state("File an issue about the Docker build failure"), config=_config(thread_2))

    check("__interrupt__" in result, "create_issue pauses the graph via a real interrupt()")
    interrupts = result["__interrupt__"]
    check(len(interrupts) >= 1, "at least one interrupt is present")
    payload = interrupts[0].value
    check(payload.get("tool") == "create_issue", "the interrupt payload names the correct tool")
    check(payload.get("args") == proposed_args, "the interrupt payload carries the exact proposed arguments")
    check(bool(payload.get("message")), "the interrupt payload includes a human-readable message")
    check(_read_issues() is None, "create_issue has NOT executed yet - data/issues.json does not exist")
    check(result.get("pending_tool_calls") == [{"name": "create_issue", "args": proposed_args, "id": "t2"}],
          "the pending tool call is checkpointed exactly as selected")

    # === 3. Approval: resumes the SAME thread and executes via MCP =========
    print("\n=== 3. Approving the paused create_issue executes it through MCP ===")
    result = mas.graph.invoke(Command(resume={"approved": True}), config=_config(thread_2))

    check("__interrupt__" not in result, "the graph fully resumes past the interrupt")
    check(any(r["tool_name"] == "create_issue" and r["success"] for r in result["tool_results"]),
          "create_issue is recorded as a successful tool_result after approval")
    approved_entry = [r for r in result["tool_results"] if r["tool_name"] == "create_issue"][-1]
    check(approved_entry.get("status") == "approved", "the tool_result is tagged status=approved")
    issues_text = _read_issues()
    check(issues_text is not None, "data/issues.json now exists")
    persisted = json.loads(issues_text)
    check(len(persisted) == 1, "exactly one issue was persisted")
    check(persisted[0]["title"] == proposed_args["title"], "the persisted issue has the approved title")
    check(bool(result.get("final_answer")), "the graph continued past tool_executor to completion")

    # === 4. Rejection: does not execute, does not crash =====================
    print("\n=== 4. Rejecting a paused create_issue does not execute it ===")
    reject_args = {"title": "Second issue attempt", "body": "..."}
    fake.next_tool_calls = [{"name": "create_issue", "args": dict(reject_args), "id": "t3"}]
    fake.critic_verdicts = [{"verdict": "PASS", "reasoning": "ok", "missing_info": ""}]
    fake._critic_calls = 0

    thread_3 = _new_thread()
    result = mas.graph.invoke(mas._initial_state("File an issue about the second problem"), config=_config(thread_3))
    check("__interrupt__" in result, "create_issue pauses the graph again for a fresh thread")

    issues_before_reject = _read_issues()
    result = mas.graph.invoke(
        Command(resume={"approved": False, "reason": "Not ready to file this yet."}),
        config=_config(thread_3),
    )

    check("__interrupt__" not in result, "the graph does not get stuck after a rejection")
    check(_read_issues() == issues_before_reject, "data/issues.json is unchanged after rejection")
    rejected_entries = [r for r in result["tool_results"] if r["tool_name"] == "create_issue" and r["args"] == reject_args]
    check(len(rejected_entries) == 1, "the rejected call is recorded exactly once")
    check(rejected_entries[0]["success"] is False, "the rejected call is recorded as unsuccessful")
    check(rejected_entries[0]["status"] == "rejected", "the rejected call is tagged status=rejected")
    check("Not ready to file this yet." in (rejected_entries[0]["error"] or ""),
          "the rejection reason is visible in the tool_result")
    check(bool(result.get("final_answer")), "the graph still reached the Synthesizer after a rejection (no crash)")

    # === Bonus: the identical rejected call is not immediately retried =====
    print("\n=== Bonus: an identical rejected call is not immediately re-proposed ===")
    fake.next_tool_calls = [{"name": "create_issue", "args": dict(reject_args), "id": "t3-repeat"}]
    fake.critic_verdicts = [
        {"verdict": "FAIL", "reasoning": "still need more evidence", "missing_info": "root cause"},
        {"verdict": "PASS", "reasoning": "ok now", "missing_info": ""},
    ]
    fake._critic_calls = 0
    thread_4 = _new_thread()
    # Seed this thread's history by rejecting the same (name, args) pair
    # once first, exactly like test 4 above, then let the retry loop run
    # without any further human input needed.
    result = mas.graph.invoke(mas._initial_state("File an issue about the second problem again"), config=_config(thread_4))
    check("__interrupt__" in result, "the first attempt still interrupts (nothing rejected yet on this thread)")
    result = mas.graph.invoke(Command(resume={"approved": False, "reason": "still no."}), config=_config(thread_4))
    # critic FAILs (first scripted verdict) -> router retries -> planner ->
    # tool_agent proposes the SAME create_issue/args again -> the
    # rejection-avoidance filter swaps it for search_documents BEFORE any
    # second interrupt happens.
    check("__interrupt__" not in result, "the repeated identical destructive call does not interrupt again")
    fallback_entries = [r for r in result["tool_results"] if r.get("tool_name") == "search_documents"]
    check(len(fallback_entries) >= 1, "the Tool Agent fell back to a read-only tool instead of repeating the rejected call")
    repeat_create_issue_calls = [r for r in result["tool_results"] if r["tool_name"] == "create_issue"]
    check(len(repeat_create_issue_calls) == 1, "create_issue was proposed only once on this thread, not repeated")

    # === 5. Edited approval: the EDITED args are what gets executed ========
    print("\n=== 5. Editing arguments during approval persists the edited values ===")
    original_args = {"title": "Original title", "body": "Original body"}
    edited_args = {"title": "Edited title (human-reviewed)", "body": "Edited body with more detail"}
    fake.next_tool_calls = [{"name": "create_issue", "args": dict(original_args), "id": "t5"}]
    fake.critic_verdicts = [{"verdict": "PASS", "reasoning": "ok", "missing_info": ""}]
    fake._critic_calls = 0

    thread_5 = _new_thread()
    result = mas.graph.invoke(mas._initial_state("File an issue, but I will edit it"), config=_config(thread_5))
    check("__interrupt__" in result, "create_issue interrupts before the edit is applied")
    check(result["__interrupt__"][0].value["args"] == original_args, "the human is shown the ORIGINAL proposed arguments")

    result = mas.graph.invoke(Command(resume={"approved": True, "args": edited_args}), config=_config(thread_5))
    check("__interrupt__" not in result, "the graph resumes after an edited approval")
    persisted = json.loads(_read_issues())
    edited_issue = [i for i in persisted if i["title"] == edited_args["title"]]
    check(len(edited_issue) == 1, "an issue with the EDITED title was persisted")
    check(edited_issue[0]["body"] == edited_args["body"], "the persisted issue has the EDITED body, not the original")
    check(not any(i["title"] == original_args["title"] for i in persisted),
          "the original (pre-edit) title was never persisted")

    # === 6. MCP tool failure becomes a structured result, not a crash ======
    print("\n=== 6. A tool-level MCP failure does not crash the graph ===")
    fake.next_tool_calls = [{"name": "this_tool_does_not_exist", "args": {"query": "x"}, "id": "t6"}]
    fake.critic_verdicts = [
        {"verdict": "FAIL", "reasoning": "no evidence", "missing_info": "everything"},
        {"verdict": "FAIL", "reasoning": "still no evidence", "missing_info": "everything"},
        {"verdict": "FAIL", "reasoning": "still no evidence", "missing_info": "everything"},
    ]
    fake._critic_calls = 0

    thread_6 = _new_thread()
    result = mas.graph.invoke(mas._initial_state("This will keep failing"), config=_config(thread_6))

    check("__interrupt__" not in result, "an unknown-tool failure does not pause the graph (it isn't destructive)")
    check(all(r["success"] is False for r in result["tool_results"]), "every attempt is recorded as a structured failure")
    check(bool(result.get("final_answer")), "the router still routes to the Synthesizer after MAX_RETRIES, no crash")

    print("\nALL HIL INTEGRATION CHECKS PASSED")


if __name__ == "__main__":
    try:
        main()
    finally:
        mas.mcp_client.get_manager().shutdown()
        # Leave no mock-issue side effects behind after the test run.
        if ISSUES_FILE.exists():
            ISSUES_FILE.unlink()
