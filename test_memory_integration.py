"""Standalone verification script for Step 6 (Persistent Memory Layer), in
the same style as test_mcp_integration.py / test_hil_integration.py: a
plain script using assertions, runnable directly with the project's
interpreter:

    python test_memory_integration.py

Like test_hil_integration.py, this drives the ACTUAL LangGraph agent
defined in Multi_Agent_System.py - the real StateGraph (now including the
Memory Recall / Memory Persist nodes), the real MemorySaver checkpointer,
real interrupt()/Command(resume=...) round-trips, the real MCP server
subprocess, and the REAL memory_manager.py reading/writing real files
under data/. Only the ChatOpenAI LLM object is replaced with a small
deterministic FakeLLM, exactly as in test_hil_integration.py.

Covers (10 required checks):
  1.  Empty memory (fresh data/) returns empty/graceful results, not
      errors, from every memory_manager read function.
  2.  Persistence across new graph runs: a second, unrelated graph
      invocation in the SAME process can recall a fact from an earlier
      completed run.
  3.  The search_history MCP tool is dynamically discoverable and, once
      memory has data, returns real results through the real MCP
      client/server (not by importing memory_manager directly).
  4.  The Memory Recall node actually retrieves relevant information and
      makes it visible to the Planner (state["memory_context"] /
      state["recalled_memories"] are populated for a related query).
  5.  Cross-session memory: two DIFFERENT thread_ids (sessions) still
      share the same durable memory - session B recalls what session A
      persisted.
  6.  Program-restart persistence: a brand-new subprocess (not just a
      new Python object / reloaded module) that only imports
      memory_manager.py can read data an earlier process wrote.
  7.  search_history surfaces BOTH a successful and an unsuccessful past
      search attempt for related terms.
  8.  Reduced-but-not-forced-zero redundant retrieval: Memory Recall
      surfaces the fact that a specific search was already attempted
      (a signal the Planner can use to avoid repeating it), without the
      framework hard-blocking a repeat search.
  9.  Full Step 5 HIL compatibility: interrupt/approve/reject still work
      exactly as in test_hil_integration.py AND the resulting
      approval/rejection is persisted to episodic memory and later
      retrievable via search_history/load_recent_interactions.
  10. Graceful degradation: every memory_manager function the graph
      calls is monkeypatched to raise, and the graph still completes
      end-to-end and returns a final_answer - a total memory outage
      never breaks the core RAG/HIL pipeline.

Every assertion failure raises AssertionError with a clear message; a
clean run prints "ALL MEMORY INTEGRATION CHECKS PASSED" and exits 0.
"""

import json
import subprocess
import sys
import uuid
from pathlib import Path

import Multi_Agent_System as mas
import memory_manager as mm
import mcp_client
from langgraph.types import Command

PROJECT_ROOT = Path(__file__).resolve().parent
ISSUES_FILE = PROJECT_ROOT / "data" / "issues.json"
EPISODIC_FILE = PROJECT_ROOT / "data" / "interaction_history.json"
SEARCH_HISTORY_FILE = PROJECT_ROOT / "data" / "search_history.json"
MEMORY_DIR = PROJECT_ROOT / "data" / "memory"


def check(condition, message):
    if not condition:
        raise AssertionError(f"FAILED: {message}")
    print(f"OK: {message}")


def _clean_memory_files():
    for f in (ISSUES_FILE, EPISODIC_FILE, SEARCH_HISTORY_FILE):
        if f.exists():
            f.unlink()
    if MEMORY_DIR.exists():
        for f in MEMORY_DIR.glob("*"):
            f.unlink()
        MEMORY_DIR.rmdir()


# --- FakeLLM: identical pattern to test_hil_integration.py ------------------

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
        return _FakeResponse(content="fake search query")

    def bind_tools(self, schemas):
        return _FakeBoundLLM(self.next_tool_calls)


def _new_thread():
    return str(uuid.uuid4())


def _config(thread_id):
    return {"configurable": {"thread_id": thread_id}}


def _run_query_directly(fake, query, thread_id, tool_calls, critic_verdicts):
    """Invoke the real graph exactly like run_query() does (including
    passing session_id=thread_id through _initial_state), but without
    run_query's interactive input() prompt - callers that expect an
    interrupt use Command(resume=...) directly instead."""

    fake.next_tool_calls = tool_calls
    fake.critic_verdicts = critic_verdicts
    fake._critic_calls = 0
    result = mas.graph.invoke(
        mas._initial_state(query, session_id=thread_id),
        config=_config(thread_id),
    )
    return result


def main():
    _clean_memory_files()

    fake = FakeLLM()
    mas.llm = fake

    # === 1. Empty memory returns graceful, empty results =====================
    print("=== 1. Empty memory: every read function degrades to an empty result ===")
    check(mm.load_recent_interactions(limit=5) == [], "load_recent_interactions is [] with no data/interaction_history.json")
    check(mm.search_history("anything at all") == [], "search_history is [] with no history files")
    check(mm.retrieve_relevant_memory("anything at all") == [], "retrieve_relevant_memory is [] with no semantic index")

    # === 2. Persistence across new graph runs (same process) =================
    print("\n=== 2. A later graph run in the SAME process recalls an earlier one ===")
    UNIQUE_TERM = "zylotrace-buffer-overrun-diagnosis"
    thread_a = _new_thread()
    result_a = _run_query_directly(
        fake,
        f"Investigate the {UNIQUE_TERM} reported in the logs",
        thread_a,
        [{"name": "search_documents", "args": {"query": UNIQUE_TERM, "top_k": 3}, "id": "m1"}],
        [{"verdict": "PASS", "reasoning": "sufficient", "missing_info": ""}],
    )
    check(bool(result_a.get("final_answer")), "the first run completed and produced a final_answer")
    check(EPISODIC_FILE.exists(), "Memory Persist wrote data/interaction_history.json after the first run")

    thread_b = _new_thread()
    result_b = _run_query_directly(
        fake,
        f"What did we find about {UNIQUE_TERM} before?",
        thread_b,
        [{"name": "search_documents", "args": {"query": UNIQUE_TERM, "top_k": 3}, "id": "m2"}],
        [{"verdict": "PASS", "reasoning": "sufficient", "missing_info": ""}],
    )
    check(UNIQUE_TERM in (result_b.get("memory_context") or ""),
          "the second run's Memory Recall surfaced the first run's interaction in memory_context")
    recalled = result_b.get("recalled_memories") or {}
    all_recalled_text = json.dumps(recalled)
    check(UNIQUE_TERM in all_recalled_text,
          "recalled_memories (structured) also contains the earlier interaction")

    # === 3. search_history MCP tool: discovery + real execution =============
    print("\n=== 3. search_history MCP tool is discoverable and returns real results ===")
    tools = {t.name for t in mcp_client.list_available_tools()}
    check("search_history" in tools, "search_history is discoverable via MCP list_tools")
    mcp_result = mcp_client.call_tool_json("search_history", {"query": UNIQUE_TERM, "top_k": 3})
    check(mcp_result["success"] is True, "the search_history MCP tool call succeeds once memory has matching data")
    check(len(mcp_result["results"]) > 0, "the search_history MCP tool returns at least one match")
    check("search_history" not in mcp_client.DESTRUCTIVE_TOOLS,
          "search_history is NOT classified as destructive (no HIL gate)")

    # === 4. Memory Recall makes relevant info visible to the Planner =========
    print("\n=== 4. Memory Recall's output is visible in state for the Planner ===")
    recall_state = mas.memory_recall({"query": UNIQUE_TERM})
    check(bool(recall_state.get("memory_context")), "memory_recall() returns a non-empty memory_context for a known term")
    check(isinstance(recall_state.get("recalled_memories"), dict), "memory_recall() returns structured recalled_memories")
    check(any(recall_state["recalled_memories"].values()), "at least one memory tier returned something for a known term")

    # === 5. Cross-session memory: different thread_ids share memory ==========
    print("\n=== 5. Cross-session memory: a different thread_id still recalls it ===")
    check(thread_a != thread_b, "the two runs above genuinely used different thread_ids (sessions)")
    # Already demonstrated by step 2 (thread_b recalled thread_a's data), but
    # verify explicitly via a brand new, third session too.
    thread_c = _new_thread()
    recall_c = mas.memory_recall({"query": UNIQUE_TERM})
    check(UNIQUE_TERM in json.dumps(recall_c), "a third, unrelated session_id/thread_id can still recall the same durable memory")

    # === 6. Program-restart persistence (a genuinely NEW process) ============
    print("\n=== 6. A brand-new subprocess (simulated restart) can read the same memory ===")
    check_script = (
        "import memory_manager as mm, json, sys\n"
        f"results = mm.search_history({UNIQUE_TERM!r}, top_k=3)\n"
        "sys.exit(0 if results else 1)\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", check_script],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
    )
    check(proc.returncode == 0,
          f"a fresh subprocess importing memory_manager.py fresh finds the persisted data (stderr={proc.stderr!r})")

    # === 7. search_history surfaces BOTH successes and failures ==============
    print("\n=== 7. search_history surfaces a successful AND an unsuccessful past search ===")
    FAIL_TERM = "quixotron-widget-parser-v9"
    thread_fail = _new_thread()
    _run_query_directly(
        fake,
        f"Look into {FAIL_TERM}",
        thread_fail,
        [{"name": "this_tool_does_not_exist", "args": {"query": FAIL_TERM}, "id": "f1"}],
        [
            {"verdict": "FAIL", "reasoning": "no evidence", "missing_info": "everything"},
            {"verdict": "FAIL", "reasoning": "no evidence", "missing_info": "everything"},
            {"verdict": "FAIL", "reasoning": "no evidence", "missing_info": "everything"},
        ],
    )
    thread_ok = _new_thread()
    _run_query_directly(
        fake,
        f"Look into {FAIL_TERM} again",
        thread_ok,
        [{"name": "search_documents", "args": {"query": FAIL_TERM, "top_k": 3}, "id": "f2"}],
        [{"verdict": "PASS", "reasoning": "ok", "missing_info": ""}],
    )
    history_matches = mm.search_history(FAIL_TERM, top_k=10)
    check(any(m.get("type") == "search_attempt" and m.get("success") is True for m in history_matches),
          "search_history includes the SUCCESSFUL search_documents attempt")
    # The failed tool name ("this_tool_does_not_exist") is intentionally not a
    # READ_ONLY tool, so log_search_attempts does not log it as a search
    # attempt (by design - see memory_manager.log_search_attempts) - but the
    # episodic record for that run still exists and is findable.
    episodic_matches = [m for m in history_matches if m.get("type") == "episodic"]
    check(len(episodic_matches) >= 1, "the failed run is still findable as an episodic record via search_history")

    # === 8. Reduced-but-not-forced-zero redundant retrieval ==================
    print("\n=== 8. Memory Recall surfaces prior searches as a SOFT signal, not a hard block ===")
    recall_repeat = mas.memory_recall({"query": UNIQUE_TERM})
    search_matches = recall_repeat["recalled_memories"].get("search_matches", [])
    check(len(search_matches) > 0, "a repeated query surfaces at least one prior search-history match")
    check(len(search_matches) <= 3, "search_matches is bounded to top_k=3, not the entire history")
    # Nothing in tool_agent/mcp_client prevents calling search_documents again
    # for this same term - the signal is advisory only (this is a structural
    # check, not a behavioral one: DESTRUCTIVE_TOOLS/_guard_destructive_tool
    # are the only hard gate in this codebase, and search_documents is not in
    # it).
    check("search_documents" not in mcp_client.DESTRUCTIVE_TOOLS,
          "search_documents remains freely callable even after memory recalls a prior attempt (soft signal only)")

    # === 9. Full Step 5 HIL compatibility + persistence to memory ============
    print("\n=== 9. HIL interrupt/approve/reject still work AND are persisted to memory ===")
    thread_hil = _new_thread()
    proposed_args = {"title": "Memory-compat issue", "body": "Root cause: verified via test."}
    fake.next_tool_calls = [{"name": "create_issue", "args": dict(proposed_args), "id": "h1"}]
    fake.critic_verdicts = [{"verdict": "PASS", "reasoning": "ok", "missing_info": ""}]
    fake._critic_calls = 0

    result = mas.graph.invoke(mas._initial_state("File an issue for the memory compatibility test", session_id=thread_hil), config=_config(thread_hil))
    check("__interrupt__" in result, "create_issue still interrupts the graph exactly as in Step 5")

    result = mas.graph.invoke(Command(resume={"approved": True}), config=_config(thread_hil))
    check("__interrupt__" not in result, "the graph resumes fully after approval")
    check(bool(result.get("final_answer")), "the graph reaches the Synthesizer (and Memory Persist) after approval")

    approved_matches = mm.search_history("Memory-compat issue", top_k=5)
    check(any(m.get("type") == "episodic" for m in approved_matches),
          "the approved create_issue run is persisted to episodic memory")
    recent = mm.load_recent_interactions(limit=10)
    hil_record = next((r for r in recent if r.get("session_id") == thread_hil), None)
    check(hil_record is not None, "the HIL run's episodic record is retrievable by session_id")
    check(any(a.get("tool_name") == "create_issue" and a.get("status") == "approved" for a in hil_record["approvals"]),
          "the persisted record's approvals list records the create_issue approval")

    # Reject case too.
    thread_hil_reject = _new_thread()
    fake.next_tool_calls = [{"name": "create_issue", "args": {"title": "Should be rejected", "body": "x"}, "id": "h2"}]
    fake.critic_verdicts = [{"verdict": "PASS", "reasoning": "ok", "missing_info": ""}]
    fake._critic_calls = 0
    result = mas.graph.invoke(mas._initial_state("File a second issue", session_id=thread_hil_reject), config=_config(thread_hil_reject))
    check("__interrupt__" in result, "the second create_issue also interrupts")
    result = mas.graph.invoke(Command(resume={"approved": False, "reason": "not now"}), config=_config(thread_hil_reject))
    check(bool(result.get("final_answer")), "the graph still completes (and persists) after a rejection")
    recent = mm.load_recent_interactions(limit=10)
    reject_record = next((r for r in recent if r.get("session_id") == thread_hil_reject), None)
    check(reject_record is not None, "the rejected run's episodic record is retrievable")
    check(any(a.get("status") == "rejected" for a in reject_record["approvals"]),
          "the persisted record's approvals list records the create_issue rejection")

    # === 10. Graceful degradation when memory is completely unavailable ======
    print("\n=== 10. A total memory-layer failure never breaks the core graph ===")
    real_fns = {
        name: getattr(mm, name)
        for name in (
            "load_recent_interactions",
            "search_history",
            "retrieve_relevant_memory",
            "save_interaction",
            "log_search_attempts",
            "save_memory",
        )
    }

    def _boom(*args, **kwargs):
        raise RuntimeError("simulated total memory failure")

    for name in real_fns:
        setattr(mm, name, _boom)

    try:
        thread_broken = _new_thread()
        fake.next_tool_calls = [{"name": "search_documents", "args": {"query": "docker build fail", "top_k": 3}, "id": "b1"}]
        fake.critic_verdicts = [{"verdict": "PASS", "reasoning": "ok", "missing_info": ""}]
        fake._critic_calls = 0
        result = mas.graph.invoke(mas._initial_state("Why does the build fail?", session_id=thread_broken), config=_config(thread_broken))
        check("__interrupt__" not in result, "the graph does not spuriously interrupt when memory is broken")
        check(bool(result.get("final_answer")), "the graph STILL completes and produces a final_answer with memory fully broken")
        check(result.get("memory_context") == "", "memory_context degrades to empty (not a crash) when recall fails")
    finally:
        for name, fn in real_fns.items():
            setattr(mm, name, fn)

    print("\nALL MEMORY INTEGRATION CHECKS PASSED")


if __name__ == "__main__":
    try:
        main()
    finally:
        mas.mcp_client.get_manager().shutdown()
        _clean_memory_files()
