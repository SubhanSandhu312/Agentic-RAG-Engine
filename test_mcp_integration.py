"""Focused, standalone verification script for the MCP retrieval layer.

The project has no existing test suite (no pytest, no tests/ directory), so
this follows the same convention: a plain script using assertions, runnable
directly with the project's own interpreter/venv:

    python test_mcp_integration.py

It exercises the MCP server (mcp_server.py) through the real MCP client
(mcp_client.py) over stdio - i.e. it launches the server as a subprocess and
talks MCP to it, exactly like the LangGraph agent does. It does NOT import
mcp_server.py's tool functions directly.

Covers:
  - dynamic tool discovery (list_tools)
  - a real search_documents call
  - graceful handling of an invalid/empty query
  - retrieve_file for an existing file, a missing file, and a path-traversal
    attempt
  - search_code
  - an unknown tool name
  - the MCP-server-unreachable path (MCPServerError)
  - MCP tool schemas -> OpenAI/LLM function-calling format (get_llm_tool_schemas)
  - Pydantic validation at the client boundary rejecting a malformed payload
  - automatic respawn-and-retry after a simulated broken connection
  - the destructive-tool interception pre-hook (call_tool_json never lets a
    DESTRUCTIVE_TOOLS name reach the JSON-RPC call)

Every assertion failure raises AssertionError with a clear message; a clean
run prints "ALL MCP INTEGRATION CHECKS PASSED" and exits 0.
"""

import sys

import mcp_client


def check(condition, message):
    if not condition:
        raise AssertionError(f"FAILED: {message}")
    print(f"OK: {message}")


def main():
    print("=== 1. Dynamic tool discovery ===")
    tools = {t.name for t in mcp_client.list_available_tools()}
    check("search_documents" in tools, "search_documents is discoverable via MCP list_tools")
    check("retrieve_file" in tools, "retrieve_file is discoverable via MCP list_tools")
    check("create_issue" in tools, "create_issue (Step 5 destructive tool) is discoverable via MCP list_tools")
    check("search_history" in tools, "search_history (Step 6 memory tool) is discoverable via MCP list_tools")

    print("\n=== 2. search_documents: real query ===")
    result = mcp_client.call_tool_json(
        "search_documents",
        {"query": "docker build fail dependency", "top_k": 3},
    )
    check(isinstance(result, dict) and "success" in result, "search_documents returns a structured dict")
    check(result["success"] is True, "search_documents succeeds for a real query")
    check(len(result["results"]) > 0, "search_documents returns at least one ranked chunk")
    check(all({"file", "text", "score"} <= item.keys() for item in result["results"]),
          "each result has file/text/score fields")

    print("\n=== 3. search_documents: invalid (empty) query ===")
    result = mcp_client.call_tool_json("search_documents", {"query": "", "top_k": 3})
    check(result["success"] is False, "empty query fails gracefully (no exception raised)")
    check(result["results"] == [], "empty query returns no results")
    check(bool(result["error"]), "empty query includes an error message")

    print("\n=== 4. retrieve_file: existing file ===")
    result = mcp_client.call_tool_json("retrieve_file", {"path": "broken-pipeline-lab/Dockerfile"})
    check(result["success"] is True, "retrieve_file succeeds for an existing corpus file")
    check(bool(result["content"]), "retrieve_file returns non-empty content")

    print("\n=== 5. retrieve_file: missing file ===")
    result = mcp_client.call_tool_json("retrieve_file", {"path": "does/not/exist.py"})
    check(result["success"] is False, "retrieve_file fails gracefully for a missing file")
    check(result["content"] is None, "retrieve_file returns no content for a missing file")

    print("\n=== 6. retrieve_file: path traversal is blocked ===")
    result = mcp_client.call_tool_json("retrieve_file", {"path": "../../../etc/passwd"})
    check(result["success"] is False, "path traversal outside data/ is rejected")

    print("\n=== 7. search_code ===")
    result = mcp_client.call_tool_json("search_code", {"query": "pytest", "top_k": 3})
    check(isinstance(result, dict) and "success" in result, "search_code returns a structured dict")

    print("\n=== 8. Unknown tool name ===")
    result = mcp_client.call_tool_json("this_tool_does_not_exist", {})
    check(result["success"] is False, "calling an unknown tool fails gracefully, not with a crash")

    print("\n=== 9. MCP server unreachable ===")
    original_script = mcp_client.SERVER_SCRIPT
    mcp_client.SERVER_SCRIPT = "/nonexistent/mcp_server.py"
    manager = mcp_client._MCPClientManager()
    try:
        manager.list_tools()
        raise AssertionError("FAILED: expected MCPServerError for an unreachable server")
    except mcp_client.MCPServerError:
        print("OK: an unreachable MCP server raises MCPServerError instead of hanging/crashing silently")
    finally:
        mcp_client.SERVER_SCRIPT = original_script
        manager.shutdown()

    print("\n=== 10. MCP tool schemas convert to OpenAI/LLM function-calling format ===")
    schemas = mcp_client.get_llm_tool_schemas()
    names = {s["function"]["name"] for s in schemas}
    check(names == {"search_documents", "retrieve_file", "search_code", "create_issue", "search_history"},
          "get_llm_tool_schemas() returns exactly the 5 tools the server exposes")
    check(all(s["type"] == "function" and "parameters" in s["function"] for s in schemas),
          "every schema has the {type: function, function: {..., parameters}} shape bind_tools() expects")

    print("\n=== 11. Pydantic validation rejects a malformed payload at the client boundary ===")
    malformed = {"results": ["missing the required 'success' field"]}
    validated = mcp_client._validate_tool_response("search_documents", malformed)
    check(validated["success"] is False, "a payload missing 'success' is rejected, not passed through")
    check("Subprocess transport error" in validated["error"], "the rejection is labeled as a transport error")

    print("\n=== 12. Automatic respawn-and-retry after a broken connection ===")
    mcp_client.call_tool_json("search_documents", {"query": "docker", "top_k": 1})  # ensure started
    live_manager = mcp_client.get_manager()
    session_before = live_manager._session
    real_call_tool = session_before.call_tool
    flaky_state = {"calls": 0}

    async def _flaky_call_tool(name, arguments):
        flaky_state["calls"] += 1
        if flaky_state["calls"] == 1:
            raise ConnectionResetError("simulated broken pipe")
        return await real_call_tool(name, arguments)

    live_manager._session.call_tool = _flaky_call_tool
    respawned_result = mcp_client.call_tool_json("search_documents", {"query": "docker", "top_k": 1})
    check(respawned_result["success"] is True, "a call succeeds transparently after one auto-respawn")
    check(live_manager._session is not session_before, "a broken connection gets a brand-new session, not a reused dead one")

    print("\n=== 13. Destructive-tool interception pre-hook ===")
    try:
        mcp_client.call_tool_json("modify_file", {"path": "anything"})
        raise AssertionError("FAILED: expected DestructiveToolBlockedError for a DESTRUCTIVE_TOOLS name")
    except mcp_client.DestructiveToolBlockedError:
        print("OK: a DESTRUCTIVE_TOOLS-listed tool is intercepted before any JSON-RPC call is made")

    print("\n=== 14. create_issue is blocked outside a real graph context ===")
    import json as _json
    from pathlib import Path as _Path
    issues_file = _Path(__file__).resolve().parent / "data" / "issues.json"
    issues_before = issues_file.read_text(encoding="utf-8") if issues_file.exists() else None
    try:
        mcp_client.call_tool_json("create_issue", {"title": "should not be created"})
        raise AssertionError("FAILED: expected DestructiveToolBlockedError for create_issue outside a graph")
    except mcp_client.DestructiveToolBlockedError:
        print("OK: create_issue is intercepted before any JSON-RPC call when there is no checkpointer/thread")
    issues_after = issues_file.read_text(encoding="utf-8") if issues_file.exists() else None
    check(issues_before == issues_after, "data/issues.json was not modified by the blocked create_issue call")

    print("\n=== 15. search_history: empty memory returns a graceful failure ===")
    result = mcp_client.call_tool_json("search_history", {"query": "nonexistent query xyz123", "top_k": 3})
    check(isinstance(result, dict) and "success" in result, "search_history returns a structured dict")
    check(result["success"] is False, "search_history reports no results gracefully when nothing matches")
    check(result["results"] == [], "search_history returns an empty list, not an error, for no matches")

    print("\n=== 16. search_history: invalid (empty) query ===")
    result = mcp_client.call_tool_json("search_history", {"query": "", "top_k": 3})
    check(result["success"] is False, "empty query fails gracefully (no exception raised)")

    print("\nALL MCP INTEGRATION CHECKS PASSED")


if __name__ == "__main__":
    try:
        main()
    finally:
        mcp_client.get_manager().shutdown()
