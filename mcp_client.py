"""MCP client layer used by the LangGraph agent (Multi_Agent_System.py).

This module is the ONLY place in the LangGraph process that knows how to
talk to the retrieval MCP server (mcp_server.py). It launches that server
as a subprocess over stdio using the official `mcp` Python SDK's client
APIs, discovers its tools dynamically (list_tools), and exposes:

  - get_llm_tool_schemas() - converts the MCP tool definitions into
    OpenAI-style function-calling schemas so the Tool Agent node can bind
    them to the LLM with `llm.bind_tools(...)` and let the model choose
    which tool(s) to call, instead of the graph hardcoding a tool name.
  - call_tool_json(name, arguments) - dynamically dispatches a tool call
    by name (whatever the model - or a fallback - chose) and returns a
    validated, JSON-serializable dict.

The LangGraph node functions are synchronous, while the MCP SDK's client
session is async. To avoid re-launching (and re-loading the embedding /
cross-encoder models inside) the server subprocess on every single
tool_agent call/retry, a single MCP ClientSession is started once, on a
dedicated background event loop thread, and reused for the lifetime of the
process. If that subprocess/session dies mid-run, exactly one automatic
respawn+reinitialize is attempted before an MCPServerError is raised to
the caller (see _MCPClientManager._run).

Nothing here imports mcp_server.py as a Python module - the connection is
real MCP-over-stdio, started via `sys.executable mcp_server.py`.
"""

import asyncio
import atexit
import json
import sys
import threading
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Optional

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from pydantic import BaseModel, ConfigDict, ValidationError

SERVER_SCRIPT = str(Path(__file__).resolve().parent / "mcp_server.py")

# --- Tool safety classification (Step 5: Human-in-the-Loop) ----------------
#
# search_documents/retrieve_file/search_code are read-only and always
# execute automatically. create_issue is a REAL destructive/write tool
# (mcp_server.py persists it to data/issues.json) and requires human
# authorization before it executes - see `_guard_destructive_tool`, called
# from `call_tool_json` before any JSON-RPC request is sent. modify_file
# and run_sql_mutation remain placeholders (no such tool exists on the
# server) kept only so the classification set doesn't need to change again
# if either is implemented later; calling them is blocked the same way a
# genuinely destructive tool would be, they simply never reach the server.
READ_ONLY_TOOLS = {"search_documents", "retrieve_file", "search_code", "search_history"}
# search_history (Step 6) queries the persistent memory layer (memory_manager.py) -
# it never mutates anything, so it is classified read-only exactly like the other
# 3 retrieval tools and is NOT added to DESTRUCTIVE_TOOLS below.
DESTRUCTIVE_TOOLS = {"create_issue", "modify_file", "run_sql_mutation"}


class MCPServerError(RuntimeError):
    """Raised when the MCP server subprocess can't be started or connected
    to, including after one automatic respawn attempt has already failed."""


class DestructiveToolBlockedError(RuntimeError):
    """Raised when a call to a DESTRUCTIVE_TOOLS-listed tool is intercepted
    before it reaches the MCP server.

    Inside an actual LangGraph run with a checkpointer configured (Step 5),
    the interception instead surfaces as `langgraph.errors.GraphInterrupt`
    (raised by `langgraph.types.interrupt`), which pauses the graph and
    waits for a human decision via `Command(resume=...)` - that exception
    is intentionally NOT caught here and propagates unchanged. This error
    only fires when `interrupt()` is called outside a runnable graph
    context (no checkpointer/thread yet), so destructive tools still fail
    safely instead of silently executing.
    """


class DestructiveActionRejectedError(RuntimeError):
    """Raised when a human reviewer REJECTS a DESTRUCTIVE_TOOLS call after
    a real interrupt()/Command(resume=...) round-trip (Step 5).

    Distinct from DestructiveToolBlockedError: this means a human was
    actually asked and said no (or the resume payload was malformed), not
    that no human was available to ask at all. `tool_agent`/`tool_executor`
    catch this and record a structured rejection in `tool_results` instead
    of letting it crash the graph.
    """

    def __init__(self, tool_name, reason):
        self.tool_name = tool_name
        self.reason = reason
        super().__init__(f"human reviewer rejected '{tool_name}': {reason}")


class _MCPToolResponse(BaseModel):
    """Minimal schema every MCP tool response must satisfy.

    Only `success` is required - the three current tools return different
    extra shapes (`results` for search_documents/search_code, `path`/
    `content` for retrieve_file), so `extra="allow"` lets those through
    untouched. This exists purely to catch a malformed or truncated
    payload crossing the client boundary (a corrupted stdout write, a
    partially-written subprocess response, stray stderr text that leaked
    into stdout, etc.) before it reaches a LangGraph node as an unexpected
    shape.
    """

    model_config = ConfigDict(extra="allow")

    success: bool
    error: Optional[str] = None


def _validate_tool_response(name, data):
    """Validate a tool's parsed JSON payload at the client boundary.

    Returns the validated dict on success, or a structured
    {"success": False, ...} error dict (never raises) if `data` doesn't
    even satisfy the minimal contract every tool response must have.
    """

    try:
        validated = _MCPToolResponse.model_validate(data)
    except ValidationError as e:
        return {
            "success": False,
            "results": [],
            "error": f"Subprocess transport error: malformed response from '{name}': {e}",
        }
    return validated.model_dump()


def _guard_destructive_tool(name, arguments):
    """Interception pre-hook: block DESTRUCTIVE_TOOLS before the JSON-RPC
    call is made, and gate them behind a real human decision.

    Returns the arguments that should actually be executed (identical to
    `arguments` unless the human edited them during approval). Raises
    instead of returning when the tool must not execute:

      - langgraph.errors.GraphInterrupt: first pass through a real,
        checkpointed graph run - this PAUSES the graph and is NOT caught
        here; it propagates unchanged and is the human-in-the-loop gate
        itself. LangGraph re-invokes the calling node on resume, at which
        point this same interrupt() call returns the resume payload
        instead of raising again.
      - DestructiveActionRejectedError: a human reviewer was actually
        asked (a real resume happened) and said no, or the resume payload
        was malformed/missing "approved".
      - DestructiveToolBlockedError: interrupt() could not even be posed
        (e.g. called outside a runnable/checkpointed graph, such as a
        plain script or a unit test) - fails safe rather than executing.
    """

    if name not in DESTRUCTIVE_TOOLS:
        return arguments

    payload = {
        "tool": name,
        "args": arguments,
        "message": (
            f"Human approval required to execute the destructive tool "
            f"'{name}'."
        ),
    }

    try:
        from langgraph.errors import GraphInterrupt
        from langgraph.types import interrupt as lg_interrupt
    except ImportError as e:
        raise DestructiveToolBlockedError(
            f"blocked destructive tool '{name}': langgraph interrupt() unavailable: {e}"
        ) from e

    try:
        # Inside a real graph run with a checkpointer (Step 5), this raises
        # GraphInterrupt on the first pass (pausing the graph) and RETURNS
        # the value passed to Command(resume=...) once the driver resumes
        # the same thread.
        decision = lg_interrupt(payload)
    except GraphInterrupt:
        raise
    except Exception as e:
        # interrupt() was called outside a runnable graph context (no
        # checkpointer/thread configured yet, e.g. a plain script or this
        # module's own tests) - fail safe rather than executing the tool.
        raise DestructiveToolBlockedError(
            f"blocked destructive tool '{name}' pre-execution: {e}"
        ) from e

    if not isinstance(decision, dict):
        raise DestructiveToolBlockedError(
            f"blocked destructive tool '{name}': resume payload must be a "
            f"dict with an 'approved' key, got {type(decision).__name__}"
        )

    if not decision.get("approved"):
        reason = decision.get("reason") or "rejected by human reviewer"
        raise DestructiveActionRejectedError(name, reason)

    # Approved - if the human edited the arguments during review, THOSE
    # are what get executed, not the original proposal.
    edited_args = decision.get("args")
    if isinstance(edited_args, dict) and edited_args:
        return edited_args
    return arguments


class _MCPClientManager:
    """Owns one long-lived MCP ClientSession connected to mcp_server.py.

    Runs its own asyncio event loop on a background thread so that plain,
    synchronous LangGraph node functions can call `call_tool()` /
    `list_tools()` like ordinary blocking function calls. If the
    connection/subprocess turns out to be dead, `_run` respawns it exactly
    once and retries before giving up.
    """

    def __init__(self):
        self._loop = None
        self._thread = None
        self._session = None
        self._stack = None
        self._tools_cache = None
        self._lock = threading.Lock()

    def _ensure_started(self):
        with self._lock:
            if self._session is not None:
                return

            if self._loop is None:
                self._loop = asyncio.new_event_loop()
                self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)
                self._thread.start()

            future = asyncio.run_coroutine_threadsafe(self._connect(), self._loop)
            try:
                future.result(timeout=120)
            except Exception as e:
                self._session = None
                raise MCPServerError(f"failed to start/connect to MCP server: {e}") from e

    async def _connect(self):
        server_params = StdioServerParameters(
            command=sys.executable,
            args=[SERVER_SCRIPT],
            # retriver.py (existing code) walks a relative "data" path, so
            # the server subprocess must run with the project root as its
            # working directory regardless of where the client process was
            # launched from.
            cwd=str(Path(__file__).resolve().parent),
        )
        self._stack = AsyncExitStack()
        read_stream, write_stream = await self._stack.enter_async_context(
            stdio_client(server_params)
        )
        session = await self._stack.enter_async_context(
            ClientSession(read_stream, write_stream)
        )
        await session.initialize()
        self._session = session

    def _reset_connection(self):
        """Best-effort teardown of a dead/broken connection, clearing state
        so the next `_ensure_started()` launches a fresh server subprocess
        and re-runs the MCP initialize handshake."""

        if self._stack is not None and self._loop is not None and self._loop.is_running():
            future = asyncio.run_coroutine_threadsafe(self._stack.aclose(), self._loop)
            try:
                future.result(timeout=10)
            except Exception:
                pass
        self._session = None
        self._stack = None
        self._tools_cache = None

    def _run(self, coro_fn, timeout=60, _respawned=False):
        """Run `coro_fn()` (a zero-arg callable returning a coroutine) on
        the manager's event loop, starting the connection first if needed.

        Takes a factory rather than an already-built coroutine so that the
        coroutine (which references self._session) is only constructed
        AFTER _ensure_started() has set self._session.

        Context preservation on subprocess restarts: if the underlying
        server subprocess/session has died (broken pipe, closed stream,
        crashed process), this catches the failure, tears down and
        respawns the connection exactly once, and retries the same call
        before surfacing MCPServerError to the caller.
        """

        self._ensure_started()

        try:
            future = asyncio.run_coroutine_threadsafe(coro_fn(), self._loop)
            return future.result(timeout=timeout)
        except Exception as e:
            if _respawned:
                raise MCPServerError(
                    f"MCP server connection failed even after an automatic restart: {e}"
                ) from e

            with self._lock:
                self._reset_connection()

            try:
                return self._run(coro_fn, timeout=timeout, _respawned=True)
            except MCPServerError:
                raise
            except Exception as e2:
                raise MCPServerError(
                    f"failed to respawn MCP server after connection loss: {e2}"
                ) from e2

    def list_tools(self):
        """Dynamically discover the tools the MCP server currently exposes."""
        if self._tools_cache is None:
            result = self._run(lambda: self._session.list_tools())
            self._tools_cache = result.tools
        return self._tools_cache

    def call_tool(self, name, arguments):
        return self._run(lambda: self._session.call_tool(name, arguments))

    def shutdown(self):
        if self._loop is None:
            return
        if self._stack is not None:
            future = asyncio.run_coroutine_threadsafe(self._stack.aclose(), self._loop)
            try:
                future.result(timeout=10)
            except Exception:
                pass
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=5)
        self._session = None
        self._loop = None
        self._thread = None
        self._tools_cache = None


_manager = _MCPClientManager()


def get_manager():
    """Return the process-wide MCP client manager (starts it on first use)."""
    return _manager


def list_available_tools():
    """Return the list of MCP Tool objects currently exposed by the server."""
    return _manager.list_tools()


def get_llm_tool_schemas():
    """Convert the dynamically-discovered MCP tool definitions into
    OpenAI-style function-calling tool schemas.

    The result can be passed straight to `llm.bind_tools(...)` (LangChain
    accepts this exact `{"type": "function", "function": {...}}` shape),
    which is how the Tool Agent node lets the model choose between
    search_documents / retrieve_file / search_code dynamically, rather
    than the graph hardcoding which one to call.
    """

    schemas = []
    for tool in list_available_tools():
        schemas.append({
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description or "",
                "parameters": tool.inputSchema or {"type": "object", "properties": {}},
            },
        })
    return schemas


def _extract_text(result):
    parts = []
    for item in getattr(result, "content", []) or []:
        if getattr(item, "type", None) == "text":
            parts.append(item.text)
    return "\n".join(parts)


def call_tool_json(name, arguments):
    """Dynamically dispatch an MCP tool call by name and return its result
    as a plain, schema-validated dict.

    This is the single dispatch point every tool call goes through -
    callers pass whatever tool name/arguments the LLM (or a fallback)
    chose; nothing here branches on `if name == "search_documents"` etc.

    Never raises for ordinary tool-level failures (bad args, no results,
    tool-reported errors, malformed/truncated responses) - those come back
    as {"success": False, "results": [], "error": "..."} so LangGraph
    nodes can hand them to the Critic/Planner loop instead of crashing the
    graph.

    Raises:
        DestructiveToolBlockedError / langgraph.errors.GraphInterrupt /
            DestructiveActionRejectedError: if `name` is in
            DESTRUCTIVE_TOOLS (see _guard_destructive_tool).
        MCPServerError: if the server process itself could not be
            started/reached, even after one automatic respawn attempt.
    """

    # May return edited arguments (human-in-the-loop "edit" approval) - the
    # EDITED arguments are what actually get executed below, never the
    # original proposal silently.
    arguments = _guard_destructive_tool(name, arguments)

    manager = get_manager()

    try:
        result = manager.call_tool(name, arguments)
    except MCPServerError:
        raise
    except Exception as e:
        return {"success": False, "results": [], "error": f"MCP tool call to '{name}' failed: {e}"}

    if getattr(result, "isError", False):
        text = _extract_text(result) or f"tool '{name}' reported an error"
        return {"success": False, "results": [], "error": text}

    structured = getattr(result, "structuredContent", None)
    if isinstance(structured, dict):
        return _validate_tool_response(name, structured)

    text = _extract_text(result)
    if text:
        try:
            parsed = json.loads(text)
        except (json.JSONDecodeError, TypeError) as e:
            return {
                "success": False,
                "results": [],
                "error": f"Subprocess transport error: could not parse response from '{name}': {e}",
            }
        if not isinstance(parsed, dict):
            return {
                "success": False,
                "results": [],
                "error": f"Subprocess transport error: unexpected response shape from '{name}'",
            }
        return _validate_tool_response(name, parsed)

    return {"success": False, "results": [], "error": f"empty response from tool '{name}'"}


# Make sure the MCP server subprocess is terminated cleanly when the
# LangGraph process exits, instead of being left as an orphan.
atexit.register(_manager.shutdown)
