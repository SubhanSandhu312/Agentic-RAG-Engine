#!/usr/bin/env python
"""Standalone MCP server for the Agentic RAG Engine.

This process exposes the project's EXISTING retrieval pipeline (BM25 +
FAISS + Reciprocal Rank Fusion + cross-encoder reranking, all implemented
in Hybrid_Search.py / Faiss_Searach.py / embeddings.py / llm_calling.py) as
Model Context Protocol (MCP) tools, using the official `mcp` Python SDK.

It does NOT reimplement any retrieval algorithm - each tool below is a
thin adapter that calls the existing pipeline and returns a structured,
JSON-serializable result.

Run it directly for manual testing:

    python mcp_server.py

Talking to it manually (or via any MCP client) uses stdio: the client
launches this file as a subprocess and speaks MCP over stdin/stdout. The
LangGraph agent (Multi_Agent_System.py) never imports this module - it
only talks to it through mcp_client.py over the MCP protocol.
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

# Make sure the project root is importable regardless of the working
# directory the server happens to be launched from.
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from mcp.server.fastmcp import FastMCP

from llm_calling import search_documents_structured

# Step 6: persistent memory layer. Imported defensively - if memory_manager
# is missing or fails to import for any reason, the search_history tool
# degrades to a structured error instead of crashing the whole server (the
# 4 pre-existing retrieval/action tools must keep working either way).
try:
    from memory_manager import search_history as _memory_search_history
except Exception as _memory_import_error:
    _memory_search_history = None

DATA_DIR = (PROJECT_ROOT / "data").resolve()

# Local mock "issue tracker" storage for the create_issue action tool
# (Step 5). This never talks to GitHub or any external service - it is
# a safe local stand-in used to demonstrate a destructive/write MCP
# tool gated by human approval.
ISSUES_FILE = (DATA_DIR / "issues.json").resolve()

# Extensions treated as "code/config" for the search_code tool. Anything
# else indexed under data/ (docs, txt, etc.) is left to search_documents.
CODE_EXTENSIONS = {
    ".py", ".yml", ".yaml", ".sh", ".json", ".cfg", ".ini",
    ".toml", ".dockerfile", ".txt",
}

mcp = FastMCP("agentic-rag-retrieval")


@mcp.tool()
def search_documents(query: str, top_k: int = 5) -> dict:
    """Search the indexed document corpus (data/) using the project's
    existing hybrid retrieval pipeline: BM25 + FAISS candidates fused with
    Reciprocal Rank Fusion, then reranked with a cross-encoder.

    Args:
        query: Natural-language or keyword search query.
        top_k: Maximum number of ranked chunks to return (default 5).

    Returns:
        A dict: {"success": bool, "results": [{"file": str, "text": str,
        "score": float}], "error": str | None}. success is False and
        results is empty when the query is invalid, nothing relevant was
        found, or retrieval failed - it never raises for those cases.
    """

    if not isinstance(query, str) or not query.strip():
        return {"success": False, "results": [], "error": "query must be a non-empty string"}

    try:
        top_k = int(top_k)
    except (TypeError, ValueError):
        return {"success": False, "results": [], "error": "top_k must be an integer"}

    if top_k <= 0:
        return {"success": False, "results": [], "error": "top_k must be a positive integer"}

    try:
        return search_documents_structured(query, top_k=top_k)
    except Exception as e:
        return {"success": False, "results": [], "error": f"unexpected error: {e}"}


def _resolve_within_data(relative_path: str) -> Path:
    """Resolve `relative_path` against data/, refusing anything that would
    escape the corpus directory (e.g. via `..` or an absolute path).

    `search_documents` reports file paths as e.g. "data/foo/bar.py" (since
    the existing corpus walker in retriver.py starts from Path("data")), so
    a leading "data/" (or "data\\") is stripped if present, letting callers
    pass either the exact string search_documents returned or a path
    relative to the data/ directory itself.
    """

    normalized = relative_path.replace("\\", "/")
    if normalized == "data" or normalized.startswith("data/"):
        normalized = normalized[len("data"):].lstrip("/")

    candidate = (DATA_DIR / normalized).resolve()

    if candidate != DATA_DIR and DATA_DIR not in candidate.parents:
        raise ValueError("path escapes the data/ corpus directory")

    return candidate


@mcp.tool()
def retrieve_file(path: str) -> dict:
    """Retrieve the full contents of a specific file from the indexed
    data/ corpus. Paths are resolved relative to data/ and cannot escape
    it, so this cannot be used to read arbitrary files on the filesystem.

    Args:
        path: File path relative to the data/ directory, e.g.
            "broken-pipeline-lab/app/main.py".

    Returns:
        {"success": bool, "path": str, "content": str | None, "error": str | None}
    """

    if not isinstance(path, str) or not path.strip():
        return {"success": False, "path": path, "content": None, "error": "path must be a non-empty string"}

    try:
        resolved = _resolve_within_data(path)
    except ValueError as e:
        return {"success": False, "path": path, "content": None, "error": str(e)}
    except Exception as e:
        return {"success": False, "path": path, "content": None, "error": f"invalid path: {e}"}

    if not resolved.exists():
        return {"success": False, "path": path, "content": None, "error": f"file not found: {path}"}

    if not resolved.is_file():
        return {"success": False, "path": path, "content": None, "error": f"not a file: {path}"}

    try:
        content = resolved.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        return {"success": False, "path": path, "content": None, "error": f"failed to read file: {e}"}

    return {"success": True, "path": path, "content": content, "error": None}


@mcp.tool()
def search_code(query: str, top_k: int = 5) -> dict:
    """Keyword search over code/config files in the data/ corpus using the
    project's existing BM25 index directly (no embedding/reranking pass).
    Useful for exact identifier, filename, or config-key lookups where
    lexical matching beats semantic similarity.

    Args:
        query: Keyword(s) to search for (e.g. a function name, YAML key,
            or error string).
        top_k: Maximum number of matches to return (default 5).

    Returns:
        {"success": bool, "results": [{"file": str, "text": str}], "error": str | None}
    """

    if not isinstance(query, str) or not query.strip():
        return {"success": False, "results": [], "error": "query must be a non-empty string"}

    try:
        top_k = int(top_k)
    except (TypeError, ValueError):
        return {"success": False, "results": [], "error": "top_k must be an integer"}

    if top_k <= 0:
        return {"success": False, "results": [], "error": "top_k must be a positive integer"}

    # Imported lazily so importing mcp_server.py doesn't force-build the
    # BM25 index for callers that only need search_documents/retrieve_file.
    from Hybrid_Search import query_bm25
    from chunking import chunks

    try:
        candidate_ids = query_bm25(query, top_k=max(top_k * 4, 20))
    except Exception as e:
        return {"success": False, "results": [], "error": f"search failed: {e}"}

    results = []
    for chunk_id in candidate_ids:
        file_path, text = chunks[int(chunk_id)]
        if Path(file_path).suffix.lower() in CODE_EXTENSIONS:
            results.append({"file": str(file_path), "text": text})
        if len(results) >= top_k:
            break

    if not results:
        return {"success": False, "results": [], "error": "No matching code found"}

    return {"success": True, "results": results, "error": None}


def _load_issues():
    """Read data/issues.json, tolerating a missing or corrupted file (a
    fresh/empty list, not a crash - this is a local mock store, not a
    real database)."""

    if not ISSUES_FILE.exists():
        return []
    try:
        with open(ISSUES_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except (json.JSONDecodeError, OSError):
        return []


def _save_issues(issues):
    ISSUES_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(ISSUES_FILE, "w", encoding="utf-8") as f:
        json.dump(issues, f, indent=2)


@mcp.tool()
def create_issue(title: str, body: str = "", metadata: dict = None) -> dict:
    """Create a tracked issue and persist it locally to data/issues.json.

    This is intentionally a MOCK external side effect - it does NOT call
    the real GitHub API or any other external service. Its purpose is to
    demonstrate a genuinely destructive/write MCP tool that the Tool Agent
    can select, which the client (mcp_client.py) gates behind human
    approval via LangGraph's interrupt() BEFORE this function ever runs
    (see DESTRUCTIVE_TOOLS / call_tool_json). Calling this function
    directly bypasses that gate, so nothing in this project does so.

    Args:
        title: Issue title. Required, non-empty.
        body: Issue body/description. Optional, defaults to "".
        metadata: Optional JSON-serializable object with extra context
            (e.g. {"file": "Dockerfile", "severity": "high"}).

    Returns:
        {"success": bool, "issue": {"id", "title", "body", "metadata",
        "created_at"} | None, "error": str | None}
    """

    if not isinstance(title, str) or not title.strip():
        return {"success": False, "issue": None, "error": "title must be a non-empty string"}

    if metadata is not None and not isinstance(metadata, dict):
        return {"success": False, "issue": None, "error": "metadata must be an object/dict"}

    if body is not None and not isinstance(body, str):
        return {"success": False, "issue": None, "error": "body must be a string"}

    try:
        issues = _load_issues()
        next_id = max((int(i.get("id", 0)) for i in issues), default=0) + 1
        issue = {
            "id": next_id,
            "title": title.strip(),
            "body": body or "",
            "metadata": metadata or {},
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        issues.append(issue)
        _save_issues(issues)
    except Exception as e:
        return {"success": False, "issue": None, "error": f"failed to persist issue: {e}"}

    return {"success": True, "issue": issue, "error": None}



@mcp.tool()
def search_history(query: str, top_k: int = 3) -> dict:
    """Search persistent agent memory (Step 6) for entries relevant to
    `query`: both past COMPLETED interactions (episodic memory) and past
    individual search attempts (search/retrieval memory), most relevant
    first. This is READ-ONLY - it never modifies memory and is never
    added to DESTRUCTIVE_TOOLS, so it never triggers human-in-the-loop
    approval.

    Use this to check whether a similar question has already been
    investigated, or whether a particular search has already been tried
    (and whether it succeeded or failed) before repeating it. Results are
    CONTEXT from past runs, not verified ground truth - the caller should
    still confirm anything important via search_documents/search_code/
    retrieve_file.

    Args:
        query: Natural-language description of what you are looking for
            in the agent's memory (e.g. the current investigation goal).
        top_k: Maximum number of memory entries to return (default 3).

    Returns:
        {"success": bool, "results": [...], "error": str | None}. Each
        result is either {"type": "episodic", "query", "final_answer_snippet",
        "status", "timestamp", "session_id"} or {"type": "search_attempt",
        "tool", "query", "success", "result_count", "timestamp",
        "session_id"}. success is False and results is empty when the
        query is invalid, memory is unavailable, or nothing relevant has
        been recorded yet - it never raises for those cases.
    """

    if not isinstance(query, str) or not query.strip():
        return {"success": False, "results": [], "error": "query must be a non-empty string"}

    try:
        top_k = int(top_k)
    except (TypeError, ValueError):
        top_k = 3

    if top_k <= 0:
        top_k = 3

    if _memory_search_history is None:
        return {"success": False, "results": [], "error": "memory subsystem unavailable"}

    try:
        results = _memory_search_history(query, top_k=top_k)
    except Exception as e:
        return {"success": False, "results": [], "error": f"memory search failed: {e}"}

    if not results:
        return {"success": False, "results": [], "error": "No relevant history found"}

    return {"success": True, "results": results, "error": None}


if __name__ == "__main__":
    mcp.run(transport="stdio")
