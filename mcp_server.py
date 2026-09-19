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

import sys
from pathlib import Path

# Make sure the project root is importable regardless of the working
# directory the server happens to be launched from.
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from mcp.server.fastmcp import FastMCP

from llm_calling import search_documents_structured

DATA_DIR = (PROJECT_ROOT / "data").resolve()

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


if __name__ == "__main__":
    mcp.run(transport="stdio")
