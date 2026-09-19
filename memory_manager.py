"""Persistent, cross-session memory layer for the Agentic RAG Engine (Step 6).

This module is deliberately independent of LangGraph and of the MCP
server/client - it is a plain data-access layer that both processes can
import safely:

  - Multi_Agent_System.py (the LangGraph process) imports it directly to
    implement the "Memory Recall" node (before the Planner) and the
    "Memory Persist" node (after the Synthesizer).
  - mcp_server.py (the retrieval MCP server subprocess) imports it to
    implement the new `search_history` MCP tool, so the Tool Agent can
    also query memory on demand, exactly like any other dynamically
    discovered tool.

SHORT-TERM vs LONG-TERM MEMORY - Step 6 explicitly does NOT replace or
duplicate LangGraph's `MemorySaver` checkpointer (see Multi_Agent_System.py).
That checkpointer is short-term/active-execution memory: it lets a single
run pause at a `interrupt()` (Step 5's human-in-the-loop gate) and resume
the SAME thread later, but it lives only in this process's RAM and is gone
the moment the process exits. Everything in THIS module is long-term/
durable memory: plain files under data/, read and written fresh on every
call (no in-process cache), so it survives both a new graph run in the
same process AND a full process restart. The two systems never overlap:
MemorySaver's checkpoints are never read by this module, and this module's
files are never used to resume a paused LangGraph thread.

Four memory tiers, three of which live here (the fourth, short-term, is
LangGraph's checkpointer described above and is intentionally NOT
reimplemented in this file):

  1. Episodic memory   - data/interaction_history.json: one structured
     record per completed graph run (query, plan, tools used, HIL
     approvals/rejections, key evidence snippets, final answer).
  2. Semantic memory    - data/memory/semantic.index (FAISS IndexFlatL2,
     384-dim, same embedding model/dimension the project's existing
     retrieval pipeline already uses) + data/memory/metadata.json (a
     sidecar list mapping each vector to its source text). Stores
     concise, durable facts/conclusions (not raw chunk dumps).
  3. Search/retrieval memory - data/search_history.json: a log of
     individual read-only tool-call attempts (query, tool, timestamp,
     result count, success/failure), used to surface "you already tried
     this" context - a hint, never a hard block.

Design principle (explicitly requested): MEMORY IS CONTEXT, NOT
UNQUESTIONABLE TRUTH. Every read function here returns bounded, small
result sets (top_k, default 3) for the Planner/Tool Agent to consider
alongside - never instead of - live retrieval. Nothing in this module
ever raises out to its caller: every disk/index/embedding operation is
wrapped so a missing file, corrupt JSON, unavailable FAISS index, or
embedding failure degrades to an empty result (or a no-op write) instead
of crashing the core RAG/HIL pipeline.
"""

import json
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = (PROJECT_ROOT / "data").resolve()
MEMORY_DIR = (DATA_DIR / "memory").resolve()

EPISODIC_FILE = (DATA_DIR / "interaction_history.json").resolve()
SEARCH_HISTORY_FILE = (DATA_DIR / "search_history.json").resolve()
SEMANTIC_INDEX_FILE = (MEMORY_DIR / "semantic.index").resolve()
SEMANTIC_METADATA_FILE = (MEMORY_DIR / "metadata.json").resolve()

# Bound file growth so these logs stay small and fast to scan - this is a
# rolling window (oldest entries drop off), not a hard cap on what can ever
# be recalled in one call (top_k for reads is separate and much smaller).
MAX_EPISODIC_RECORDS = 500
MAX_SEARCH_RECORDS = 2000

# Every read used to build a MAX_EPISODIC_RECORDS-record answer or search
# only inspects the small top_k the caller asked for, never the full log.

_lock = threading.Lock()

# Embedding model used ONLY for semantic memory (tier 2). Reuses the exact
# same model name/dimension (384, all-MiniLM-L6-v2) as the project's
# existing retrieval pipeline (embeddings.py) - intentionally NOT the same
# Python module, because embeddings.py's module-level code eagerly embeds
# the entire document corpus as a side effect of import, which would be a
# needless, heavy cost to pay just to embed a short memory string. This is
# the same model, loaded lazily and only if a semantic memory operation is
# actually invoked; if sentence-transformers/faiss are unavailable, every
# semantic function below degrades to a no-op/empty result rather than
# raising.
_embed_model = None


def _now_iso():
    return datetime.now(timezone.utc).isoformat()


# --- Generic JSON-list storage helpers -------------------------------------

def _load_json_list(path):
    if not path.exists():
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except (json.JSONDecodeError, OSError, UnicodeDecodeError):
        # Missing/corrupt file: treat as empty history rather than crash.
        return []


def _save_json_list(path, items):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(items, f, indent=2)
    tmp_path.replace(path)  # atomic on POSIX - avoids a half-written file


def _tokenize(text):
    return set(str(text or "").lower().split())


def _keyword_score(query_tokens, text):
    tokens = _tokenize(text)
    if not tokens or not query_tokens:
        return 0
    return len(query_tokens & tokens)


# --- Tier 2: Episodic memory -------------------------------------------------

def save_interaction(state, session_id=None):
    """Persist one completed (or best-effort) graph run to episodic memory.

    `state` is the AgentState dict (or any mapping with the same keys) at
    the point Memory Persist runs. Only a concise, structured summary is
    stored - not a raw dump of the whole state - per the project's
    "concise summaries, not huge raw dumps" requirement.

    Never raises: any failure (disk full, permissions, corrupt existing
    file) is caught, logged to stdout, and this function returns None
    without persisting - the caller (Memory Persist) must not let a
    memory failure break the graph.
    """

    try:
        tool_results = state.get("tool_results", []) or []
        retrieved_chunks = state.get("retrieved_chunks", []) or []
        final_answer = state.get("final_answer", "") or ""

        tools_used = [
            {
                "tool_name": r.get("tool_name"),
                "success": r.get("success"),
                "status": r.get("status"),
            }
            for r in tool_results
        ]

        approvals = [
            {
                "tool_name": r.get("tool_name"),
                "args": r.get("args"),
                "status": r.get("status"),
                "reason": r.get("error") if r.get("status") in ("rejected", "blocked") else None,
            }
            for r in tool_results
            if r.get("status") in ("approved", "rejected", "blocked")
        ]

        record = {
            "id": str(uuid.uuid4()),
            "timestamp": _now_iso(),
            "session_id": session_id or state.get("session_id") or "",
            "query": state.get("query", ""),
            "plan": list(state.get("plan", []) or []),
            "tools_used": tools_used,
            "approvals": approvals,
            "key_chunks": [c[:400] for c in retrieved_chunks[:3] if isinstance(c, str)],
            "final_answer": final_answer[:4000],
            "status": "completed" if final_answer else "incomplete",
        }

        with _lock:
            items = _load_json_list(EPISODIC_FILE)
            items.append(record)
            if len(items) > MAX_EPISODIC_RECORDS:
                items = items[-MAX_EPISODIC_RECORDS:]
            _save_json_list(EPISODIC_FILE, items)

        return record
    except Exception as e:
        print(f"[memory_manager] WARNING: failed to save episodic interaction: {e}")
        return None


def load_recent_interactions(limit=5):
    """Return up to `limit` most recent episodic records, newest first.

    Never raises: returns [] on any read failure.
    """

    try:
        items = _load_json_list(EPISODIC_FILE)
        if not items:
            return []
        limit = max(0, int(limit))
        return list(reversed(items[-limit:])) if limit else []
    except Exception as e:
        print(f"[memory_manager] WARNING: failed to load recent interactions: {e}")
        return []


# --- Tier 3: Search/retrieval memory ----------------------------------------

def log_search_attempts(session_id, tool_results):
    """Append one search-history record per READ-ONLY tool call found in
    `tool_results` (a list of tool_executor's structured entries).

    Destructive tool calls (create_issue, etc.) are intentionally skipped
    here - those are tracked as episodic "approvals", not searches. Never
    raises: a failure here is logged and swallowed.
    """

    if not tool_results:
        return

    try:
        now = _now_iso()
        records = []
        for r in tool_results:
            tool_name = r.get("tool_name")
            if tool_name not in ("search_documents", "search_code", "retrieve_file"):
                continue
            args = r.get("args") or {}
            query_text = args.get("query") or args.get("path") or ""
            records.append({
                "session_id": session_id or "",
                "timestamp": now,
                "tool": tool_name,
                "query": query_text,
                "result_count": r.get("result_count"),
                "success": bool(r.get("success")),
            })

        if not records:
            return

        with _lock:
            existing = _load_json_list(SEARCH_HISTORY_FILE)
            existing.extend(records)
            if len(existing) > MAX_SEARCH_RECORDS:
                existing = existing[-MAX_SEARCH_RECORDS:]
            _save_json_list(SEARCH_HISTORY_FILE, existing)
    except Exception as e:
        print(f"[memory_manager] WARNING: failed to log search attempts: {e}")


def search_history(query, top_k=3):
    """Keyword-overlap search across BOTH episodic interactions and the
    search-attempt log for entries relevant to `query`.

    This is the function backing the new `search_history` MCP tool AND
    the Memory Recall node's "search memory" tier. Deliberately lexical
    (word-overlap), not embedding-based, so it needs no ML model and
    stays fast/lightweight for what is essentially a "have I seen this
    before" lookup; semantic similarity is covered separately by
    `retrieve_relevant_memory` below.

    Returns BOTH successful and unsuccessful past attempts - a past
    failure is useful context too (e.g. "search_code already failed for
    this term, try search_documents instead").

    Never raises: returns [] on any failure.
    """

    if not query or not isinstance(query, str) or not query.strip():
        return []

    try:
        query_tokens = _tokenize(query)
        scored = []

        for rec in _load_json_list(EPISODIC_FILE):
            text = " ".join([str(rec.get("query", "")), str(rec.get("final_answer", ""))])
            score = _keyword_score(query_tokens, text)
            if score > 0:
                scored.append((score, {
                    "type": "episodic",
                    "session_id": rec.get("session_id"),
                    "timestamp": rec.get("timestamp"),
                    "query": rec.get("query"),
                    "final_answer_snippet": (rec.get("final_answer") or "")[:300],
                    "status": rec.get("status"),
                }))

        for rec in _load_json_list(SEARCH_HISTORY_FILE):
            score = _keyword_score(query_tokens, rec.get("query", ""))
            if score > 0:
                scored.append((score, {
                    "type": "search_attempt",
                    "session_id": rec.get("session_id"),
                    "timestamp": rec.get("timestamp"),
                    "tool": rec.get("tool"),
                    "query": rec.get("query"),
                    "success": rec.get("success"),
                    "result_count": rec.get("result_count"),
                }))

        scored.sort(key=lambda pair: pair[0], reverse=True)
        top_k = max(0, int(top_k))
        return [item for _, item in scored[:top_k]]
    except Exception as e:
        print(f"[memory_manager] WARNING: search_history failed: {e}")
        return []


# --- Tier: Semantic/vector memory -------------------------------------------

def _get_embed_model():
    global _embed_model
    if _embed_model is None:
        from sentence_transformers import SentenceTransformer
        _embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _embed_model


def _embed(text):
    model = _get_embed_model()
    vec = model.encode([text])
    import numpy as np
    return np.asarray(vec, dtype="float32")


def _load_semantic_index():
    import faiss
    if SEMANTIC_INDEX_FILE.exists():
        try:
            return faiss.read_index(str(SEMANTIC_INDEX_FILE))
        except Exception as e:
            print(f"[memory_manager] WARNING: semantic index unreadable, starting fresh: {e}")
    return faiss.IndexFlatL2(384)


def save_memory(text, source=None, session_id=None):
    """Persist a concise durable fact/conclusion to semantic memory.

    Intended for short, high-value summaries (e.g. a confirmed root
    cause) - NOT for raw chunk dumps. Returns True on success, False on
    any failure (missing faiss/sentence-transformers, disk error,
    embedding failure) - never raises.
    """

    if not text or not isinstance(text, str) or not text.strip():
        return False

    try:
        import faiss

        text = text.strip()
        vec = _embed(text)

        with _lock:
            index = _load_semantic_index()
            index.add(vec)
            meta = _load_json_list(SEMANTIC_METADATA_FILE)
            meta.append({
                "id": index.ntotal - 1,
                "text": text[:800],
                "source": source,
                "session_id": session_id,
                "timestamp": _now_iso(),
            })
            MEMORY_DIR.mkdir(parents=True, exist_ok=True)
            faiss.write_index(index, str(SEMANTIC_INDEX_FILE))
            _save_json_list(SEMANTIC_METADATA_FILE, meta)
        return True
    except Exception as e:
        print(f"[memory_manager] WARNING: failed to save semantic memory: {e}")
        return False


def retrieve_relevant_memory(query, top_k=3):
    """Return up to `top_k` durable semantic-memory entries most similar
    to `query` (FAISS L2 nearest neighbors), most relevant first.

    Never raises: returns [] if the index is empty/missing, faiss or
    sentence-transformers is unavailable, or embedding fails.
    """

    if not query or not isinstance(query, str) or not query.strip():
        return []

    try:
        with _lock:
            index = _load_semantic_index()
            if index.ntotal == 0:
                return []
            meta = _load_json_list(SEMANTIC_METADATA_FILE)

        vec = _embed(query.strip())
        k = min(max(1, int(top_k)), index.ntotal)
        distances, indices = index.search(vec, k)

        results = []
        for dist, i in zip(distances[0], indices[0]):
            i = int(i)
            if i < 0 or i >= len(meta):
                continue
            entry = dict(meta[i])
            entry["distance"] = float(dist)
            results.append(entry)
        return results
    except Exception as e:
        print(f"[memory_manager] WARNING: semantic memory retrieval failed: {e}")
        return []
