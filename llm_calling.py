from chunking import chunks
from Hybrid_Search import query_bm25,rerank
from embeddings import query_embedding
from Faiss_Searach import query_indices
# from embeddings import embeddings_list_whole
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()



load_dotenv()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY")
)

def generate_answer(query, context):

    prompt = f"""
You are a helpful RAG assistant.

Answer the user's question using ONLY the provided context.

If the answer cannot be found in the context, say:
"I don't have enough information in the provided context."

Context:
{context}

Question:
{query}
"""

    response = client.chat.completions.create(
        model="openrouter/free",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response.choices[0].message.content


def _rrf_fuse(bm25_indices, faiss_indices, k=60):
    """Reciprocal Rank Fusion of the BM25 and FAISS candidate lists.

    Extracted out of the old inline `the_call` loop so both `the_call` and
    the new MCP `search_documents` tool (see mcp_server.py) share the exact
    same fusion logic instead of duplicating it.
    """

    rrf_scores = {}

    for rank, chunk_id in enumerate(bm25_indices, start=1):
        rrf_scores[chunk_id] = (
            rrf_scores.get(chunk_id, 0)
            + 1 / (k + rank)
        )

    for rank, chunk_id in enumerate(faiss_indices, start=1):
        rrf_scores[chunk_id] = (
            rrf_scores.get(chunk_id, 0)
            + 1 / (k + rank)
        )

    return sorted(
        rrf_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )


def retrieve_ranked_chunks(query, top_k=10, rerank_top_k=5):
    """Shared retrieval pipeline: embedding -> FAISS + BM25 -> RRF fuse ->
    cross-encoder rerank.

    Returns a list of (chunk_id, cross_encoder_score) tuples, most relevant
    first. Returns an empty list if there is nothing to rank (e.g. an empty
    corpus), rather than raising, so callers can treat "no results" as a
    normal, structured outcome.

    Both `the_call` (used by the standalone/manual pipeline) and
    `search_documents_structured` (used by the MCP `search_documents` tool)
    call this function, so the BM25 + FAISS + RRF + cross-encoder algorithms
    themselves live in exactly one place.
    """

    query_vector = query_embedding(query)

    scores, faiss_indices = query_indices(
        query_vector,
        top_k=top_k
    )

    faiss_indices = faiss_indices[0]

    bm25_indices = query_bm25(
        query,
        top_k=top_k
    )

    ranked_chunks = _rrf_fuse(bm25_indices, faiss_indices)

    print("RRF RESULTS:")
    print(ranked_chunks[:10])

    if not ranked_chunks:
        return []

    return rerank(
        query,
        ranked_chunks,
        top_k=rerank_top_k
    )


def the_call(query):

    reranked_chunks = retrieve_ranked_chunks(query, top_k=10, rerank_top_k=5)

    print("\nCROSS-ENCODER RESULTS:")
    print(reranked_chunks)

    if not reranked_chunks:
        return ""

    top_chunks = [
        chunks[chunk_id]
        for chunk_id, score in reranked_chunks[:2]
    ]

    context_text = "\n\n".join(
        chunk[1]
        for chunk in top_chunks
    )

    return context_text


def search_documents_structured(query, top_k=5):
    """Structured, MCP-friendly wrapper around the same retrieval pipeline
    used by `the_call`. Used by the MCP server's `search_documents` tool.

    Returns:
        {"success": bool, "results": [{"file": str, "text": str, "score": float}], "error": str|None}
    """

    if not isinstance(query, str) or not query.strip():
        return {"success": False, "results": [], "error": "query must be a non-empty string"}

    try:
        top_k = int(top_k)
    except (TypeError, ValueError):
        top_k = 5
    if top_k <= 0:
        top_k = 5

    try:
        reranked_chunks = retrieve_ranked_chunks(
            query,
            top_k=max(10, top_k * 2),
            rerank_top_k=top_k
        )
    except Exception as e:
        return {"success": False, "results": [], "error": f"retrieval failed: {e}"}

    if not reranked_chunks:
        return {"success": False, "results": [], "error": "No relevant documents found"}

    results = []
    for chunk_id, score in reranked_chunks:
        file_path, chunk_text = chunks[chunk_id]
        results.append({
            "file": str(file_path),
            "text": chunk_text,
            "score": float(score)
        })

    return {"success": True, "results": results, "error": None}




# def the_call(query):

#     query_vector = query_embedding(query)

#     scores, faiss_indices = query_indices(
#         query_vector,
#         top_k=10
#     )

#     faiss_indices = faiss_indices[0]

#     bm25_indices = query_bm25(
#         query,
#         top_k=10
#     )

#     rrf_scores = {}

#     for rank, chunk_id in enumerate(
#         bm25_indices,
#         start=1
#     ):
#         rrf_scores[chunk_id] = (
#             rrf_scores.get(chunk_id, 0)
#             + 1 / (60 + rank)
#         )

#     for rank, chunk_id in enumerate(
#         faiss_indices,
#         start=1
#     ):
#         rrf_scores[chunk_id] = (
#             rrf_scores.get(chunk_id, 0)
#             + 1 / (60 + rank)
#         )

#     ranked_chunks = sorted(
#         rrf_scores.items(),
#         key=lambda x: x[1],
#         reverse=True
#     )

#     print("RRF RESULTS:")
#     print(ranked_chunks[:10])

#     reranked_chunks = rerank(
#         query,
#         ranked_chunks,
#         top_k=5
#     )

#     print("\nCROSS-ENCODER RESULTS:")
#     print(reranked_chunks)

#     top_chunks = [
#         chunks[chunk_id]
#         for chunk_id, score in reranked_chunks[:2]
#     ]

#     context_text = "\n\n".join(
#         chunk[1]
#         for chunk in top_chunks
#     )

#     print("\nANSWER:")
#     print(
#         generate_answer(
#             query,
#             context=context_text
#         )
#     )

# query = "What is the issue in the docker-build-fail.yml file and how can it be resolved?"
# the_call(query)
