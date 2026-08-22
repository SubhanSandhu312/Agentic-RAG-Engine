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




def the_call(query):

    query_vector = query_embedding(query)

    scores, faiss_indices = query_indices(
        query_vector,
        top_k=10
    )

    faiss_indices = faiss_indices[0]

    bm25_indices = query_bm25(
        query,
        top_k=10
    )

    rrf_scores = {}

    for rank, chunk_id in enumerate(
        bm25_indices,
        start=1
    ):
        rrf_scores[chunk_id] = (
            rrf_scores.get(chunk_id, 0)
            + 1 / (60 + rank)
        )

    for rank, chunk_id in enumerate(
        faiss_indices,
        start=1
    ):
        rrf_scores[chunk_id] = (
            rrf_scores.get(chunk_id, 0)
            + 1 / (60 + rank)
        )

    ranked_chunks = sorted(
        rrf_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    print("RRF RESULTS:")
    print(ranked_chunks[:10])

    reranked_chunks = rerank(
        query,
        ranked_chunks,
        top_k=5
    )

    print("\nCROSS-ENCODER RESULTS:")
    print(reranked_chunks)

    top_chunks = [
        chunks[chunk_id]
        for chunk_id, score in reranked_chunks[:2]
    ]

    context_text = "\n\n".join(
        chunk[1]
        for chunk in top_chunks
    )

    print("\nANSWER:")
    print(
        generate_answer(
            query,
            context=context_text
        )
    )

query = "What is the issue in the docker-build-fail.yml file and how can it be resolved?"
the_call(query)