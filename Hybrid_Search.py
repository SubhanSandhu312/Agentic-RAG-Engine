from chunking import chunks
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

tokenized_chunks = [
    chunk[1].lower().split()
    for chunk in chunks
]

bm25 = BM25Okapi(tokenized_chunks)

cross_encoder = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)


def query_bm25(query, top_k=5):

    tokenized_query = query.lower().split()

    scores = bm25.get_scores(tokenized_query)

    top_indices = scores.argsort()[-top_k:][::-1]

    return top_indices


def rerank(query, ranked_chunks, top_k=5):

    pairs = [
        [query, chunks[chunk_id][1]]
        for chunk_id, rrf_score in ranked_chunks
    ]

    scores = cross_encoder.predict(pairs)

    reranked = sorted(
        zip(ranked_chunks, scores),
        key=lambda x: x[1],
        reverse=True
    )

    return [
        (int(chunk_id), float(score))
        for ((chunk_id, rrf_score), score)
        in reranked[:top_k]
    ]