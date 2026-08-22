from chunking import chunks
from rank_bm25 import BM25Okapi

tokenized_chunks = [
    chunk[1].lower().split()
    for chunk in chunks
]

bm25 = BM25Okapi(tokenized_chunks)


def query_bm25(query, top_k=5):

    tokenized_query = query.lower().split()

    scores = bm25.get_scores(tokenized_query)

    top_indices = scores.argsort()[-top_k:][::-1]

    return top_indices