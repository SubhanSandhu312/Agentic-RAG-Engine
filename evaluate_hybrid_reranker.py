import json
import os

from embeddings import query_embedding
from Faiss_Searach import query_indices
from Hybrid_Search import query_bm25, rerank
from chunking import chunks


with open("benchmark_dataset.json", "r", encoding="utf-8") as f:
    benchmark_data = json.load(f)


def normalize_path(path_str):

    clean_path = os.path.normpath(
        str(path_str)
    ).replace("\\", "/")

    if clean_path.startswith("data/"):
        clean_path = clean_path[5:]

    return clean_path


def get_bm25_indices(query, k):

    indices = query_bm25(
        query,
        top_k=k
    )

    return [int(x) for x in indices]


def get_faiss_indices(query, k):

    query_vector = query_embedding(query)

    scores, indices = query_indices(
        query_vector,
        top_k=k
    )

    return [
        int(x)
        for x in indices[0]
    ]


def rrf_fusion(bm25_indices, faiss_indices):

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

    ranked = sorted(
        rrf_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    return ranked


def calculate_metrics(
    retrieved_ids,
    expected_chunk_ids,
    expected_sources,
    k
):

    retrieved_ids = retrieved_ids[:k]

    expected_chunk_ids = set(
        int(x)
        for x in expected_chunk_ids
    )

    normalized_expected_sources = set(
        normalize_path(x)
        for x in expected_sources
    )

    # --------------------------------
    # Exact chunk hits
    # --------------------------------

    exact_hits = [
        idx
        for idx in retrieved_ids
        if idx in expected_chunk_ids
    ]

    # --------------------------------
    # Source file hits
    # --------------------------------

    source_hits = []

    for idx in retrieved_ids:

        retrieved_file = normalize_path(
            chunks[idx][0]
        )

        if retrieved_file in normalized_expected_sources:
            source_hits.append(idx)

    # --------------------------------
    # Recall
    # --------------------------------

    exact_recall = (
        len(set(exact_hits))
        /
        len(expected_chunk_ids)
        if expected_chunk_ids
        else 0.0
    )

    source_recall = (
        1.0
        if source_hits
        else 0.0
    )

    # --------------------------------
    # Precision
    # --------------------------------

    exact_precision = (
        len(exact_hits) / k
        if k
        else 0.0
    )

    source_precision = (
        len(source_hits) / k
        if k
        else 0.0
    )

    # --------------------------------
    # MRR
    # --------------------------------

    reciprocal_rank = 0.0

    for rank, idx in enumerate(
        retrieved_ids,
        start=1
    ):

        if idx in expected_chunk_ids:

            reciprocal_rank = 1 / rank
            break

    return {
        "exact_hits": exact_hits,
        "source_hits": source_hits,
        "recall": min(exact_recall, 1.0),
        "source_recall": source_recall,
        "precision": exact_precision,
        "source_precision": source_precision,
        "mrr": reciprocal_rank
    }


def evaluate_retrieval(
    retrieval_k=10,
    final_k_values=(3, 5, 10),
    output_filename="Hybrid_Reranker_results.txt"
):

    logs = []

    # ==========================================
    # Totals
    # ==========================================

    totals = {

        "bm25_recall": 0.0,
        "faiss_recall": 0.0,
        "rrf_recall": 0.0,

        "bm25_mrr": 0.0,
        "faiss_mrr": 0.0,
        "rrf_mrr": 0.0,

        "reranker": {}
    }

    for k in final_k_values:

        totals["reranker"][k] = {
            "recall": 0.0,
            "precision": 0.0,
            "mrr": 0.0
        }

    header = [
        "",
        "==========================================",
        "       RAG RETRIEVAL EVALUATION",
        "==========================================",
        f"Benchmark Questions: {len(benchmark_data)}",
        f"Retrieval K: {retrieval_k}",
        f"Final K Values: {final_k_values}",
        ""
    ]

    for line in header:
        print(line)
        logs.append(line)

    # ==========================================
    # Evaluate every question
    # ==========================================

    for entry in benchmark_data:

        question = entry["question"]

        expected_chunk_ids = [
            int(x)
            for x in entry["expected_chunk_ids"]
        ]

        expected_sources = entry[
            "expected_sources"
        ]

        print("\n" + "-" * 60)
        print("QUESTION:")
        print(question)

        logs.append("\n" + "-" * 60)
        logs.append("QUESTION:")
        logs.append(question)

        # ======================================
        # BM25
        # ======================================

        bm25_indices = get_bm25_indices(
            question,
            retrieval_k
        )

        # ======================================
        # FAISS
        # ======================================

        faiss_indices = get_faiss_indices(
            question,
            retrieval_k
        )

        # ======================================
        # RRF
        # ======================================

        rrf_ranked = rrf_fusion(
            bm25_indices,
            faiss_indices
        )

        rrf_ids = [
            chunk_id
            for chunk_id, score
            in rrf_ranked
        ]

        # ======================================
        # Evaluate BM25
        # ======================================

        bm25_metrics = calculate_metrics(
            bm25_indices,
            expected_chunk_ids,
            expected_sources,
            retrieval_k
        )

        # ======================================
        # Evaluate FAISS
        # ======================================

        faiss_metrics = calculate_metrics(
            faiss_indices,
            expected_chunk_ids,
            expected_sources,
            retrieval_k
        )

        # ======================================
        # Evaluate RRF
        # ======================================

        rrf_metrics = calculate_metrics(
            rrf_ids,
            expected_chunk_ids,
            expected_sources,
            retrieval_k
        )

        totals["bm25_recall"] += (
            bm25_metrics["recall"]
        )

        totals["faiss_recall"] += (
            faiss_metrics["recall"]
        )

        totals["rrf_recall"] += (
            rrf_metrics["recall"]
        )

        totals["bm25_mrr"] += (
            bm25_metrics["mrr"]
        )

        totals["faiss_mrr"] += (
            faiss_metrics["mrr"]
        )

        totals["rrf_mrr"] += (
            rrf_metrics["mrr"]
        )

        # ======================================
        # Print retrieval results
        # ======================================

        retrieval_log = [
            "",
            f"Expected chunks: {expected_chunk_ids}",
            "",
            f"BM25:  {bm25_indices}",
            f"Recall: {bm25_metrics['recall']:.2f}",
            f"MRR:    {bm25_metrics['mrr']:.2f}",
            "",
            f"FAISS: {faiss_indices}",
            f"Recall: {faiss_metrics['recall']:.2f}",
            f"MRR:    {faiss_metrics['mrr']:.2f}",
            "",
            f"RRF:   {rrf_ids}",
            f"Recall: {rrf_metrics['recall']:.2f}",
            f"MRR:    {rrf_metrics['mrr']:.2f}"
        ]

        for line in retrieval_log:
            print(line)
            logs.append(line)

        # ======================================
        # Cross Encoder
        # ======================================

        reranked = rerank(
            question,
            rrf_ranked,
            top_k=retrieval_k
        )

        reranked_ids = [
            int(chunk_id)
            for chunk_id, score
            in reranked
        ]

        # ======================================
        # Evaluate Cross Encoder
        # ======================================

        for k in final_k_values:

            metrics = calculate_metrics(
                reranked_ids,
                expected_chunk_ids,
                expected_sources,
                k
            )

            totals["reranker"][k][
                "recall"
            ] += metrics["recall"]

            totals["reranker"][k][
                "precision"
            ] += metrics["precision"]

            totals["reranker"][k][
                "mrr"
            ] += metrics["mrr"]

        # ======================================
        # Print Cross Encoder ranking
        # ======================================

        ce_log = [
            "",
            "Cross-Encoder ranking:",
            str(reranked_ids)
        ]

        for line in ce_log:
            print(line)
            logs.append(line)

        # ======================================
        # Show final K results
        # ======================================

        for k in final_k_values:

            metrics = calculate_metrics(
                reranked_ids,
                expected_chunk_ids,
                expected_sources,
                k
            )

            line = (
                f"CE Recall@{k}: "
                f"{metrics['recall']:.2f} | "
                f"Precision@{k}: "
                f"{metrics['precision']:.2f} | "
                f"MRR: "
                f"{metrics['mrr']:.2f}"
            )

            print(line)
            logs.append(line)

    # ==========================================
    # AVERAGES
    # ==========================================

    n = len(benchmark_data)

    avg_bm25_recall = (
        totals["bm25_recall"] / n
    )

    avg_faiss_recall = (
        totals["faiss_recall"] / n
    )

    avg_rrf_recall = (
        totals["rrf_recall"] / n
    )

    avg_bm25_mrr = (
        totals["bm25_mrr"] / n
    )

    avg_faiss_mrr = (
        totals["faiss_mrr"] / n
    )

    avg_rrf_mrr = (
        totals["rrf_mrr"] / n
    )

    # ==========================================
    # SUMMARY
    # ==========================================

    summary = [
        "",
        "",
        "==========================================",
        "       FINAL RETRIEVAL RESULTS",
        "==========================================",
        "",
        "                 Recall@K",
        "------------------------------------------",
        f"BM25:            {avg_bm25_recall:.4f}",
        f"FAISS:           {avg_faiss_recall:.4f}",
        f"RRF:              {avg_rrf_recall:.4f}"
    ]

    for k in final_k_values:

        avg_recall = (
            totals["reranker"][k]["recall"]
            / n
        )

        avg_precision = (
            totals["reranker"][k]["precision"]
            / n
        )

        avg_mrr = (
            totals["reranker"][k]["mrr"]
            / n
        )

        summary.append(
            f"RRF + CE @ {k}:    "
            f"Recall={avg_recall:.4f} | "
            f"Precision={avg_precision:.4f} | "
            f"MRR={avg_mrr:.4f}"
        )

    summary.extend([
        "",
        "                 MRR",
        "------------------------------------------",
        f"BM25:            {avg_bm25_mrr:.4f}",
        f"FAISS:           {avg_faiss_mrr:.4f}",
        f"RRF:             {avg_rrf_mrr:.4f}",
        "",
        "=========================================="
    ])

    for line in summary:
        print(line)
        logs.append(line)

    # ==========================================
    # SAVE
    # ==========================================

    with open(
        output_filename,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            "\n".join(logs)
            + "\n"
        )

    print(
        f"\nResults saved successfully to "
        f"{output_filename}!"
    )


if __name__ == "__main__":

    evaluate_retrieval(
        retrieval_k=20,
        final_k_values=(3, 5, 10)
    )