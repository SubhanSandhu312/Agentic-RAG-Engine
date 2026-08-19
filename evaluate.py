import json
import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from embeddings import embeddings_list  # Import your vector embeddings array

# 1. Load your SentenceTransformer embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# 2. Load your local lookup table (maps index ID -> metadata)
with open("chunks_metadata.json", "r", encoding="utf-8") as f:
    chunks_metadata = json.load(f)

# 3. Load the benchmark dataset
with open("benchmark_dataset.json", "r", encoding="utf-8") as f:
    benchmark_data = json.load(f)

# 4. Setup FAISS Index
embeddings_array = np.array(embeddings_list, dtype='float32')
dimension = embeddings_array.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings_array)

def normalize_path(path_str):
    """
    Standardizes path strings across OS platforms by stripping root 
    directory prefixes and converting backslashes to forward slashes.
    """
    clean_path = os.path.normpath(str(path_str)).replace("\\", "/")
    
    # Strip leading directory prefixes if present (e.g., "data/")
    if clean_path.startswith("data/"):
        clean_path = clean_path[5:]
        
    return clean_path

def evaluate_retrieval(k=3, output_filename="Naive_RAG_results.txt"):
    total_recall = 0.0
    total_precision = 0.0

    # Store formatted strings for both console display and file writing
    logs = []
    
    header = f"--- Running Step 1 Evaluation Baseline (K={k}) ---"
    print(header)
    logs.append(header)

    for entry in benchmark_data:
        question = entry["question"]
        expected_sources = entry["expected_sources"]
        expected_chunk_ids = entry["expected_chunk_ids"]

        # Encode the question using the embedding model (returns shape (1, D))
        query_vector = model.encode([question]).astype('float32')

        # Retrieve top-K from FAISS
        distances, retrieved_indices = index.search(query_vector, k)
        retrieved_ids = retrieved_indices[0].tolist()

        # Check hits against ground truth chunk IDs or expected source files
        hits = 0
        for idx in retrieved_ids:
            # Normalize retrieved file path
            raw_retrieved_file = chunks_metadata[idx]["source_file"]
            retrieved_file = normalize_path(raw_retrieved_file)
            
            # Normalize expected source paths from benchmark
            normalized_expected_sources = [normalize_path(src) for src in expected_sources]

            # Check for chunk ID or source file match
            if idx in expected_chunk_ids or retrieved_file in normalized_expected_sources:
                hits += 1

        # Calculate Recall@K & Precision@K
        recall = hits / len(expected_chunk_ids) if expected_chunk_ids else 0.0
        precision = hits / k

        total_recall += min(recall, 1.0)  # Cap recall per query at 1.0
        total_precision += precision

        log_line = f"Q: {question[:45]}... | Hits: {hits}/{k} | Precision@{k}: {precision:.2f}"
        print(log_line)
        logs.append(log_line)

    avg_recall = total_recall / len(benchmark_data)
    avg_precision = total_precision / len(benchmark_data)

    summary = [
        "\n==========================================",
        "      STEP 1 BASELINE EVALUATION METRICS  ",
        "==========================================",
        f"Average Recall@{k}:    {avg_recall:.4f}",
        f"Average Precision@{k}: {avg_precision:.4f}",
        "=========================================="
    ]

    for line in summary:
        print(line)
        logs.append(line)

    # Write logs and metrics to a text file
    with open(output_filename, "w", encoding="utf-8") as file:
        file.write("\n".join(logs) + "\n")
        
    print(f"\nResults saved successfully to {output_filename}!")
    
    return {"recall": avg_recall, "precision": avg_precision}

if __name__ == "__main__":
    evaluate_retrieval(k=3)