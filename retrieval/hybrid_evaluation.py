import time
import json
from glob import glob
from typing import List, Dict
import sys
import os
from sentence_transformers.util import cos_sim
from langchain_community.embeddings import HuggingFaceEmbeddings

# Add parent directory to import config and hybrid_retriever
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import EMBED_MODEL, QDRANT_COLLECTIONS
from retrieval.hybrid_retrieval import HybridRetrieverOptimal

# -----------------------------
# Load all benchmark JSONs
# -----------------------------
def load_all_benchmarks(folder_path: str) -> List[Dict[str, str]]:
    all_data = []
    for file_path in glob(os.path.join(folder_path, "*.json")):
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            all_data.extend(data)
    return all_data

# -----------------------------
# Benchmark hybrid retriever (no logs, only final summary)
# -----------------------------
def benchmark_hybrid(hybrid: HybridRetrieverOptimal, benchmark_data: List[Dict[str, str]], k: int = 10):
    total_time = 0.0
    total_similarity = 0.0
    valid_queries = 0

    for item in benchmark_data:
        query = item.get("query") or item.get("question") or item.get("prompt")
        ground_truth = item.get("answer") or item.get("solution")

        if not query or not ground_truth:
            continue

        valid_queries += 1
        start_time = time.time()
        top_docs = hybrid.get_relevant_documents(query, k=k)
        total_time += time.time() - start_time

        # Compute average cosine similarity over top k docs
        ground_emb = hybrid.embedding_model.embed_documents([ground_truth])[0]
        similarities = []
        for doc in top_docs:
            doc_emb = hybrid.embedding_model.embed_documents([doc.page_content])[0]
            similarities.append(cos_sim(doc_emb, ground_emb).item())

        avg_sim = sum(similarities) / len(similarities) if similarities else 0.0
        total_similarity += avg_sim

    if valid_queries == 0:
        return None

    return {
        "total_queries": valid_queries,
        "avg_time": total_time / valid_queries,
        "avg_similarity": total_similarity / valid_queries
    }

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    benchmark_folder = r"D:\AGENTIC_AI\PROJECTS\hr-assistant\Benchmark"
    sparse_folder = r"D:\AGENTIC_AI\PROJECTS\hr-assistant\data"

    benchmark_data = load_all_benchmarks(benchmark_folder)
    print(f"Loaded {len(benchmark_data)} benchmark queries.\n")

    for collection_name in QDRANT_COLLECTIONS:
        # Initialize retriever for each collection
        hybrid = HybridRetrieverOptimal(
            dense_index_type=collection_name, 
            sparse_docs_folder=sparse_folder, 
            k=10
        )
        summary = benchmark_hybrid(hybrid, benchmark_data, k=10)
        if summary:
            print("="*25 + f" BENCHMARK SUMMARY: {collection_name} " + "="*25)
            print(f"Total Queries:          {summary['total_queries']}")
            print(f"Average Retrieval Time: {summary['avg_time']:.3f} sec/query")
            print(f"Average Cosine Similarity (top 10 docs): {summary['avg_similarity']:.4f}")
            print("="*70 + "\n")
