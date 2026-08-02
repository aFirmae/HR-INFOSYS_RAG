import time
import json
from typing import List, Dict
from qdrant_client import QdrantClient
from langchain_community.vectorstores import Qdrant
from langchain_community.embeddings import HuggingFaceEmbeddings
from sentence_transformers.util import cos_sim
import sys
import os
from glob import glob

# Add parent directory to import config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import QDRANT_HOST, QDRANT_API_KEY, QDRANT_COLLECTIONS, EMBED_MODEL

# -----------------------------
# Load all benchmark JSONs from folder
# -----------------------------
def load_all_benchmarks(folder_path: str) -> List[Dict[str, str]]:
    all_data = []
    json_files = glob(os.path.join(folder_path, "*.json"))
    for file_path in json_files:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            all_data.extend(data)
    return all_data

# -----------------------------
# Initialize embedding model
# -----------------------------
embedding_model = HuggingFaceEmbeddings(
    model_name=EMBED_MODEL or "sentence-transformers/all-MiniLM-L6-v2"
)

# -----------------------------
# Initialize Qdrant client
# -----------------------------
qdrant_client = QdrantClient(
    url=QDRANT_HOST,
    api_key=QDRANT_API_KEY
)

# -----------------------------
# Benchmark function
# -----------------------------
def benchmark_retriever(collection_name: str, benchmark_data: List[Dict[str, str]], k: int = 8):
    """
    Benchmarks a dense retriever using Qdrant on response time and average similarity
    among all retrieved chunks.
    Returns overall stats for aggregation.
    """
    print(f"\n🚀 Benchmarking Collection: {collection_name}")
    print("-" * 80)
    
    vectorstore = Qdrant(
        client=qdrant_client,
        collection_name=collection_name,
        embeddings=embedding_model
    )

    total_time = 0.0
    total_accuracy = 0.0
    valid_queries = 0

    for i, item in enumerate(benchmark_data, 1):
        # Robustly fetch query and answer
        query = item.get("query") or item.get("question") or item.get("prompt")
        ground_truth_answer = item.get("answer") or item.get("solution")

        if not query or not ground_truth_answer:
            continue

        valid_queries += 1

        # --- Measure Retrieval Time ---
        start_time = time.time()
        docs = vectorstore.similarity_search(query, k=k)
        elapsed_time = time.time() - start_time
        total_time += elapsed_time

        # --- Measure Accuracy (Average cosine similarity across all retrieved chunks) ---
        avg_similarity = 0.0
        if docs:
            ground_emb = embedding_model.embed_documents([ground_truth_answer])[0]
            similarities = []
            for doc in docs:
                doc_emb = embedding_model.embed_documents([doc.page_content])[0]
                sim = cos_sim(doc_emb, ground_emb).item()
                similarities.append(sim)
            avg_similarity = sum(similarities) / len(similarities)
        total_accuracy += avg_similarity

    if valid_queries == 0:
        print("No valid benchmark queries found. Exiting.")
        return {"total_queries": 0, "avg_time": 0.0, "avg_accuracy": 0.0}

    avg_time = total_time / valid_queries
    avg_accuracy = total_accuracy / valid_queries

    print(f"\n✅ Collection '{collection_name}' Summary:")
    print(f"Total Queries:          {valid_queries}")
    print(f"Average Retrieval Time: {avg_time:.3f} sec/query")
    print(f"Average Accuracy Score: {avg_accuracy:.4f}")
    print("="*69 + "\n")

    return {"total_queries": valid_queries, "avg_time": avg_time, "avg_accuracy": avg_accuracy}

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    benchmark_folder = r"D:\AGENTIC_AI\PROJECTS\hr-assistant\Benchmark"
    benchmark_data = load_all_benchmarks(benchmark_folder)
    print(f"Loaded {len(benchmark_data)} total benchmark queries from folder '{benchmark_folder}'\n")

    if not QDRANT_COLLECTIONS:
        print("Error: QDRANT_COLLECTIONS not defined in config.py.")
        sys.exit(1)

    # Aggregate stats across all collections
    overall_queries = 0
    overall_time = 0.0
    overall_accuracy = 0.0

    for collection in QDRANT_COLLECTIONS:
        stats = benchmark_retriever(collection, benchmark_data)
        overall_queries += stats["total_queries"]
        overall_time += stats["avg_time"] * stats["total_queries"]
        overall_accuracy += stats["avg_accuracy"] * stats["total_queries"]

    if overall_queries > 0:
        print("\n" + "="*25 + " OVERALL BENCHMARK SUMMARY " + "="*25)
        print(f"Total Queries:          {overall_queries}")
        print(f"Average Retrieval Time: {overall_time / overall_queries:.3f} sec/query")
        print(f"Average Accuracy Score: {overall_accuracy / overall_queries:.4f}")
        print("="*69 + "\n")
    else:
        print("No valid queries across all collections.")
