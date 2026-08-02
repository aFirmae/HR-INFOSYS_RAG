import time
import json
from glob import glob
from typing import List, Dict
import sys
import os
from sentence_transformers.util import cos_sim
from langchain_community.embeddings import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
from langchain_community.vectorstores import Qdrant

# Add parent directory to import config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import QDRANT_HOST, QDRANT_API_KEY, QDRANT_COLLECTIONS, EMBED_MODEL

# -----------------------------
# Load all benchmark JSONs
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
    api_key=QDRANT_API_KEY,
)

# -----------------------------
# Benchmark function (no per-query logs)
# -----------------------------
def benchmark_retriever(collection_name: str, benchmark_data: List[Dict[str, str]], k: int = 8):
    vectorstore = Qdrant(
        client=qdrant_client,
        collection_name=collection_name,
        embeddings=embedding_model
    )

    total_time = 0.0
    total_accuracy = 0.0
    valid_queries = 0

    for item in benchmark_data:
        query = item.get("query") or item.get("question") or item.get("prompt")
        ground_truth_answer = item.get("answer") or item.get("solution")
        if not query or not ground_truth_answer:
            continue

        valid_queries += 1

        # Measure retrieval time
        start_time = time.time()
        docs = vectorstore.similarity_search(query, k=k)
        elapsed_time = time.time() - start_time
        total_time += elapsed_time

        # Cosine similarity using top-k documents individually
        similarities = []
        ground_emb = embedding_model.embed_documents([ground_truth_answer])[0]
        for doc in docs[:2]:  # Only top 2 results
            doc_emb = embedding_model.embed_documents([doc.page_content])[0]
            similarities.append(cos_sim(doc_emb, ground_emb).item())
        avg_similarity = sum(similarities) / len(similarities) if similarities else 0.0
        total_accuracy += avg_similarity

    if valid_queries == 0:
        return None

    return {
        "total_queries": valid_queries,
        "avg_time": total_time / valid_queries,
        "avg_accuracy": total_accuracy / valid_queries
    }

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    benchmark_folder = r"D:\AGENTIC_AI\PROJECTS\hr-assistant\Benchmark"
    benchmark_data = load_all_benchmarks(benchmark_folder)
    print(f"Loaded {len(benchmark_data)} total benchmark queries from '{benchmark_folder}'\n")

    if not QDRANT_COLLECTIONS:
        print("Error: QDRANT_COLLECTIONS not defined in config.py.")
    else:
        for collection in QDRANT_COLLECTIONS:
            summary = benchmark_retriever(collection, benchmark_data)
            if summary:
                print("="*25 + f" BENCHMARK SUMMARY: {collection} " + "="*25)
                print(f"Total Queries:          {summary['total_queries']}")
                print(f"Average Retrieval Time: {summary['avg_time']:.3f} sec/query")
                print(f"Average Cosine Similarity (top 2 docs): {summary['avg_accuracy']:.4f}")
                print("="*70 + "\n")
