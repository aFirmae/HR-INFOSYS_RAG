import time
import json
import sys
import os
from glob import glob
from typing import List, Dict
import numpy as np
from qdrant_client import QdrantClient
from langchain_community.vectorstores import Qdrant
from langchain_community.embeddings import HuggingFaceEmbeddings
from sentence_transformers.util import cos_sim
from langchain_core.documents import Document

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import QDRANT_HOST, QDRANT_API_KEY, QDRANT_COLLECTIONS, EMBED_MODEL

# --- Load all benchmark JSONs ---
def load_all_benchmarks(folder_path: str) -> List[Dict[str, str]]:
    all_data = []
    for file in glob(os.path.join(folder_path, "*.json")):
        with open(file, "r", encoding="utf-8") as f:
            data = json.load(f)
            all_data.extend(data)
    return all_data

# --- Optimal Hybrid Retriever ---
class HybridRetrieverOptimal:
    def __init__(
        self,
        dense_index_type: str = "hnsw",
        sparse_docs_folder: str = None,
        k: int = 20,
        dense_weight: float = 0.9,
        sparse_weight: float = 0.1,
        rerank_top_n: int = 10
    ):
        self.k = k
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        self.rerank_top_n = rerank_top_n

        # Dense setup
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=EMBED_MODEL or "sentence-transformers/all-MiniLM-L6-v2"
        )
        qdrant_client = QdrantClient(url=QDRANT_HOST, api_key=QDRANT_API_KEY)

        collection_name = None
        for name in QDRANT_COLLECTIONS:
            if dense_index_type in name:
                collection_name = name
                break
        if not collection_name:
            raise ValueError(f"No Qdrant collection found for {dense_index_type}")

        dense_store = Qdrant(
            client=qdrant_client,
            collection_name=collection_name,
            embeddings=self.embedding_model
        )
        self.dense_retriever = dense_store.as_retriever(search_kwargs={"k": k})

        # Sparse setup (optional)
        from bm25_retriever import BM25Retriever, load_pdfs_from_folder
        self.sparse_retriever = None
        if sparse_docs_folder:
            all_docs = load_pdfs_from_folder(sparse_docs_folder)
            if all_docs:
                self.sparse_retriever = BM25Retriever(all_docs)

    def _normalize_scores(self, scores: List[float]) -> List[float]:
        """Normalize to [0,1]."""
        if not scores:
            return []
        min_s, max_s = min(scores), max(scores)
        if max_s - min_s < 1e-6:
            return [1.0 for _ in scores]  # avoid divide by zero
        return [(s - min_s) / (max_s - min_s) for s in scores]

    def get_relevant_documents(self, query: str, k: int = None) -> List[Document]:
        k = k or self.k

        # --- 1. Dense retrieval ---
        dense_docs = self.dense_retriever.get_relevant_documents(query)
        dense_scores = [max(doc.metadata.get("score", 0.0), 0.01) for doc in dense_docs]
        dense_scores = self._normalize_scores(dense_scores)
        for doc, score in zip(dense_docs, dense_scores):
            doc.metadata["dense_score"] = score

        # --- 2. Conditional sparse retrieval ---
        sparse_docs = []
        if self.sparse_retriever:
            # Only run sparse if query contains keywords or low dense confidence
            low_confidence = np.mean(dense_scores) < 0.5
            if low_confidence or any(word.lower() in query.lower() for word in ["policy", "leave", "salary"]):
                sparse_docs = self.sparse_retriever.get_relevant_documents(query, k=k)
                sparse_scores = [max(doc.metadata.get("bm25_score", 0.0), 0.01) for doc in sparse_docs]
                sparse_scores = self._normalize_scores(sparse_scores)
                for doc, score in zip(sparse_docs, sparse_scores):
                    doc.metadata["sparse_score"] = score

        # --- 3. Merge candidates ---
        combined = {doc.page_content: doc for doc in dense_docs}
        for doc in sparse_docs:
            if doc.page_content in combined:
                combined[doc.page_content].metadata["sparse_score"] = doc.metadata.get("sparse_score", 0.01)
            else:
                combined[doc.page_content] = doc
                doc.metadata["dense_score"] = 0.0
                doc.metadata["sparse_score"] = doc.metadata.get("sparse_score", 0.01)

        # --- 4. Compute hybrid score ---
        for doc in combined.values():
            dense = doc.metadata.get("dense_score", 0.0)
            sparse = doc.metadata.get("sparse_score", 0.0)
            doc.metadata["hybrid_score"] = self.dense_weight * dense + self.sparse_weight * sparse

        # --- 5. Optional reranking (top N) ---
        sorted_docs = sorted(combined.values(), key=lambda d: d.metadata["hybrid_score"], reverse=True)
        return sorted_docs[:k]

# --- Benchmark ---
def benchmark_hybrid(hybrid: HybridRetrieverOptimal, benchmark_data: List[Dict[str, str]], k: int = 20):
    total_time = 0.0
    total_similarity = 0.0
    valid_queries = 0

    for item in benchmark_data:
        query = item.get("query") or item.get("question") or item.get("prompt")
        answer = item.get("answer") or item.get("solution")
        if not query or not answer:
            continue

        valid_queries += 1
        start = time.time()
        docs = hybrid.get_relevant_documents(query, k=k)
        total_time += time.time() - start

        # Average cosine similarity across top-k docs
        ground_emb = hybrid.embedding_model.embed_documents([answer])[0]
        similarities = []
        for doc in docs:
            doc_emb = hybrid.embedding_model.embed_documents([doc.page_content])[0]
            sim = cos_sim(doc_emb, ground_emb).item()
            similarities.append(sim)
        avg_sim = sum(similarities) / len(similarities) if similarities else 0.0
        total_similarity += avg_sim

    avg_time = total_time / valid_queries if valid_queries else 0.0
    avg_cos_sim = total_similarity / valid_queries if valid_queries else 0.0

    print(f"\n========================= OPTIMAL HYBRID BENCHMARK =========================")
    print(f"Total Queries:          {valid_queries}")
    print(f"Average Retrieval Time: {avg_time:.3f} sec/query")
    print(f"Average Cosine Similarity (top {k} docs): {avg_cos_sim:.4f}")
    print(f"{'='*70}\n")

# --- Main ---
if __name__ == "__main__":
    benchmark_folder = r"D:\AGENTIC_AI\PROJECTS\hr-assistant\Benchmark"
    benchmark_data = load_all_benchmarks(benchmark_folder)

    folder_path = r"D:\AGENTIC_AI\PROJECTS\hr-assistant\data"
    hybrid = HybridRetrieverOptimal(dense_index_type="hnsw", sparse_docs_folder=folder_path, k=20)

    benchmark_hybrid(hybrid, benchmark_data, k=20)
