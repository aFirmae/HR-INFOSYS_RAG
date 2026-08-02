import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import csv
import time
from sentence_transformers.util import cos_sim
from retrieval.retriever import get_retriever
from load import benchmark_data
from embedding.embeddings import embedding_model
from Reranker.reranker import bm25_rerank

def benchmark_retriever(retriever, benchmark_data, k=10, output_csv="benchmark_results.csv", sim_threshold=0.3):
    total_time = 0
    recall_count = 0
    mrr_sum = 0
    similarity_sum = 0

    results = []

    for i, item in enumerate(benchmark_data, 1):
        query = item.get("question") or item.get("query") or item.get("prompt")
        ground_truth = item["answer"]

        # Measure retrieval time
        start_time = time.time()
        docs = retriever.get_relevant_documents(query)
        elapsed = time.time() - start_time
        total_time += elapsed

        # Rerank using BM25
        docs = bm25_rerank(query, docs, top_n=k)

        embedding_gt = embedding_model.embed_documents([ground_truth])[0]

        sims = []
        max_sim = 0
        max_sim_rank = 0

        # Track best rank by similarity
        for rank, doc in enumerate(docs, 1):
            embedding_doc = embedding_model.embed_documents([doc.page_content])[0]
            sim = cos_sim(embedding_gt, embedding_doc).item()
            sims.append(sim)
            if sim > max_sim:
                max_sim = sim
                max_sim_rank = rank

        # Track best rank by partial overlap (loosened)
        overlap_found = False
        overlap_rank = k + 1  # bigger than max rank to minimize

        gt_tokens = set(ground_truth.lower().split())
        for rank, doc in enumerate(docs, 1):
            doc_tokens = set(doc.page_content.lower().split())
            overlap = gt_tokens.intersection(doc_tokens)
            overlap_ratio = len(overlap) / max(1, len(gt_tokens))
            if overlap_ratio >= 0.3:
                overlap_found = True
                overlap_rank = rank
                break

        # Determine final hit & reciprocal rank by min rank from sim and overlap
        if max_sim >= sim_threshold and overlap_found:
            final_rank = min(max_sim_rank, overlap_rank)
            found = True
            rr = 1 / final_rank
        elif max_sim >= sim_threshold:
            found = True
            rr = 1 / max_sim_rank
        elif overlap_found:
            found = True
            rr = 1 / overlap_rank
        else:
            found = False
            rr = 0

        recall_count += int(found)
        mrr_sum += rr

        # Use max similarity as similarity score reported
        similarity_sum += max_sim

        results.append({
            "query": query,
            "retrieval_time": elapsed,
            "recall": int(found),
            "reciprocal_rank": rr,
            "similarity": max_sim
        })

    total_queries = len(benchmark_data)
    avg_time = total_time / total_queries
    recall_at_k = recall_count / total_queries
    mrr = mrr_sum / total_queries
    avg_similarity = similarity_sum / total_queries

    print("\n=== Benchmark Summary ===")
    print(f"Total Queries:      {total_queries}")
    print(f"Recall@{k}:         {recall_at_k:.4f}")
    print(f"MRR:                {mrr:.4f}")
    print(f"Avg Similarity:     {avg_similarity:.4f}")
    print(f"Avg Retrieval Time: {avg_time:.4f} sec")

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["query", "retrieval_time", "recall", "reciprocal_rank", "similarity"])
        writer.writeheader()
        writer.writerows(results)

    print(f"\n✅ Results saved to {output_csv}")


# Usage
dense_retriever = get_retriever("hnsw", k=10)

benchmark_retriever(dense_retriever, benchmark_data, k=10, output_csv="benchmark_results.csv", sim_threshold=0.3)
