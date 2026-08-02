from rank_bm25 import BM25Okapi
from typing import List
from langchain_core.documents import Document
import logging
from dotenv import load_dotenv
load_dotenv()

# Optional LangSmith tracing decorator
try:
    from Tracing.Tracing import traceable
except ImportError:
    def traceable(name=None):
        def decorator(func):
            return func
        return decorator

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@traceable(name="BM25 Reranker")
def bm25_rerank(query: str, documents: List[Document], top_n: int = 10) -> List[Document]:
    """
    Reranks retrieved documents using BM25 keyword relevance.

    Args:
        query (str): User input or question.
        documents (List[Document]): List of documents from vector search.
        top_n (int): Number of top relevant docs to return.

    Returns:
        List[Document]: Top-N reranked documents based on BM25 score.
    """
    if not documents:
        logger.warning("No documents to rerank.")
        return []

    logger.info(f"Starting BM25 rerank | Query: {query}")

    # Simple tokenizer using split() for speed & no external dependency
    corpus = [doc.page_content.lower().split() for doc in documents]
    tokenized_query = query.lower().split()

    # Build and score with BM25
    bm25 = BM25Okapi(corpus)
    scores = bm25.get_scores(tokenized_query)

    # Sort and select top_n
    ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    reranked_docs = [documents[i] for i in ranked_indices[:top_n]]

    # Log results
    for idx, i in enumerate(ranked_indices[:top_n]):
        logger.info(f"[{idx+1}] Score: {scores[i]:.4f} | Doc: {documents[i].page_content[:80]}...")

    return reranked_docs
