from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langchain_huggingface import HuggingFaceEmbeddings
import os
from config import QDRANT_HOST, QDRANT_API_KEY, QDRANT_COLLECTIONS

# Initialize embedding model
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Qdrant Client
qdrant_client = QdrantClient(
    url=QDRANT_HOST,
    api_key=QDRANT_API_KEY
)

def get_retriever(index_type: str = "hnsw", k: int = 10):
    """
    Returns a retriever for a given Qdrant collection (index_type).
    """
    collection_name = None
    for name in QDRANT_COLLECTIONS:
        if index_type in name:
            collection_name = name
            break

    if collection_name is None:
        raise ValueError(f"No collection found for index_type='{index_type}'")

    vectorstore = QdrantVectorStore(
        client=qdrant_client,
        collection_name=collection_name,
        embedding=embedding_model
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    return retriever

