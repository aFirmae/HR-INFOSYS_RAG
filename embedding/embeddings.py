import os
from typing import List
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings


# Load environment variables
load_dotenv()
# Instantiate Hugging Face embedding model
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2", 
    model_kwargs={"device": "cpu"})

def embed_documents(docs: List[Document]) -> List[List[float]]:
    """Embeds a list of LangChain Document objects using Hugging Face Embeddings."""
    return embedding_model.embed_documents([doc.page_content for doc in docs])

def embed_query(query: str) -> List[float]:
    """Embeds a single query string using Hugging Face Embeddings."""
    return embedding_model.embed_query(query)


