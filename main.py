import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ingest.loader import PDFStreamingLoader
from chunking.semantic_chunker import DocumentChunker
import os
from dotenv import load_dotenv
from vectorstore.qdrant_db import create_qdrant_collection, insert_documents
from config import QDRANT_COLLECTIONS
from embedding.embeddings import embedding_model
from Tracing.Tracing import trace_block, traceable_fn

load_dotenv()

# Activate LangSmith tracing if ENV is set
os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2", "false")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY", "")

@traceable_fn("run_ingestion_pipeline")
def run_ingestion_pipeline(folder_path: str):
    print("🚀 Starting document ingestion pipeline...")

    # Load PDFs (page-by-page)
    loader = PDFStreamingLoader(folder_path)
    documents = list(loader.stream_documents())
    print(f"📄 Loaded {len(documents)} pages from folder: {folder_path}")

    # Chunking
    chunker = DocumentChunker(chunk_size=512, chunk_overlap=64)
    chunks = chunker.chunk_documents(documents)
    print(f"🔗 Chunked into {len(chunks)} chunks.")

    vector_size = 384

    with trace_block():
        index_types = ["flat", "hnsw", "quantized"]
        for collection, index_type in zip(QDRANT_COLLECTIONS, index_types):
            create_qdrant_collection(collection, vector_size, index_type=index_type)
            insert_documents(collection, chunks, embedding_model)


if __name__ == "__main__":
    folder_path = os.path.join(os.path.dirname(__file__), "data")
    run_ingestion_pipeline(folder_path)












