import os
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from langchain_community.vectorstores import Qdrant
from langchain_core.documents import Document
from dotenv import load_dotenv
from qdrant_client.models import (
    ScalarQuantization,
    ScalarQuantizationConfig,
    ScalarType,
)
load_dotenv()

QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_HOST = os.getenv("QDRANT_HOST")
QDRANT_PORT = 443  # default for cloud

client = QdrantClient(
    url=QDRANT_HOST,
    api_key=QDRANT_API_KEY,
)


def create_qdrant_collection(name: str, vector_size: int, distance: str = "Cosine", index_type: str = "flat"):
    """Create a Qdrant collection with a specific indexing strategy."""
    distance_mapping = {
        "Cosine": qmodels.Distance.COSINE,
        "Dot": qmodels.Distance.DOT,
        "Euclidean": qmodels.Distance.EUCLID
    }

    if index_type == "flat":
        index_config = qmodels.VectorParams(
            size=vector_size,
            distance=distance_mapping[distance],
        )
    elif index_type == "hnsw":
        index_config = qmodels.VectorParams(
            size=vector_size,
            distance=distance_mapping[distance],
            hnsw_config=qmodels.HnswConfigDiff(m=16, ef_construct=100)
        )
    elif index_type == "quantized":
        index_config = qmodels.VectorParams(
            size=vector_size,
            distance=distance_mapping[distance],
            quantization_config=ScalarQuantization(
                        scalar=ScalarQuantizationConfig(
                            type=ScalarType.INT8,
                            quantile=0.99,
                            always_ram=True
                )
            )
        )
    else:
        raise ValueError(f"❌ Invalid index type: {index_type}")

    client.recreate_collection(
        collection_name=name,
        vectors_config=index_config
    )
    print(f"✅ Created Qdrant collection: {name} ({index_type} index)")


def insert_documents(collection_name: str, chunks: list[Document], embedding_model):
    """Insert documents into a Qdrant collection."""
    Qdrant.from_documents(
        documents=chunks,
        embedding=embedding_model,
        url=QDRANT_HOST,
        api_key=QDRANT_API_KEY,
        collection_name=collection_name
    )
    print(f"📥 Inserted {len(chunks)} chunks into {collection_name}")
