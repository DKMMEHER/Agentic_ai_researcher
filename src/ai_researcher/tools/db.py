"""Vector Database configuration."""

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from ai_researcher.logging import get_logger

logger = get_logger(__name__)

_VECTOR_STORE = None
_EMBEDDINGS = None

def get_vector_store() -> Chroma:
    """Returns the singleton Chroma Vector Database initialized with Local HuggingFace embeddings."""
    global _VECTOR_STORE, _EMBEDDINGS
    if _VECTOR_STORE is None:
        logger.info("Initializing HuggingFace Embedding Model... (This takes a moment)")
        _EMBEDDINGS = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        _VECTOR_STORE = Chroma(
            collection_name="research_papers",
            embedding_function=_EMBEDDINGS,
            persist_directory="./.chroma_db"
        )
        logger.info("Vector database loaded successfully.")
    return _VECTOR_STORE
