"""
Advanced RAG System with AI Agents & Vector Databases

A comprehensive Retrieval-Augmented Generation system with
AI agents for intelligent processing with multiple vector database backends
"""

__version__ = "0.1.0"

# Core components
from .core.rag_engine import RAGEngine
from .core.retrieval_manager import RetrievalManager
from .core.generation_manager import GenerationManager

# Vector database implementations
from .vector_db.factory import VectorDBFactory
from .vector_db.base import VectorDBBase

# Embeddings
from .embeddings.embedding_manager import EmbeddingManager

# Retrievers
from .retrievers.hybrid_retriever import HybridRetriever

# Document processing
from .processors.document_processor import DocumentProcessor

# Utils
from .utils.config import RAGConfig as Config
from .utils.logger import logger

__all__ = [
    # Core
    "RAGEngine",
    "RetrievalManager",
    "GenerationManager",
    # Vector DB
    "VectorDBFactory",
    "VectorDBBase",
    # Embeddings
    "EmbeddingManager",
    # Retrievers
    "HybridRetriever",
    # Document processing
    "DocumentProcessor",
    # Utils
    "RAGConfig",
    "logger",
] 