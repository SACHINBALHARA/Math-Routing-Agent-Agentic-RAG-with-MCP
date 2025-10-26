"""FAISS-based retriever for efficient similarity search."""

import os
import json
import logging
import numpy as np
import faiss
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer

from src.utils.logging_config import setup_logging

# Set up logging
logger = logging.getLogger(__name__)
setup_logging()

class FAISSRetriever:
    def __init__(
        self,
        dimension: int = None,
        model_name: str = 'sentence-transformers/all-MiniLM-L6-v2',
        index_path: str = None,
        documents_path: str = None
    ):
        """Initialize the retriever. Backwards-compatible constructor:
        - If `dimension` is provided, create an empty FAISS index of that dim (used in tests)
        - Otherwise, initialize using a sentence-transformer model and optional prebuilt index
        """
        self.model_name = model_name
        self.model = None
        self.index = None
        self.documents = []
        self.texts = []
        self.dimension = dimension

        # If dimension provided, create a simple IndexFlatL2 index for tests
        if dimension is not None:
            import faiss as _faiss
            self.index = _faiss.IndexFlatL2(dimension)
            self.dimension = dimension
        else:
            # Initialize embedding model for text->vector retrieval
            self.model = SentenceTransformer(model_name)

        if index_path and documents_path:
            self.load_index(index_path, documents_path)
    
    def load_index(self, index_path: str, documents_path: str):
        """Load FAISS index and documents from files."""
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"Index file not found: {index_path}")
        if not os.path.exists(documents_path):
            raise FileNotFoundError(f"Documents file not found: {documents_path}")
            
        self.index = faiss.read_index(index_path)
        
        with open(documents_path, 'r') as f:
            self.documents = json.load(f)
        
        logger.info(f"Loaded index with {self.index.ntotal} vectors and {len(self.documents)} documents")
    
    def retrieve(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve the top-k most similar documents for a query."""
        if self.index is None:
            raise ValueError("No index available - build or load an index first")

        # If model-backed retriever (text queries)
        if self.model is not None:
            query_embedding = self.model.encode([query])[0].reshape(1, -1).astype('float32')
            distances, indices = self.index.search(query_embedding, k)

            results = []
            for distance, idx in zip(distances[0], indices[0]):
                if idx >= 0 and idx < len(self.documents):
                    doc = self.documents[idx].copy()
                    doc['score'] = float(1 / (1 + distance))
                    results.append(doc)
            return results

        # Otherwise not model-backed; return texts from index using a dummy approach (not used in tests)
        return []

    # Backwards-compatible methods for unit tests
    def add_texts(self, texts: List[str], vectors: np.ndarray):
        """Add texts and their vectors to the FAISS index (used by unit tests)."""
        if self.index is None:
            # create index based on vector dimension
            import faiss as _faiss
            self.dimension = vectors.shape[1]
            self.index = _faiss.IndexFlatL2(self.dimension)

        # add vectors
        self.index.add(vectors)
        # store texts
        self.texts.extend(texts)

    def search(self, query_vector: np.ndarray, k: int = 5) -> List[str]:
        """Search index with raw vector and return top-k texts (used by unit tests)."""
        if self.index is None or self.index.ntotal == 0:
            return []

        # ensure shape
        q = query_vector.reshape(1, -1).astype('float32')
        distances, indices = self.index.search(q, k)
        results = []
        for idx in indices[0]:
            if idx >= 0 and idx < len(self.texts):
                results.append(self.texts[idx])
        return results
    
    def batch_retrieve(self, queries: List[str], k: int = 5) -> List[List[Dict[str, Any]]]:
        """Retrieve documents for multiple queries in batch."""
        return [self.retrieve(query, k) for query in queries]