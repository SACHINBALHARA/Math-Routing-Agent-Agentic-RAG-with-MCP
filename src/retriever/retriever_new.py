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
        model_name: str = 'sentence-transformers/all-MiniLM-L6-v2',
        index_path: str = None,
        documents_path: str = None
    ):
        """Initialize the retriever with the given model and optional index paths."""
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.documents = []
        
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
            
        # Embed query
        query_embedding = self.model.encode([query])[0].reshape(1, -1).astype('float32')
        
        # Search index
        distances, indices = self.index.search(query_embedding, k)
        
        # Prepare results
        results = []
        for distance, idx in zip(distances[0], indices[0]):
            if idx >= 0 and idx < len(self.documents):  # Valid index
                doc = self.documents[idx].copy()
                doc['score'] = float(1 / (1 + distance))  # Convert distance to similarity score
                results.append(doc)
        
        return results
    
    def batch_retrieve(self, queries: List[str], k: int = 5) -> List[List[Dict[str, Any]]]:
        """Retrieve documents for multiple queries in batch."""
        return [self.retrieve(query, k) for query in queries]