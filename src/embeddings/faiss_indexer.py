import os
import faiss
import numpy as np
from typing import List, Dict, Tuple
from ..utils.config_loader import load_config
from ..utils.helpers import logger, timer_decorator, save_json, load_json

class FaissIndexer:
    def __init__(self, dimension: int):
        """Initialize FAISS index
        
        Args:
            dimension: Dimensionality of the embeddings
        """
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)  # L2 distance for similarity
        self.id_to_entry = {}  # Map IDs to original KB entries
        logger.info(f"Initialized FAISS index with dimension {dimension}")
        
    @timer_decorator
    def add_entries(self, entries: List[Dict]) -> None:
        """Add entries to the FAISS index
        
        Args:
            entries: List of KB entries with embeddings
        """
        try:
            # Extract question embeddings
            embeddings = []
            for idx, entry in enumerate(entries):
                question_embedding = np.array(entry['embedding']['question'])
                embeddings.append(question_embedding)
                self.id_to_entry[idx] = entry
                
            # Add to FAISS index
            embeddings_array = np.array(embeddings).astype('float32')
            self.index.add(embeddings_array)
            
            logger.info(f"Added {len(entries)} entries to FAISS index")
            
        except Exception as e:
            logger.error(f"Error adding entries to FAISS index: {str(e)}")
            raise
            
    @timer_decorator
    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[Dict]:
        """Search for similar entries in the index
        
        Args:
            query_embedding: Embedding vector of the query
            k: Number of nearest neighbors to retrieve
            
        Returns:
            List of tuples (distance, entry)
        """
        try:
            # Reshape query embedding
            query_vector = query_embedding.reshape(1, -1).astype('float32')
            
            # Search index
            distances, indices = self.index.search(query_vector, k)
            
            # Get corresponding entries
            results = []
            for dist, idx in zip(distances[0], indices[0]):
                if idx >= 0:  # Valid index
                    entry = self.id_to_entry[idx]
                    results.append({
                        'distance': float(dist),
                        'entry': entry
                    })
                    
            return results
            
        except Exception as e:
            logger.error(f"Error searching FAISS index: {str(e)}")
            raise
            
    def save(self, directory: str) -> None:
        """Save the FAISS index and mappings
        
        Args:
            directory: Directory to save the index and mappings
        """
        try:
            os.makedirs(directory, exist_ok=True)
            
            # Save FAISS index
            index_path = os.path.join(directory, 'kb_index.faiss')
            faiss.write_index(self.index, index_path)
            
            # Save ID mappings
            mappings_path = os.path.join(directory, 'id_mappings.json')
            save_json(self.id_to_entry, mappings_path)
            
            logger.info(f"Saved FAISS index and mappings to {directory}")
            
        except Exception as e:
            logger.error(f"Error saving FAISS index: {str(e)}")
            raise
            
    @classmethod
    def load(cls, directory: str) -> 'FaissIndexer':
        """Load a saved FAISS index and mappings
        
        Args:
            directory: Directory containing the saved index and mappings
        """
        try:
            # Load FAISS index
            index_path = os.path.join(directory, 'kb_index.faiss')
            index = faiss.read_index(index_path)
            
            # Load ID mappings
            mappings_path = os.path.join(directory, 'id_mappings.json')
            id_to_entry = load_json(mappings_path)
            
            # Create instance and set attributes
            instance = cls(index.d)
            instance.index = index
            instance.id_to_entry = id_to_entry
            
            logger.info(f"Loaded FAISS index and mappings from {directory}")
            return instance
            
        except Exception as e:
            logger.error(f"Error loading FAISS index: {str(e)}")
            raise