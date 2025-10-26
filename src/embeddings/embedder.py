from typing import List, Dict, Union
import numpy as np
from sentence_transformers import SentenceTransformer
from ..utils.config_loader import load_config
from ..utils.helpers import logger, timer_decorator

class MathEmbedder:
    def __init__(self):
        config = load_config()
        self.model_name = config['embedding_model']
        self.model = SentenceTransformer(self.model_name)
        logger.info(f"Initialized embedder with model: {self.model_name}")
        
    @timer_decorator
    def embed_text(self, text: str) -> np.ndarray:
        """Generate embedding for a single text"""
        try:
            return self.model.encode(text)
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            raise
            
    @timer_decorator
    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for a batch of texts"""
        try:
            return self.model.encode(texts, batch_size=32, show_progress_bar=True)
        except Exception as e:
            logger.error(f"Error generating batch embeddings: {str(e)}")
            raise
            
    def embed_kb_entry(self, entry: Dict) -> Dict:
        """Generate embeddings for a knowledge base entry"""
        try:
            # Embed question
            question_embedding = self.embed_text(entry['question_text'])
            
            # Embed each solution step
            step_embeddings = []
            for step in entry['canonical_solution_steps']:
                step_embedding = self.embed_text(step['content'])
                step_embeddings.append(step_embedding)
                
            # Update entry with embeddings
            entry['embedding'] = {
                'question': question_embedding.tolist(),
                'steps': [emb.tolist() for emb in step_embeddings]
            }
            
            return entry
            
        except Exception as e:
            logger.error(f"Error embedding KB entry: {str(e)}")
            raise
            
    @timer_decorator
    def embed_kb(self, kb_entries: List[Dict]) -> List[Dict]:
        """Generate embeddings for all entries in the knowledge base"""
        try:
            embedded_entries = []
            for entry in kb_entries:
                embedded_entry = self.embed_kb_entry(entry)
                embedded_entries.append(embedded_entry)
                
            logger.info(f"Successfully embedded {len(embedded_entries)} KB entries")
            return embedded_entries
            
        except Exception as e:
            logger.error(f"Error embedding knowledge base: {str(e)}")
            raise