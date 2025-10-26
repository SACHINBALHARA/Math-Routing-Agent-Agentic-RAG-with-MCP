"""Cross-attention based reranker for improving retrieval quality."""

import logging
import torch
from typing import List, Dict, Any
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from tqdm import tqdm

from src.utils.logging_config import setup_logging

logger = logging.getLogger(__name__)
setup_logging()

class CrossAttentionReranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """Initialize the reranker with a cross-encoder model."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading reranker model {model_name} on {self.device}")
        
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model.eval()
    
    def rerank(self, documents: List[Dict[str, Any]], query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Rerank documents based on relevance to query.
        
        Args:
            documents: List of documents from initial retrieval
            query: Query string
            top_k: Number of documents to return after reranking
            
        Returns:
            List of reranked documents with scores
        """
        if not documents:
            return []
            
        # Prepare inputs
        pairs = [(doc.get('text', ''), query) for doc in documents]
        
        # Encode and score
        with torch.no_grad():
            inputs = self.tokenizer(
                pairs,
                truncation=True,
                padding=True,
                return_tensors="pt",
                max_length=512
            ).to(self.device)
            
            scores = torch.sigmoid(self.model(**inputs).logits).squeeze(-1)
        
        # Sort documents by score
        scores = scores.cpu().numpy()
        scored_docs = [
            {**doc, 'score': float(score)}
            for doc, score in zip(documents, scores)
        ]
        scored_docs.sort(key=lambda x: x['score'], reverse=True)
        
        return scored_docs[:top_k]
        
    def batch_rerank(
        self,
        documents_list: List[List[Dict[str, Any]]],
        questions: List[str],
        top_k: int = 3
    ) -> List[List[Dict[str, Any]]]:
        """Rerank multiple sets of documents in batch.
        
        Args:
            documents_list: List of document lists from initial retrieval
            questions: List of query strings
            top_k: Number of documents to return per query
            
        Returns:
            List of reranked document lists
        """
        results = []
        for docs, question in tqdm(
            zip(documents_list, questions),
            total=len(questions),
            desc="Reranking documents"
        ):
            results.append(self.rerank(docs, question, top_k))
        return results