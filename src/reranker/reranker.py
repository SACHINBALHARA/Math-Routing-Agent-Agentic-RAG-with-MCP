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
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", device: str = None):
        """Initialize the reranker with a cross-encoder model.
        Backwards-compatible API: `rerank(query=..., documents=[str,...], top_k=..., return_scores=bool)`
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Loading reranker model {model_name} on {self.device}")

        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model.eval()
        self.batch_size = 8
    
    def rerank(self, query: str, documents: List[str], top_k: int = 3, return_scores: bool = False):
        """Rerank documents (strings) based on relevance to query.

        Args:
            query: Query string
            documents: List of document strings
            top_k: Number of documents to return after reranking
            return_scores: If True, also return scores list

        Returns:
            If return_scores is False: List[str] (top-k documents)
            If return_scores is True: (List[str], List[float])
        """
        if not documents:
            return ([] if not return_scores else ([], []))

        # Prepare pairs (query, doc)
        pairs = [(query, doc) for doc in documents]

        # Tokenize in batch
        all_scores = []
        with torch.no_grad():
            for i in range(0, len(pairs), self.batch_size):
                batch = pairs[i:i+self.batch_size]
                inputs = self.tokenizer(
                    batch,
                    truncation=True,
                    padding=True,
                    return_tensors="pt",
                    max_length=512
                ).to(self.device)

                logits = self.model(**inputs).logits
                scores = torch.sigmoid(logits).squeeze(-1).cpu().numpy()
                # If batch contains a single item, ensure it's iterable
                if scores.shape == (): 
                    scores = [float(scores)]
                all_scores.extend([float(s) for s in scores])

        # Pair scores with documents and sort
        scored = list(zip(documents, all_scores))
        scored.sort(key=lambda x: x[1], reverse=True)

        top_docs = [doc for doc, _ in scored[:top_k]]
        top_scores = [score for _, score in scored[:top_k]]

        if return_scores:
            return top_docs, top_scores
        return top_docs
        
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