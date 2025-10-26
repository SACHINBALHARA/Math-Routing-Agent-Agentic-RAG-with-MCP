from typing import List, Dict, Any, Optional
from ..retriever import FAISSRetriever
from ..reranker import CrossAttentionReranker
from ..generator import AnswerGenerator
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGPipeline:
    def __init__(
        self,
        retriever: Optional[FAISSRetriever] = None,
        reranker: Optional[CrossAttentionReranker] = None,
        generator: Optional[AnswerGenerator] = None,
        retriever_k: int = 20,
        reranker_k: int = 5
    ):
        """
        Initialize the RAG pipeline with retriever, reranker, and generator components.
        
        Args:
            retriever: FAISS retriever instance
            reranker: Cross-attention reranker instance
            generator: Answer generator instance
            retriever_k: Number of documents to retrieve initially
            reranker_k: Number of documents to keep after reranking
        """
        self.retriever = retriever or FAISSRetriever(index_path='data/processed/equations.index', documents_path='data/processed/documents.json')
        self.reranker = reranker or CrossAttentionReranker()
        self.generator = generator or AnswerGenerator()
        self.retriever_k = retriever_k
        self.reranker_k = reranker_k
        
    def process_single_query(
        self,
        question: str,
        include_metadata: bool = True
    ) -> Dict[str, Any]:
        """
        Process a single question through the complete pipeline.
        
        Args:
            question: The input question
            include_metadata: Whether to include retrieval/reranking metadata
            
        Returns:
            Dict containing generated answer and optional metadata
        """
        logger.info(f"Processing question: {question}")
        
        # Step 1: Initial retrieval
        retrieved_docs = self.retriever.retrieve(question, k=self.retriever_k)
        logger.info(f"Retrieved {len(retrieved_docs)} documents")
        
        # Step 2: Reranking
        reranked_docs = self.reranker.rerank(
            documents=retrieved_docs,
            query=question,
            top_k=self.reranker_k
        )
        logger.info(f"Reranked and kept top {len(reranked_docs)} documents")
        
        # Step 3: Answer generation
        answer = self.generator.generate_answer(
            question=question,
            contexts=reranked_docs
        )
        logger.info("Generated answer")
        
        # Prepare response
        result = {
            "question": question,
            "answer": answer
        }
        
        if include_metadata:
            result.update({
                "retrieved_contexts": retrieved_docs,
                "reranked_contexts": reranked_docs,
                "metadata": {
                    "retriever_k": self.retriever_k,
                    "reranker_k": self.reranker_k
                }
            })
            
        return result
    
    def process_batch(
        self,
        questions: List[str],
        batch_size: int = 8,
        include_metadata: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Process a batch of questions through the pipeline.
        
        Args:
            questions: List of input questions
            batch_size: Size of batches for processing
            include_metadata: Whether to include retrieval/reranking metadata
            
        Returns:
            List of dictionaries containing answers and optional metadata
        """
        results = []
        
        for i in tqdm(range(0, len(questions), batch_size), desc="Processing batches"):
            batch = questions[i:i + batch_size]
            
            # Step 1: Batch retrieval
            retrieved_batch = self.retriever.batch_retrieve(batch, k=self.retriever_k)
            
            # Step 2: Batch reranking
            reranked_batch = self.reranker.batch_rerank(
                documents_list=retrieved_batch,
                questions=batch,
                top_k=self.reranker_k
            )
            
            # Step 3: Batch answer generation
            answers = self.generator.batch_generate(
                questions=batch,
                contexts_list=reranked_batch
            )
            
            # Prepare batch results
            for q, a, r_docs, rr_docs in zip(batch, answers, retrieved_batch, reranked_batch):
                result = {
                    "question": q,
                    "answer": a
                }
                
                if include_metadata:
                    result.update({
                        "retrieved_contexts": r_docs,
                        "reranked_contexts": rr_docs,
                        "metadata": {
                            "retriever_k": self.retriever_k,
                            "reranker_k": self.reranker_k
                        }
                    })
                    
                results.append(result)
        
        return results