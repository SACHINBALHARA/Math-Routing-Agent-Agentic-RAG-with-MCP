from typing import List, Dict, Any, Union
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from bert_score import score as bert_score
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGEvaluator:
    def __init__(self):
        """
        Initialize the RAG system evaluator with various metrics.
        """
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        self.smoothing = SmoothingFunction().method1
        
        # Load semantic similarity model for answer relevance
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.sim_model_name = "cross-encoder/stsb-roberta-large"
        self.sim_model = AutoModelForSequenceClassification.from_pretrained(self.sim_model_name).to(self.device)
        self.sim_tokenizer = AutoTokenizer.from_pretrained(self.sim_model_name)

    def evaluate_retrieval(
        self,
        queries: List[str],
        retrieved_docs: List[List[Dict[str, Any]]],
        relevant_docs: List[List[Dict[str, Any]]],
    ) -> Dict[str, float]:
        """
        Evaluate retrieval performance using various metrics.
        
        Args:
            queries: List of query strings
            retrieved_docs: List of lists of retrieved documents
            relevant_docs: List of lists of relevant (ground truth) documents
            
        Returns:
            Dictionary containing various retrieval metrics
        """
        metrics = {
            'precision@k': [],
            'recall@k': [],
            'mrr': [],  # Mean Reciprocal Rank
            'ndcg': []  # Normalized Discounted Cumulative Gain
        }
        
        for query_retrieved, query_relevant in zip(retrieved_docs, relevant_docs):
            # Get IDs of retrieved and relevant documents
            retrieved_ids = [doc.get('id') for doc in query_retrieved]
            relevant_ids = [doc.get('id') for doc in query_relevant]
            
            # Calculate Precision@K
            k = len(query_retrieved)
            relevant_retrieved = set(retrieved_ids) & set(relevant_ids)
            precision = len(relevant_retrieved) / k if k > 0 else 0
            metrics['precision@k'].append(precision)
            
            # Calculate Recall@K
            recall = len(relevant_retrieved) / len(relevant_ids) if relevant_ids else 0
            metrics['recall@k'].append(recall)
            
            # Calculate MRR
            mrr = 0
            for rank, doc_id in enumerate(retrieved_ids, 1):
                if doc_id in relevant_ids:
                    mrr = 1 / rank
                    break
            metrics['mrr'].append(mrr)
            
            # Calculate NDCG
            dcg = 0
            idcg = 0
            for i, doc_id in enumerate(retrieved_ids):
                rel = 1 if doc_id in relevant_ids else 0
                dcg += rel / np.log2(i + 2)
            for i in range(min(k, len(relevant_ids))):
                idcg += 1 / np.log2(i + 2)
            ndcg = dcg / idcg if idcg > 0 else 0
            metrics['ndcg'].append(ndcg)
        
        # Average all metrics
        return {
            'precision@k': np.mean(metrics['precision@k']),
            'recall@k': np.mean(metrics['recall@k']),
            'mrr': np.mean(metrics['mrr']),
            'ndcg': np.mean(metrics['ndcg'])
        }

    def evaluate_generation(
        self,
        generated_answers: List[str],
        reference_answers: List[str]
    ) -> Dict[str, Union[float, Dict[str, float]]]:
        """
        Evaluate generation quality using multiple metrics.
        
        Args:
            generated_answers: List of generated answers
            reference_answers: List of reference (ground truth) answers
            
        Returns:
            Dictionary containing various generation quality metrics
        """
        metrics = {
            'rouge': {'rouge1': [], 'rouge2': [], 'rougeL': []},
            'bleu': [],
            'bert_score': {'precision': [], 'recall': [], 'f1': []},
            'semantic_similarity': []
        }
        
        # Calculate ROUGE scores
        for gen, ref in zip(generated_answers, reference_answers):
            rouge_scores = self.rouge_scorer.score(gen, ref)
            for key in ['rouge1', 'rouge2', 'rougeL']:
                metrics['rouge'][key].append(rouge_scores[key].fmeasure)
        
        # Calculate BLEU scores
        for gen, ref in zip(generated_answers, reference_answers):
            bleu_score = sentence_bleu(
                [ref.split()],
                gen.split(),
                smoothing_function=self.smoothing
            )
            metrics['bleu'].append(bleu_score)
        
        # Calculate BERTScore
        if torch.cuda.is_available():
            precision, recall, f1 = bert_score(
                generated_answers,
                reference_answers,
                lang='en',
                device=self.device
            )
            metrics['bert_score']['precision'] = precision.tolist()
            metrics['bert_score']['recall'] = recall.tolist()
            metrics['bert_score']['f1'] = f1.tolist()
        
        # Calculate semantic similarity
        for gen, ref in tqdm(zip(generated_answers, reference_answers), desc="Computing semantic similarity"):
            inputs = self.sim_tokenizer(
                gen,
                ref,
                return_tensors="pt",
                truncation=True,
                padding=True
            ).to(self.device)
            
            with torch.no_grad():
                similarity_score = torch.sigmoid(self.sim_model(**inputs).logits).item()
            metrics['semantic_similarity'].append(similarity_score)
        
        # Aggregate metrics
        return {
            'rouge': {k: np.mean(v) for k, v in metrics['rouge'].items()},
            'bleu': np.mean(metrics['bleu']),
            'bert_score': {k: np.mean(v) for k, v in metrics['bert_score'].items()},
            'semantic_similarity': np.mean(metrics['semantic_similarity'])
        }

    def evaluate_pipeline(
        self,
        queries: List[str],
        pipeline_outputs: List[Dict[str, Any]],
        ground_truth: List[Dict[str, Any]]
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate the entire RAG pipeline.
        
        Args:
            queries: List of input queries
            pipeline_outputs: List of pipeline output dictionaries
            ground_truth: List of ground truth dictionaries
            
        Returns:
            Dictionary containing both retrieval and generation metrics
        """
        # Extract relevant information for evaluation
        retrieved_docs = [output['retrieved_contexts'] for output in pipeline_outputs]
        generated_answers = [output['answer'] for output in pipeline_outputs]
        relevant_docs = [gt['relevant_contexts'] for gt in ground_truth]
        reference_answers = [gt['answer'] for gt in ground_truth]
        
        # Evaluate both components
        retrieval_metrics = self.evaluate_retrieval(queries, retrieved_docs, relevant_docs)
        generation_metrics = self.evaluate_generation(generated_answers, reference_answers)
        
        return {
            'retrieval_metrics': retrieval_metrics,
            'generation_metrics': generation_metrics
        }