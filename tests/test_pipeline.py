import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipeline import RAGPipeline
from src.evaluation import RAGEvaluator
import logging
import json
from typing import List, Dict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_test_data(file_path: str) -> List[Dict]:
    """Load test questions and their ground truth answers."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def run_pipeline_tests():
    """
    Run comprehensive tests on the RAG pipeline.
    """
    logger.info("Initializing RAG Pipeline...")
    pipeline = RAGPipeline(
        retriever_k=10,  # Get top 10 documents initially
        reranker_k=3     # Rerank to top 3 documents
    )
    
    # Test questions (mathematical problems)
    test_questions = [
        "Solve the quadratic equation: x² + 5x + 6 = 0",
        "What is the derivative of f(x) = x³ + 2x² - 4x + 1?",
        "Calculate the area of a circle with radius 5 units.",
        "Find the sum of the series: 1 + 2 + 3 + ... + 100"
    ]
    
    logger.info("Processing test questions...")
    try:
        # Process questions one by one first
        logger.info("\nTesting single query processing:")
        for question in test_questions:
            logger.info(f"\nProcessing question: {question}")
            result = pipeline.process_single_query(question, include_metadata=True)
            
            print(f"\nQuestion: {question}")
            print(f"Generated Answer: {result['answer']}")
            print("\nTop retrieved context:")
            for i, ctx in enumerate(result['reranked_contexts'][:2], 1):
                print(f"\n{i}. Relevance Score: {ctx.get('score', 'N/A')}")
                print(f"Context: {ctx.get('text', 'N/A')[:200]}...")
        
        # Test batch processing
        logger.info("\nTesting batch processing:")
        batch_results = pipeline.process_batch(
            questions=test_questions,
            batch_size=2,
            include_metadata=True
        )
        
        print("\nBatch Processing Results:")
        for result in batch_results:
            print(f"\nQuestion: {result['question']}")
            print(f"Generated Answer: {result['answer']}")
        
        # Optional: Test with evaluation metrics if ground truth is available
        evaluator = RAGEvaluator()
        
        # Example ground truth (in practice, this would come from your test dataset)
        example_ground_truth = [
            {
                "answer": "The solutions are x = -2 and x = -3",
                "relevant_contexts": [{"id": "1", "text": "To solve x² + 5x + 6 = 0..."}]
            },
            {
                "answer": "f'(x) = 3x² + 4x - 4",
                "relevant_contexts": [{"id": "2", "text": "The derivative of x³ is 3x²..."}]
            },
            {
                "answer": "A = πr² = 78.54 square units",
                "relevant_contexts": [{"id": "3", "text": "The area of a circle is given by πr²..."}]
            },
            {
                "answer": "The sum is 5050",
                "relevant_contexts": [{"id": "4", "text": "The sum of first n natural numbers..."}]
            }
        ]
        
        # Evaluate the pipeline results
        logger.info("\nEvaluating pipeline performance...")
        evaluation_results = evaluator.evaluate_pipeline(
            queries=test_questions,
            pipeline_outputs=batch_results,
            ground_truth=example_ground_truth
        )
        
        print("\nEvaluation Results:")
        print("\nRetrieval Metrics:")
        for metric, value in evaluation_results['retrieval_metrics'].items():
            print(f"{metric}: {value:.4f}")
        
        print("\nGeneration Metrics:")
        for metric_group, metrics in evaluation_results['generation_metrics'].items():
            if isinstance(metrics, dict):
                print(f"\n{metric_group}:")
                for metric, value in metrics.items():
                    print(f"  {metric}: {value:.4f}")
            else:
                print(f"{metric_group}: {metrics:.4f}")
                
    except Exception as e:
        logger.error(f"Error during testing: {str(e)}")
        raise

if __name__ == "__main__":
    run_pipeline_tests()