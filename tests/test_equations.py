import sys
import os
import json
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipeline import RAGPipeline
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_example_data():
    """Load example equation data"""
    with open('data/examples/equations/simple_equations.json', 'r') as f:
        return json.load(f)

def test_equations():
    """
    Test the RAG pipeline with simple equations
    """
    logger.info("Loading example data...")
    data = load_example_data()
    
    logger.info("Initializing RAG Pipeline...")
    pipeline = RAGPipeline(
        retriever_k=5,    # Get top 5 documents initially
        reranker_k=2      # Rerank to top 2 most relevant documents
    )
    
    logger.info("\nProcessing equations one by one:")
    for example in data['examples']:
        question = example['question']
        expected = example['solution']
        
        logger.info(f"\nProcessing equation: {question}")
        result = pipeline.process_single_query(question, include_metadata=True)
        
        print(f"\nQuestion: {question}")
        print(f"Generated Answer: {result['answer']}")
        print(f"Expected Solution: {expected}")
        print("\nTop retrieved context used:")
        for i, ctx in enumerate(result['reranked_contexts'][:1], 1):
            print(f"Context {i}: {ctx.get('text', 'N/A')[:200]}...")
        print("\n" + "="*80)

if __name__ == "__main__":
    test_equations()