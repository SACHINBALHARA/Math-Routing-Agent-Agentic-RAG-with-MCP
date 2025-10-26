import os
import sys
from typing import Dict, List
from ..utils.config_loader import load_config
from ..utils.helpers import logger, save_json, load_json
from ..data_processing.load_datasets import load_gsm8k, load_math_qa, merge_datasets
from ..embeddings.embedder import MathEmbedder
from ..embeddings.faiss_indexer import FaissIndexer

def build_knowledge_base():
    """Build and index the knowledge base from datasets"""
    try:
        # 1. Load datasets
        logger.info("Loading datasets...")
        gsm8k_data = load_gsm8k()
        math_qa_data = load_math_qa()
        
        # 2. Merge datasets
        logger.info("Merging datasets...")
        merged_data = merge_datasets([gsm8k_data, math_qa_data])
        
        # 3. Initialize embedder
        logger.info("Initializing embedder...")
        embedder = MathEmbedder()
        
        # 4. Generate embeddings for training data
        logger.info("Generating embeddings for training data...")
        embedded_train = embedder.embed_kb(merged_data['train'])
        
        # 5. Save processed training data
        logger.info("Saving processed training data...")
        save_json(embedded_train, 'data/processed/math_kb_train_embedded.json')
        
        # 6. Initialize and build FAISS index
        logger.info("Building FAISS index...")
        # Get dimension from first embedding
        embedding_dim = len(embedded_train[0]['embedding']['question'])
        indexer = FaissIndexer(embedding_dim)
        indexer.add_entries(embedded_train)
        
        # 7. Save FAISS index
        logger.info("Saving FAISS index...")
        indexer.save('outputs/vectorstore')
        
        # 8. Process test data if available
        if merged_data['test']:
            logger.info("Processing test data...")
            embedded_test = embedder.embed_kb(merged_data['test'])
            save_json(embedded_test, 'data/processed/math_kb_test_embedded.json')
            
        logger.info("Knowledge base building completed successfully")
        
    except Exception as e:
        logger.error(f"Error building knowledge base: {str(e)}")
        sys.exit(1)

def verify_kb(kb_entries: List[Dict]) -> bool:
    """Verify the structure and content of knowledge base entries"""
    try:
        for entry in kb_entries:
            # Check required fields
            required_fields = ['question_text', 'canonical_solution_steps', 'embedding']
            for field in required_fields:
                if field not in entry:
                    raise ValueError(f"Missing required field: {field}")
                    
            # Verify embedding structure
            if not isinstance(entry['embedding'], dict):
                raise ValueError("Embedding must be a dictionary")
            if 'question' not in entry['embedding'] or 'steps' not in entry['embedding']:
                raise ValueError("Embedding must contain 'question' and 'steps'")
                
        return True
        
    except Exception as e:
        logger.error(f"Knowledge base verification failed: {str(e)}")
        return False

def main():
    """Main execution function"""
    try:
        # Create necessary directories
        os.makedirs('data/processed', exist_ok=True)
        os.makedirs('outputs/vectorstore', exist_ok=True)
        
        # Build knowledge base
        build_knowledge_base()
        
        # Verify the built knowledge base
        kb_train = load_json('data/processed/math_kb_train_embedded.json')
        if verify_kb(kb_train):
            logger.info("Knowledge base verification passed")
        else:
            logger.error("Knowledge base verification failed")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()