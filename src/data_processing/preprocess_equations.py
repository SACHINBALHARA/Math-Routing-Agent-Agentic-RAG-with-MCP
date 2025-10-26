import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def preprocess_equations():
    """Preprocess equation examples and create FAISS index"""
    logger.info("Loading example data...")
    with open('data/examples/equations/simple_equations.json', 'r') as f:
        data = json.load(f)
    
    # Extract contexts and create documents
    documents = []
    for example in data['examples']:
        documents.append({
            'text': example['context'],
            'question': example['question'],
            'solution': example['solution']
        })
    
    # Initialize sentence transformer
    logger.info("Initializing sentence transformer...")
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    
    # Create embeddings
    logger.info("Creating embeddings...")
    texts = [doc['text'] for doc in documents]
    embeddings = model.encode(texts, convert_to_tensor=True)
    embeddings_np = embeddings.cpu().numpy()
    
    # Create FAISS index
    logger.info("Creating FAISS index...")
    dimension = embeddings_np.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings_np)
    
    # Save index and documents
    logger.info("Saving processed data...")
    os.makedirs('data/processed', exist_ok=True)
    faiss.write_index(index, 'data/processed/equations.index')
    
    with open('data/processed/documents.json', 'w') as f:
        json.dump(documents, f, indent=2)
    
    logger.info("Data preprocessing completed!")

if __name__ == "__main__":
    preprocess_equations()