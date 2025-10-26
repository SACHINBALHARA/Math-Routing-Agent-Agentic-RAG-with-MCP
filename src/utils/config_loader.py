import os
from dotenv import load_dotenv

def load_config():
    """Load environment variables from .env file"""
    load_dotenv()
    
    return {
        'openai_api_key': os.getenv('OPENAI_API_KEY'),
        'hf_api_key': os.getenv('HF_API_KEY'),
        'pinecone_api_key': os.getenv('PINECONE_API_KEY'),
        'pinecone_environment': os.getenv('PINECONE_ENVIRONMENT', 'us-west1-gcp'),
        'model_name': os.getenv('MODEL_NAME', 'gpt-4'),
        'embedding_model': os.getenv('EMBEDDING_MODEL', 'sentence-transformers/all-MiniLM-L6-v2'),
        'vector_store': os.getenv('VECTOR_STORE', 'faiss'),  # or pinecone
    }