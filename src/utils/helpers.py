import logging
import time
import json
from typing import Any, Dict, List, Optional, Union
from sentence_transformers import SentenceTransformer
import torch
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/retrieval_logs.txt'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def embed_text(
    text: Union[str, List[str]],
    model_name: str = 'sentence-transformers/all-MiniLM-L6-v2'
) -> np.ndarray:
    """
    Embed text using a sentence transformer model.
    
    Args:
        text: Single string or list of strings to embed
        model_name: Name of the sentence transformer model to use
        
    Returns:
        numpy.ndarray: Text embeddings
    """
    model = SentenceTransformer(model_name)
    embeddings = model.encode(text, convert_to_tensor=True)
    return embeddings.cpu().numpy()

def timer_decorator(func):
    """Decorator to measure execution time of functions"""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logger.info(f"{func.__name__} took {end_time - start_time:.2f} seconds to execute")
        return result
    return wrapper

def save_json(data: Any, filepath: str) -> None:
    """Save data to JSON file"""
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def load_json(filepath: str) -> Any:
    """Load data from JSON file"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def normalize_math_notation(text: str) -> str:
    """Convert math notation to LaTeX format where possible"""
    # TODO: Implement math notation normalization
    return text

def format_solution_steps(steps: List[str]) -> List[Dict[str, str]]:
    """Format solution steps into structured format"""
    return [
        {
            'step_number': i + 1,
            'content': step.strip(),
            'explanation': '',  # To be filled by the agent
        }
        for i, step in enumerate(steps)
    ]

def create_kb_entry(
    question: str,
    solution_steps: List[str],
    difficulty: Optional[str] = None,
    topics: Optional[List[str]] = None,
    source: Optional[str] = None,
    metadata: Optional[Dict] = None
) -> Dict:
    """Create a structured knowledge base entry"""
    return {
        'question_text': normalize_math_notation(question),
        'canonical_solution_steps': format_solution_steps(solution_steps),
        'difficulty': difficulty or 'medium',
        'topic_tags': topics or [],
        'source': source,
        'metadata': metadata or {},
        'embedding': None  # To be filled by the embedding module
    }