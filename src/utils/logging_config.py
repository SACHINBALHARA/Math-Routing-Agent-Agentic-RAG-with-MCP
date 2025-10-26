import logging
import os

def setup_logging(log_dir: str = 'logs'):
    """
    Set up logging configuration.
    
    Args:
        log_dir: Directory to store log files
    """
    os.makedirs(log_dir, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, 'rag_system.log')),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)