"""
Enhanced Math Solver CLI
Run with: python -m src.main solve "your equation here"
"""

import sys
import logging
import argparse
from src.pipeline.enhanced_pipeline import EnhancedPipeline

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_argparse():
    parser = argparse.ArgumentParser(description='Enhanced Math Problem Solver')
    parser.add_argument('command', choices=['solve'], help='Command to execute')
    parser.add_argument('equation', help='Math equation to solve')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    return parser

def check_dependencies():
    """Check if all required dependencies are available"""
    try:
        import torch
        import transformers
        import sentence_transformers
        import faiss
        import sympy
        return True
    except ImportError as e:
        logger.error(f"Missing required dependency: {str(e)}")
        logger.error("Please install all dependencies: pip install -r requirements.txt")
        return False

def main():
    parser = setup_argparse()
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Verify dependencies first
    if not check_dependencies():
        return 1
    
    try:
        logger.debug("Initializing pipeline...")
        pipeline = EnhancedPipeline()
        
        if args.command == 'solve':
            logger.info(f"Processing equation: {args.equation}")
            
            # Basic equation validation
            if '=' not in args.equation:
                logger.error("Invalid equation format. Must contain '='")
                return 1
            
            result = pipeline.process_question(args.equation)
            
            if "error" in result:
                logger.error(result["error"])
                return 1
                
            # Display solution
            print("\nSolution:")
            print("=========")
            for i, step in enumerate(result["solution"]["steps"], 1):
                print(f"Step {i}: {step}")
            
            print("\nFinal Answer:")
            print("============")
            print(result["solution"]["final_answer"])
            
            print(f"\nConfidence: {result['solution']['confidence']:.2%}")
            
            if result["solution"].get("requires_human_review", False):
                print("\nNote: This solution requires human verification.")
                
            return 0
            
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main())