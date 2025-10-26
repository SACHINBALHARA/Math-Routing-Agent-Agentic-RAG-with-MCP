import sys
from typing import Dict, List
from datasets import load_dataset
from ..utils.helpers import create_kb_entry, save_json, logger

def load_gsm8k():
    """Load GSM8K dataset from HuggingFace"""
    try:
        dataset = load_dataset("openai/gsm8k", "main")
        
        # Process training data
        train_data = []
        for item in dataset['train']:
            entry = create_kb_entry(
                question=item['question'],
                solution_steps=item['answer'].split('\n'),  # Split answer into steps
                source='gsm8k',
                metadata={
                    'split': 'train',
                    'id': item['id'] if 'id' in item else None
                }
            )
            train_data.append(entry)
            
        # Process test data if available
        test_data = []
        if 'test' in dataset:
            for item in dataset['test']:
                entry = create_kb_entry(
                    question=item['question'],
                    solution_steps=item['answer'].split('\n'),
                    source='gsm8k',
                    metadata={
                        'split': 'test',
                        'id': item['id'] if 'id' in item else None
                    }
                )
                test_data.append(entry)
        
        return {
            'train': train_data,
            'test': test_data
        }
        
    except Exception as e:
        logger.error(f"Error loading GSM8K dataset: {str(e)}")
        raise

def load_math_qa():
    """Load MathQA dataset from HuggingFace"""
    try:
        dataset = load_dataset("MU-NLPC/Calc-math_qa")
        
        data = []
        for item in dataset['train']:  # Assuming we want to use the train split
            entry = create_kb_entry(
                question=item['question'],  # Using the correct field name
                solution_steps=[step.strip() for step in item['solution'].split('\n')] if 'solution' in item else [],
                source='math_qa',
                metadata={
                    'id': str(item['idx']) if 'idx' in item else None,  # Using idx field as id
                    'difficulty': item.get('level', 'medium'),  # Using level as difficulty
                    'topics': [item.get('category', 'general')],  # Using category as topic
                    'answer': item.get('answer', '')  # Storing the answer separately
                }
            )
            data.append(entry)
            
        return {'train': data}
        
    except Exception as e:
        logger.error(f"Error loading MathQA dataset: {str(e)}")
        raise

def merge_datasets(datasets: List[Dict]) -> Dict:
    """Merge multiple datasets into a unified knowledge base"""
    merged_data = {
        'train': [],
        'test': []
    }
    
    for dataset in datasets:
        if 'train' in dataset:
            merged_data['train'].extend(dataset['train'])
        if 'test' in dataset:
            merged_data['test'].extend(dataset['test'])
            
    return merged_data

def main():
    """Main function to load and process datasets"""
    try:
        # Load datasets
        gsm8k_data = load_gsm8k()
        math_qa_data = load_math_qa()
        
        # Merge datasets
        merged_data = merge_datasets([gsm8k_data, math_qa_data])
        
        # Save processed data
        save_json(merged_data['train'], 'data/processed/math_kb_train.json')
        if merged_data['test']:
            save_json(merged_data['test'], 'data/processed/math_kb_test.json')
            
        logger.info("Successfully processed and saved datasets")
        
    except Exception as e:
        logger.error(f"Error in data processing pipeline: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()