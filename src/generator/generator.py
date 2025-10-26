from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
from typing import List, Dict, Any

class AnswerGenerator:
    def __init__(self, model_name: str = "google/flan-t5-large"):
        """Initialize the AnswerGenerator with a language model."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)
    
    def generate_answer(self, question: str, contexts: List[Dict[Any, Any]], max_length: int = 512) -> str:
        """Generate an answer based on the question and retrieved contexts."""
        # Format prompt to encourage specific step-by-step solutions
        context_text = "\n".join([ctx.get('text', '') for ctx in contexts])
        prompt = f"""Based on the context below, solve this equation step by step, showing each operation and its result:

Context:
{context_text}

Problem:
{question}

Let's solve this step by step:
1. First, identify the equation: {question.split(': ')[-1]}
2."""
        
        # Tokenize and generate
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True).to(self.device)
        outputs = self.model.generate(
            **inputs,
            max_length=max_length,
            num_beams=5,
            temperature=0.3,  # Lower temperature for more focused outputs
            top_p=0.95,
            no_repeat_ngram_size=3,
            length_penalty=1.0,  # Prefer longer, more detailed answers
            do_sample=True
        )
        
        # Decode and format answer
        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Clean up and format the answer
        if answer.startswith("1."):
            # Already numbered, just format it
            steps = answer.split("\n")
            answer = "\n".join(steps)
        else:
            # Add step-by-step formatting
            steps = answer.split(". ")
            formatted_steps = []
            for i, step in enumerate(steps, 1):
                if step:
                    formatted_steps.append(f"{i}. {step.strip()}")
            answer = "\n".join(formatted_steps)
        
        # Add conclusion if not present
        if "therefore" not in answer.lower() and "thus" not in answer.lower():
            # Extract the final value of x if possible
            import re
            x_value = re.search(r"x\s*=\s*(-?\d+(?:/\d+)?)", answer)
            if x_value:
                answer += f"\n\nTherefore, x = {x_value.group(1)}"
        
        return answer
        
    def batch_generate(self, questions: List[str], contexts_list: List[List[Dict[Any, Any]]], 
                      max_length: int = 512) -> List[str]:
        """Generate answers for multiple questions in batch."""
        return [
            self.generate_answer(q, c, max_length)
            for q, c in zip(questions, contexts_list)
        ]