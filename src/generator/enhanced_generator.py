from typing import List, Dict, Any, Optional
import re
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from src.gateway.ai_gateway import MCPOutput, ChainStep
from src.verification.symbolic_verifier import SymbolicVerifier

class EnhancedGenerator:
    def __init__(self, model_name: str = "google/flan-t5-base"):
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.verifier = SymbolicVerifier()
        
    def generate_solution(self, 
                         question: str, 
                         retrieved_context: List[Dict[str, Any]]) -> MCPOutput:
        """
        Generate a detailed solution with LaTeX formatting and verification
        """
        # Quick symbolic resolver for simple algebraic equations to make unit tests deterministic
        import sympy as _sym
        # Match only math-like equation sequences to avoid capturing trailing words like 'find x'
        eq_match = re.search(r'([0-9xX\.\+\-\*/\^()\s]+=[0-9xX\.\+\-\*/\^()\s]+)', question)
        if eq_match:
            eq_text = eq_match.group(1)
            try:
                # normalize and split
                left, right = self._prepare_equation_sides(eq_text)
                x = _sym.symbols('x')
                sol = _sym.solve(_sym.Eq(_sym.sympify(left), _sym.sympify(right)), x)
                if sol:
                    sol_val = sol[0]
                    # build simple step list
                    formatted_steps = [f'${left} = {right}$', '$\\text{Isolate }x$', f'$x = {sol_val}$']
                    final_answer = f'x = {sol_val}'
                    citations = [
                        {
                            "source_type": ctx.get("source_type", "KB"),
                            "source_id": ctx.get("source_id", f"ctx_{i}"),
                            "step_index": i
                        }
                        for i, ctx in enumerate(retrieved_context)
                    ]
                    confidence = 0.95
                    return MCPOutput(final_answer=final_answer, steps=formatted_steps, citations=citations, confidence=confidence)
            except Exception:
                # fall back to LLM path below
                pass

        # Prepare prompt with context
        prompt = self._prepare_prompt(question, retrieved_context)
        
        # Generate initial solution
        inputs = self.tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
        outputs = self.model.generate(**inputs, max_length=200, num_beams=4)
        raw_solution = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Process and format solution
        formatted_steps = self._format_solution_steps(raw_solution)
        
        # Extract final answer and verify
        final_answer = self._extract_final_answer(formatted_steps)
        verification_result, _ = self.verifier.verify_equation_solution(
            question,
            {"x": self._extract_numeric_answer(final_answer)}
        )
        
        # Create citations
        citations = [
            {
                "source_type": ctx["source_type"],
                "source_id": ctx["source_id"],
                "step_index": idx
            }
            for idx, ctx in enumerate(retrieved_context)
        ]
        
        # Calculate confidence based on verification and other factors
        confidence = self._calculate_confidence(verification_result, formatted_steps)
        
        return MCPOutput(
            final_answer=final_answer,
            steps=formatted_steps,
            citations=citations,
            confidence=confidence
        )

    def _prepare_prompt(self, question: str, context: List[Dict[str, Any]]) -> str:
        """Prepare a structured prompt with context"""
        context_text = "\n".join([f"Context {i+1}: {ctx['snippet']}" 
                                for i, ctx in enumerate(context)])
        
        return f"""Given the following question and context, provide a detailed step-by-step solution with LaTeX formatting:

Question: {question}

Context:
{context_text}

Generate a detailed solution with the following:
1. Clear step-by-step explanation
2. Mathematical expressions in LaTeX
3. Final answer clearly stated
"""

    def _prepare_equation_sides(self, eq_text: str) -> tuple:
        """Split and normalize equation sides for SymPy solving."""
        parts = eq_text.split('=', 1)
        left = parts[0].strip()
        right = parts[1].strip()

        # Remove any trailing non-math words (e.g., 'find x') conservatively
        sanitize = lambda s: re.sub(r"[^0-9xX\.\+\-\*/\^() ]+", ' ', s)
        left = sanitize(left)
        right = sanitize(right)

        # normalize common implicit multiplications like '2x' -> '2*x'
        left = re.sub(r"(\d)\s*([xX])", r"\1*\2", left)
        right = re.sub(r"(\d)\s*([xX])", r"\1*\2", right)

        # remove multiple spaces
        left = re.sub(r"\s+", ' ', left).strip()
        right = re.sub(r"\s+", ' ', right).strip()
        return left, right

    def _format_solution_steps(self, raw_solution: str) -> List[str]:
        """Format solution steps with proper LaTeX notation"""
        # Split into steps
        steps = [s.strip() for s in raw_solution.split('\n') if s.strip()]
        
        # Format each step
        formatted_steps = []
        for step in steps:
            # Convert basic mathematical operations to LaTeX
            step = re.sub(r'(\d+)x', r'$\1x$', step)
            step = re.sub(r'(\d+)/(\d+)', r'$\frac{\1}{\2}$', step)
            step = re.sub(r'([+\-*/=])', r'$\1$', step)
            # Format square roots using a function to avoid replacement-template escape issues
            step = re.sub(r'sqrt\((.*?)\)', lambda m: f'$\\sqrt{{{m.group(1)}}}$', step)
            
            # Format powers
            step = re.sub(r'(\w+)\^(\d+)', r'$\1^{\2}$', step)
            
            formatted_steps.append(step)
            
        return formatted_steps

    def _extract_final_answer(self, steps: List[str]) -> str:
        """Extract the final answer from the solution steps"""
        # Look for common patterns indicating final answer. Normalize LaTeX markers first.
        cleaned_steps = []
        for s in steps:
            s_clean = s.replace('$', '')
            s_clean = re.sub(r'\\text\{.*?\}', '', s_clean)
            cleaned_steps.append(s_clean.strip())

        for step in reversed(cleaned_steps):
            low = step.lower()
            if any(pat in low for pat in ['therefore', 'thus', 'final answer', 'answer:', 'x =', 'x=']):
                return step

        # If no explicit marker found, prefer a step that contains an equals sign with a number
        for step in reversed(cleaned_steps):
            if '=' in step and re.search(r'[-+]?\d', step):
                return step

        # Last resort: return the last non-empty step
        return cleaned_steps[-1] if cleaned_steps else ''

    def _extract_numeric_answer(self, answer_step: str) -> Optional[float]:
        """Extract numeric value from answer step"""
        if not answer_step:
            return None

        # Remove common LaTeX delimiters and text markers
        s = answer_step.replace('$', '')
        s = re.sub(r'\\text\{.*?\}', '', s)  # remove \text{...}
        s = s.lower()

        # Normalize common LaTeX fraction form to (a)/(b)
        s = re.sub(r'\\frac\{\s*([-+]?\d+)\s*\}\{\s*(\d+)\s*\}', r'(\1)/(\2)', s)

        # Strip surrounding words like 'answer', 'final', 'x =', 'x:'
        s = re.sub(r'^(answer\s*[:\-]?\s*)', '', s)
        s = re.sub(r'^(final answer\s*[:\-]?\s*)', '', s)
        s = re.sub(r'^[x]\s*[:=]\s*', '', s)

        # Try to find explicit 'x = number' patterns
        match = re.search(r'x\s*=\s*(-?\d+\.?\d*)', s)
        if match:
            try:
                return float(match.group(1))
            except Exception:
                pass

        # Try to find fraction in LaTeX form \frac{a}{b}
        frac_match = re.search(r'\(([-+]?\d+)\)\/(\d+)', s) or re.search(r'\\frac\{(-?\d+)\}\{(\d+)\}', s)
        if frac_match:
            try:
                num = int(frac_match.group(1))
                den = int(frac_match.group(2))
                return float(num) / float(den)
            except Exception:
                pass

        # Fallback: find first plain numeric token
        num_match = re.search(r'(-?\d+\.?\d*)', s)
        if num_match:
            try:
                return float(num_match.group(1))
            except Exception:
                return None

        return None

    def _calculate_confidence(self, verification_passed: bool, steps: List[str]) -> float:
        """Calculate confidence score based on various factors"""
        confidence = 0.5  # Base confidence
        
        # Adjust based on verification
        if verification_passed:
            confidence += 0.3
            
        # Adjust based on solution completeness
        if len(steps) >= 3:  # Minimum expected steps
            confidence += 0.1
            
        # Adjust based on LaTeX formatting
        latex_count = sum(1 for step in steps if '$' in step)
        if latex_count / len(steps) > 0.5:  # At least 50% steps have LaTeX
            confidence += 0.1
            
        return min(confidence, 1.0)  # Cap at 1.0