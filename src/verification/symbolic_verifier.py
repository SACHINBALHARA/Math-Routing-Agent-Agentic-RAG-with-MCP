from sympy import symbols, solve, Eq, sympify, SympifyError
from typing import Tuple, Optional, List
import re

class SymbolicVerifier:
    def __init__(self):
        self.supported_operations = ['solve', 'simplify', 'expand', 'factor']
        
    def verify_equation_solution(self, equation: str, solution: dict) -> Tuple[bool, Optional[str]]:
        """
        Verify if the solution satisfies the equation
        
        Args:
            equation: String representation of the equation (e.g., "2*x + 3 = 11")
            solution: Dictionary of variable assignments (e.g., {"x": 4})
            
        Returns:
            Tuple of (is_correct: bool, error_message: Optional[str])
        """
        try:
            # Parse equation into left and right sides
            left_side, right_side = self._parse_equation(equation)
            
            # Convert to SymPy expressions
            x = symbols('x')  # Currently supporting single variable 'x'
            left_expr = sympify(left_side)
            right_expr = sympify(right_side)
            
            # Substitute solution and evaluate
            solution_value = solution.get('x')
            if solution_value is None:
                return False, "No solution provided for variable x"
                
            left_val = left_expr.subs(x, solution_value)
            right_val = right_expr.subs(x, solution_value)
            
            # Check if equation balances
            is_correct = abs(left_val - right_val) < 1e-10  # Using small epsilon for float comparison
            
            if not is_correct:
                error_msg = f"Solution x = {solution_value} does not satisfy the equation. {left_val} ≠ {right_val}"
                return False, error_msg
                
            return True, None
            
        except SympifyError as e:
            return False, f"Error parsing mathematical expression: {str(e)}"
        except Exception as e:
            return False, f"Verification error: {str(e)}"

    def verify_solution_steps(self, steps: List[str], original_equation: str) -> Tuple[bool, Optional[str]]:
        """
        Verify that each step in the solution is mathematically valid
        
        Args:
            steps: List of solution steps
            original_equation: The original equation being solved
            
        Returns:
            Tuple of (all_steps_valid: bool, error_message: Optional[str])
        """
        try:
            current_equation = original_equation
            x = symbols('x')
            
            for step in steps:
                # Extract equation from step if present
                equation_match = re.search(r'([^=]+=[^=]+)', step)
                if not equation_match:
                    continue
                    
                new_equation = equation_match.group(1)
                
                # Verify this step maintains equality with previous
                if not self._verify_equivalent_equations(current_equation, new_equation):
                    return False, f"Invalid step: {step}"
                
                current_equation = new_equation
                
            return True, None
            
        except Exception as e:
            return False, f"Step verification error: {str(e)}"

    def _parse_equation(self, equation: str) -> Tuple[str, str]:
        """Parse equation string into left and right side expressions.

        This is more tolerant to user input that contains extra words like
        'find x' after the equation (e.g. '5x+10=50 find x'). We locate the
        central '=' and expand outward only over characters that look like
        math (digits, x, operators, parentheses, spaces, decimal points).
        """
        # Pre-clean common LaTeX or noise wrappers and unicode symbols
        eq = equation
        eq = eq.replace('$', ' ')
        eq = re.sub(r'\\\(|\\\)', ' ', eq)  # remove \( \)
        eq = eq.replace('\n', ' ')
        eq = eq.replace(',', '')
        # Replace Unicode minus with ASCII
        eq = eq.replace('\u2212', '-')

        # Replace LaTeX fractions \frac{a}{b} -> (a)/(b) to make sympify happier
        eq = re.sub(r'\\frac\{\s*([-+]?\d+)\s*\}\{\s*(\d+)\s*\}', r'(\1)/(\2)', eq)

        # Prefer a strict regex capturing only math-like characters around '='
        # This avoids pulling trailing words like 'find x' into the equation sides.
        math_side_pattern = re.compile(r'([0-9a-zA-Z\._\s\+\-\*/\^()\\/]+)=([0-9a-zA-Z\._\s\+\-\*/\^()\\/]+)')
        m = math_side_pattern.search(eq)
        if m:
            left_part = m.group(1).strip()
            right_part = m.group(2).strip()
            return self._normalize_expr(left_part), self._normalize_expr(right_part)

        # Fallback: split on the first '=' and strip non-math characters more aggressively
        if '=' not in eq:
            raise ValueError("Invalid equation format. Must contain '='")

        parts = eq.split('=', 1)
        left_raw, right_raw = parts[0].strip(), parts[1].strip()

        # Remove obvious trailing words like 'find x', 'solve for x'
        right_raw = re.sub(r'\b(find|solve|for|compute|what|calculate)\b', ' ', right_raw, flags=re.I)

        # Remove any non-math characters from sides but keep variable names
        sanitize = lambda s: re.sub(r'[^0-9a-zA-Z\.\+\-\*/\^() \\_/]+', ' ', s)
        left_clean = sanitize(left_raw)
        right_clean = sanitize(right_raw)

        # If still no digits found on either side, raise
        if not (re.search(r'\d', left_clean) and re.search(r'\d', right_clean)):
            raise ValueError("Could not parse a valid numeric equation from input")

        return self._normalize_expr(left_clean), self._normalize_expr(right_clean)

    def _normalize_expr(self, expr: str) -> str:
        """Normalize expression to be SymPy-friendly (insert * between number and variable, remove extra spaces)."""
        # Insert multiplication symbol between digit and variable, e.g., '2x' -> '2*x'
        expr = re.sub(r"(\d)\s*([a-zA-Z])", r"\1*\2", expr)
        # Insert multiplication between variable and variable or variable and parenthesis: 'x(' -> 'x*(' or 'xy' -> 'x*y'
        expr = re.sub(r"([a-zA-Z])\s*([a-zA-Z])", r"\1*\2", expr)
        expr = re.sub(r"([a-zA-Z0-9\)])\s*\(", r"\1*(", expr)
        return expr

    def _verify_equivalent_equations(self, eq1: str, eq2: str) -> bool:
        """Verify that two equations are mathematically equivalent"""
        try:
            x = symbols('x')
            left1, right1 = self._parse_equation(eq1)
            left2, right2 = self._parse_equation(eq2)
            
            # Convert to SymPy expressions
            expr1 = Eq(sympify(left1), sympify(right1))
            expr2 = Eq(sympify(left2), sympify(right2))
            
            # Solve both equations
            sol1 = solve(expr1)
            sol2 = solve(expr2)

            # Normalize solutions to floats and compare with tolerance
            def normalize(solutions):
                vals = []
                for s in solutions:
                    try:
                        vals.append(float(s))
                    except Exception:
                        # fallback: use sympy N conversion
                        try:
                            vals.append(float(s.evalf()))
                        except Exception:
                            pass
                return set(round(v, 8) for v in vals)

            return normalize(sol1) == normalize(sol2)
            
        except Exception:
            return False  # If we can't verify equivalence, assume they're not equivalent