import unittest
from src.gateway.ai_gateway import AIGateway, MCPOutput
from src.verification.symbolic_verifier import SymbolicVerifier

class TestAIGateway(unittest.TestCase):
    def setUp(self):
        self.gateway = AIGateway()
        
    def test_input_sanitization(self):
        # Test valid math question
        question = "Solve 2x + 3 = 11"
        sanitized, is_valid = self.gateway.sanitize_input(question)
        self.assertTrue(is_valid)
        self.assertEqual(sanitized, question)
        
        # Test question with PII
        question_with_pii = "Solve 2x + 3 = 11 for john@email.com"
        sanitized, is_valid = self.gateway.sanitize_input(question_with_pii)
        self.assertFalse(is_valid)
        
        # Test non-math question
        non_math = "What is the weather today?"
        sanitized, is_valid = self.gateway.sanitize_input(non_math)
        self.assertFalse(is_valid)
        
    def test_output_validation(self):
        # Test valid output
        valid_output = MCPOutput(
            final_answer="x = 4",
            steps=["$2x + 3 = 11$", "$2x = 8$", "$x = 4$"],
            citations=[{"source_type": "KB", "source_id": "123"}],
            confidence=0.9
        )
        validated, is_valid = self.gateway.validate_output(valid_output)
        self.assertTrue(is_valid)
        
        # Test low confidence output
        low_conf_output = MCPOutput(
            final_answer="x = 4",
            steps=["$2x + 3 = 11$", "$2x = 8$", "$x = 4$"],
            citations=[{"source_type": "KB", "source_id": "123"}],
            confidence=0.3
        )
        validated, is_valid = self.gateway.validate_output(low_conf_output)
        self.assertFalse(is_valid)

class TestSymbolicVerifier(unittest.TestCase):
    def setUp(self):
        self.verifier = SymbolicVerifier()
        
    def test_equation_verification(self):
        # Test correct solution
        is_correct, error = self.verifier.verify_equation_solution(
            "2*x + 3 = 11",
            {"x": 4}
        )
        self.assertTrue(is_correct)
        self.assertIsNone(error)
        
        # Test incorrect solution
        is_correct, error = self.verifier.verify_equation_solution(
            "2*x + 3 = 11",
            {"x": 5}
        )
        self.assertFalse(is_correct)
        self.assertIsNotNone(error)
        
    def test_solution_steps_verification(self):
        steps = [
            "2x + 3 = 11",
            "2x = 8",
            "x = 4"
        ]
        is_valid, error = self.verifier.verify_solution_steps(
            steps,
            "2x + 3 = 11"
        )
        self.assertTrue(is_valid)
        self.assertIsNone(error)

if __name__ == '__main__':
    unittest.main()