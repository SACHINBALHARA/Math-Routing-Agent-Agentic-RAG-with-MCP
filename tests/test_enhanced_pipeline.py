import unittest
from src.pipeline.enhanced_pipeline import EnhancedPipeline

class TestEnhancedPipeline(unittest.TestCase):
    def setUp(self):
        self.pipeline = EnhancedPipeline()
        
    def test_simple_equation_solving(self):
        # Test basic equation
        question = "Solve 2x + 3 = 11"
        result = self.pipeline.process_question(question)
        
        self.assertNotIn("error", result)
        self.assertIn("solution", result)
        self.assertIn("mcp_envelope", result)
        
        solution = result["solution"]
        self.assertTrue(len(solution["steps"]) >= 3)
        self.assertIn("final_answer", solution)
        self.assertGreater(solution["confidence"], 0.5)
        
    def test_invalid_input_handling(self):
        # Test with PII
        question = "Solve 2x + 3 = 11 for john@email.com"
        result = self.pipeline.process_question(question)
        self.assertIn("error", result)
        
    def test_verification_integration(self):
        # Test equation with known solution
        question = "Solve: x/2 + 4 = 7"
        result = self.pipeline.process_question(question)
        
        self.assertNotIn("error", result)
        solution = result["solution"]
        
        # Solution should be x = 6
        final_answer = solution["final_answer"]
        self.assertIn("6", final_answer)
        
        # Check LaTeX formatting
        steps = solution["steps"]
        latex_formatted = any('$' in step for step in steps)
        self.assertTrue(latex_formatted)
        
    def test_complex_equation_handling(self):
        # Test more complex equation
        question = "Solve: 3x - 6 = 9"
        result = self.pipeline.process_question(question)
        
        self.assertNotIn("error", result)
        solution = result["solution"]
        
        # Verify MCP structure
        mcp = result["mcp_envelope"]
        self.assertIn("mcp_header", mcp)
        self.assertIn("mcp_chain_steps", mcp)
        self.assertIn("mcp_evidence_flags", mcp)
        
        # Check solution quality
        self.assertTrue(len(solution["steps"]) >= 3)
        self.assertGreater(solution["confidence"], 0.5)

if __name__ == '__main__':
    unittest.main()