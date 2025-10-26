import re
import hashlib
from typing import Dict, Any, Optional
from dataclasses import dataclass
import json

@dataclass
class MCPHeader:
    message_id: str
    timestamp: str
    agent_role: str
    model_name: str
    request_id: str

@dataclass
class MCPInput:
    user_query: str
    user_profile: Optional[Dict[str, Any]] = None
    constraints: Optional[Dict[str, Any]] = None

@dataclass
class RetrievalItem:
    source_type: str  # 'KB' or 'web'
    source_id: str
    snippet: str
    similarity_score: float
    extraction_timestamp: str

@dataclass
class ChainStep:
    step_id: str
    content: str
    produced_by_model: str
    confidence: float

@dataclass
class MCPOutput:
    final_answer: str
    steps: list[str]
    citations: list[Dict[str, str]]
    confidence: float

class AIGateway:
    def __init__(self):
        self.pii_pattern = re.compile(r'\b[\w\.-]+@[\w\.-]+\.\w+\b|(?:\+\d{1,2}\s)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b')
        
    def sanitize_input(self, query: str) -> tuple[str, bool]:
        """Remove PII and validate input"""
        # Check for PII
        if self.pii_pattern.search(query):
            return "", False
            
        # Validate it's a math query (basic check)
        math_indicators = ['solve', 'calculate', 'evaluate', 'find', 'compute', 'prove',
                         '+', '-', '*', '/', '=', '<', '>', '∫', '∑', '∏', 'log', 'sin', 'cos']
        
        has_math = any(indicator in query.lower() for indicator in math_indicators)
        if not has_math:
            return "", False
            
        return query, True

    def validate_output(self, output: MCPOutput) -> tuple[MCPOutput, bool]:
        """Validate output format and content"""
        # Check confidence threshold
        if output.confidence < 0.5:
            output.steps.append("Note: This solution requires human verification due to low confidence.")
            return output, False
            
        # Ensure LaTeX formatting for mathematical expressions
        for i, step in enumerate(output.steps):
            output.steps[i] = self._ensure_latex_formatting(step)
            
        # Validate citations
        if output.citations:
            for citation in output.citations:
                if citation['source_type'] == 'web' and 'url' not in citation:
                    return output, False
                    
        return output, True

    def _ensure_latex_formatting(self, text: str) -> str:
        """Ensure mathematical expressions are in LaTeX format"""
        # Basic replacement of common math patterns
        patterns = [
            (r'(\d+)x', r'$\1x$'),
            (r'(\d+)/(\d+)', r'$\frac{\1}{\2}$'),
            (r'([+\-*/=])', r'$\1$'),
        ]
        
        result = text
        for pattern, replacement in patterns:
            result = re.sub(pattern, replacement, result)
        return result

    def create_mcp_envelope(self, 
                          header: MCPHeader,
                          input_data: MCPInput,
                          retrieval: list[RetrievalItem],
                          chain_steps: list[ChainStep],
                          output: MCPOutput) -> Dict[str, Any]:
        """Create the MCP envelope for the entire transaction"""
        return {
            "mcp_header": header.__dict__,
            "mcp_input": input_data.__dict__,
            "mcp_retrieval": [item.__dict__ for item in retrieval],
            "mcp_chain_steps": [step.__dict__ for step in chain_steps],
            "mcp_output": output.__dict__,
            "mcp_provenance": self._generate_provenance_hash(retrieval),
            "mcp_evidence_flags": {
                "verified": output.confidence > 0.8,
                "requires_human": output.confidence < 0.5
            }
        }

    def _generate_provenance_hash(self, retrieval: list[RetrievalItem]) -> str:
        """Generate a hash of the sources used"""
        # Deterministic cryptographic provenance hash using SHA-256 over
        # canonical JSON (sorted keys, compact separators). This is stable
        # across processes and runs and suitable for audit logging.
        try:
            items = [item.__dict__ for item in retrieval]
            canonical = json.dumps(items, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
            digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
            return digest
        except Exception:
            # Fallback: return a short sha1 of the string representation
            try:
                fallback = hashlib.sha1(json.dumps([item.__dict__ for item in retrieval]).encode("utf-8")).hexdigest()
                return fallback
            except Exception:
                return ""