import logging
import traceback
import uuid
from datetime import datetime
from typing import Dict, Any, List

from src.gateway.ai_gateway import (
    AIGateway, MCPHeader, MCPInput,
    RetrievalItem, ChainStep, MCPOutput
)
from src.gateway.mcp_persistence import save_mcp_envelope
from src.utils.config_loader import load_config
from src.web.web_search import WebSearcher
from src.retriever.retriever import FAISSRetriever
from src.reranker.reranker import CrossAttentionReranker
from src.generator.enhanced_generator import EnhancedGenerator
from src.verification.symbolic_verifier import SymbolicVerifier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedPipeline:
    def __init__(self, init_models: bool = True):
        """Initialize the pipeline.

        If init_models is False, avoid loading heavy models (useful for quick UI start or demos).
        Callers can instantiate with init_models=False and later create models if needed.
        """
        self.gateway = AIGateway()
        self.retriever = None
        self.reranker = None
        self.generator = None
        self.verifier = None
        self.web_searcher = None

        if init_models:
            self._init_models()

    def _init_models(self):
        """Load heavy model-backed components. Separated so callers can defer model loading."""
        # Avoid re-initializing if already present
        if self.retriever is None:
            self.retriever = FAISSRetriever()
        if self.reranker is None:
            self.reranker = CrossAttentionReranker()
        if self.generator is None:
            self.generator = EnhancedGenerator()
        if self.verifier is None:
            self.verifier = SymbolicVerifier()
        if self.web_searcher is None:
            self.web_searcher = WebSearcher()
        
    def process_question(self, question: str, user_profile: Dict[str, Any] = None, mock: bool = False) -> Dict[str, Any]:
        """
        Process a math question through the enhanced pipeline with MCP tracking
        """
        # Generate request ID and create MCP header
        request_id = str(uuid.uuid4())
        header = MCPHeader(
            message_id=str(uuid.uuid4()),
            timestamp=datetime.now().isoformat(),
            agent_role="math_solver",
            model_name="enhanced_pipeline_v1",
            request_id=request_id
        )
        
        # Input validation through AI Gateway
        sanitized_question, is_valid = self.gateway.sanitize_input(question)
        if not is_valid:
            return {
                "error": "Invalid input. Please provide a valid mathematical question without personal information."
            }
        
        try:
            # Create MCP input record
            input_data = MCPInput(
                user_query=sanitized_question,
                user_profile=user_profile,
                constraints={"max_steps": 5, "difficulty_level": "medium"}
            )

            # Fast mock path: return a deterministic, high-confidence example without loading models
            if mock:
                mock_output = MCPOutput(
                    final_answer="4",
                    steps=["2x + 3 = 11 -> 2x = 8", "x = 4"],
                    citations=[],
                    confidence=0.99
                )

                retrieval_items = [
                    RetrievalItem(
                        source_type="KB",
                        source_id="mock_kb_0",
                        snippet="Example: isolate x and solve",
                        similarity_score=1.0,
                        extraction_timestamp=datetime.now().isoformat()
                    )
                ]

                chain_steps = [
                    ChainStep(step_id="step_1", content="Isolate variable: 2x = 8", produced_by_model="mock", confidence=0.99),
                    ChainStep(step_id="step_2", content="Divide both sides: x = 4", produced_by_model="mock", confidence=0.99)
                ]

                # Create and persist MCP envelope for audit/demo
                mcp_envelope = self.gateway.create_mcp_envelope(
                    header=header,
                    input_data=input_data,
                    retrieval=retrieval_items,
                    chain_steps=chain_steps,
                    output=mock_output
                )
                try:
                    save_mcp_envelope(mcp_envelope)
                except Exception:
                    pass

                validated_output, is_valid = self.gateway.validate_output(mock_output)
                return {
                    "mcp_envelope": mcp_envelope,
                    "solution": {
                        "steps": validated_output.steps,
                        "final_answer": validated_output.final_answer,
                        "confidence": validated_output.confidence,
                        "requires_human_review": not is_valid,
                    }
                }
            
            # Retrieval phase
            logger.info(f"Processing question: {sanitized_question}")
            try:
                # Attempt to retrieve from KB
                retrieved_docs = self.retriever.retrieve(sanitized_question)
            except Exception:
                # Fallback: provide simple template contexts if retriever not configured
                retrieved_docs = [
                    {"content": "To solve linear equations, isolate x and perform inverse operations.", "score": 0.9},
                    {"content": "Subtract constants and divide by coefficients to find x.", "score": 0.8}
                ]
            logger.info(f"Retrieved {len(retrieved_docs)} documents")
            # Determine KB threshold from config (allow override via env/.env)
            config = load_config()
            KB_THRESHOLD = float(config.get('kb_threshold', 0.78))

            # Check maximum KB similarity score
            try:
                max_sim = max((doc.get("score", 0.0) for doc in retrieved_docs), default=0.0)
            except Exception:
                max_sim = 0.0

            web_results = []
            if max_sim < KB_THRESHOLD:
                logger.info("KB match below threshold (%.3f) — running web search", max_sim)
                try:
                    web_results = self.web_searcher.search(sanitized_question, top_k=5)
                except Exception as e:
                    logger.warning("Web search failed: %s", str(e))

            # Build combined candidate list (KB + web) for reranking
            combined_candidates = []
            # KB candidates first
            for i, doc in enumerate(retrieved_docs):
                combined_candidates.append({
                    'source_type': 'KB',
                    'source_id': f'doc_{i}',
                    'text': doc.get('content') or doc.get('snippet') or '',
                    'orig_score': float(doc.get('score', 0.0))
                })
            # Web candidates (if any)
            for i, wr in enumerate(web_results):
                combined_candidates.append({
                    'source_type': wr.get('source_type', 'web'),
                    'source_id': wr.get('source_id', f'web_{i}'),
                    'text': wr.get('snippet', ''),
                    'orig_score': float(wr.get('score', 0.0))
                })

            # Prepare list of texts for reranker
            texts_for_rerank = [c['text'] for c in combined_candidates if c['text']]

            # If there are no texts, keep empty lists
            if texts_for_rerank:
                # Rerank combined candidates and obtain scores
                try:
                    top_texts, top_scores = self.reranker.rerank(
                        query=sanitized_question,
                        documents=texts_for_rerank,
                        top_k=min(5, len(texts_for_rerank)),
                        return_scores=True
                    )
                except TypeError:
                    # Older reranker signature - fall back to no scores
                    top_texts = self.reranker.rerank(query=sanitized_question, documents=texts_for_rerank, top_k=min(5, len(texts_for_rerank)))
                    top_scores = [0.0] * len(top_texts)
                except Exception as e:
                    logger.warning("Reranker failed: %s", str(e))
                    top_texts, top_scores = [], []

                # Reconstruct reranked docs preserving source metadata
                retrieval_items = []
                reranked_docs = []
                for text, score in zip(top_texts, top_scores):
                    try:
                        idx = texts_for_rerank.index(text)
                    except ValueError:
                        # If not found, skip
                        continue
                    meta = combined_candidates[idx]
                    retrieval_items.append(
                        RetrievalItem(
                            source_type=meta['source_type'],
                            source_id=meta['source_id'],
                            snippet=meta['text'],
                            similarity_score=float(score),
                            extraction_timestamp=datetime.now().isoformat()
                        )
                    )
                    reranked_docs.append({
                        'source_type': meta['source_type'],
                        'source_id': meta['source_id'],
                        'snippet': meta['text'],
                        'score': float(score)
                    })
                logger.info(f"Reranked and kept top {len(reranked_docs)} documents (KB+web)")
            else:
                retrieval_items = []
                reranked_docs = []
            
            # Generation phase with enhanced generator
            chain_steps = []
            step_id = 1
            
            # Generate solution with LaTeX formatting and verification
            output = self.generator.generate_solution(sanitized_question, reranked_docs)
            
            # Record chain steps
            for step in output.steps:
                chain_steps.append(
                    ChainStep(
                        step_id=f"step_{step_id}",
                        content=step,
                        produced_by_model="enhanced_generator_v1",
                        confidence=output.confidence
                    )
                )
                step_id += 1
            
            # Verify solution
            if output.final_answer:
                verification_result, error_msg = self.verifier.verify_equation_solution(
                    sanitized_question,
                    {"x": self.generator._extract_numeric_answer(output.final_answer)}
                )
                if not verification_result:
                    logger.warning(f"Solution verification failed: {error_msg}")
                    output.confidence *= 0.5  # Reduce confidence if verification fails
            
            # Create complete MCP envelope
            mcp_envelope = self.gateway.create_mcp_envelope(
                header=header,
                input_data=input_data,
                retrieval=retrieval_items,
                chain_steps=chain_steps,
                output=output
            )

            # Persist MCP envelope for audit
            try:
                save_mcp_envelope(mcp_envelope)
            except Exception:
                logger.warning("Failed to persist MCP envelope")
            
            # Final validation through AI Gateway
            validated_output, is_valid = self.gateway.validate_output(output)
            if not is_valid:
                logger.warning("Output validation failed - marking for human review")
                mcp_envelope["mcp_evidence_flags"]["requires_human"] = True
            
            return {
                "mcp_envelope": mcp_envelope,
                "solution": {
                    "steps": validated_output.steps,
                    "final_answer": validated_output.final_answer,
                    "confidence": validated_output.confidence,
                    "requires_human_review": not is_valid,
                }
            }
            
        except Exception as e:
            tb = traceback.format_exc()
            logger.error(f"Error processing question: {str(e)}\n{tb}")
            return {
                "error": "An error occurred while processing your question",
                "details": str(e),
                "traceback": tb
            }