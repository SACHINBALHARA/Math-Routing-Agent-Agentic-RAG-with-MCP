"""Script to prefetch and warm up models used by the pipeline.

Run this before starting the server if you want to download model weights once and avoid first-request latency.
"""
import time
import logging
import os
import sys

# Ensure project root on path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.pipeline.enhanced_pipeline import EnhancedPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('warmup')


def main(blocking: bool = True):
    pipeline = EnhancedPipeline()
    logger.info('Starting warmup: retriever/reranker/generator will be initialized')
    start = time.time()

    # Retriever
    try:
        retr = getattr(pipeline, 'retriever', None)
        if retr is not None:
            try:
                retr.retrieve('1+1')
            except Exception:
                pass
    except Exception as e:
        logger.warning('Retriever warmup error: %s', e)

    # Reranker
    try:
        rer = getattr(pipeline, 'reranker', None)
        if rer is not None:
            try:
                rer.rerank(query='1+1', documents=['sample doc'], top_k=1)
            except Exception:
                pass
    except Exception as e:
        logger.warning('Reranker warmup error: %s', e)

    # Generator
    try:
        gen = getattr(pipeline, 'generator', None)
        if gen is not None:
            try:
                gen.generate_solution('1+1', [])
            except Exception:
                pass
    except Exception as e:
        logger.warning('Generator warmup error: %s', e)

    elapsed = time.time() - start
    logger.info('Warmup finished in %.1fs', elapsed)


if __name__ == '__main__':
    main(blocking=True)
