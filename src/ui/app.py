import os
import sys
import threading
import time
import json
import logging
from flask import Flask, render_template, request, redirect, url_for, jsonify

# Ensure project root is on sys.path so `from src...` imports work when running this file
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.pipeline.enhanced_pipeline import EnhancedPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__, template_folder="templates")

# instantiate pipeline once (defer heavy model loading unless requested)
pipeline = EnhancedPipeline(init_models=False)

# Global state for tracking model initialization
MODEL_STATUS = {
    'initialized': False,
    'last_error': None,
    'warmup_in_progress': False
}

def initialize_models():
    """Initialize models if not already done"""
    if not MODEL_STATUS['initialized'] and not MODEL_STATUS['warmup_in_progress']:
        try:
            pipeline._init_models()
            MODEL_STATUS['initialized'] = True
            MODEL_STATUS['last_error'] = None
        except Exception as e:
            MODEL_STATUS['last_error'] = str(e)
            logger.exception("Failed to initialize models")

def warmup_models(pipeline: EnhancedPipeline, queries: list[str] | None = None, timeout: int = 300):
    """Try to preload common models used by the pipeline to avoid reloader restarts and long first-request latency.

    This is best-effort: it will attempt safe calls to trigger model downloads/initialization and swallow errors.
    """
    logger.info("Starting model warmup (this may take a while on first run)...")
    # Update global status
    try:
        WARMUP_STATUS['in_progress'] = True
        WARMUP_STATUS['started_at'] = time.time()
    except Exception:
        pass
    start = time.time()
    if queries is None:
        queries = ["1+1", "2x+3=11"]

    try:
        # Retriever warmup attempts
        retr = getattr(pipeline, 'retriever', None)
        if retr is not None:
            # call a warmup method if present
            if hasattr(retr, 'warmup'):
                try:
                    retr.warmup()
                except Exception:
                    pass
            else:
                # try a harmless retrieve on a short query
                try:
                    retr.retrieve(queries[0])
                except Exception:
                    pass

        # Reranker warmup: run a tiny rerank to ensure model loads
        rer = getattr(pipeline, 'reranker', None)
        if rer is not None:
            try:
                rer.rerank(query=queries[0], documents=["sample doc"], top_k=1)
            except Exception:
                pass

        # Generator/verifier warmup: call generate_solution with a trivial query if available
        gen = getattr(pipeline, 'generator', None)
        if gen is not None:
            try:
                # Some generators may expect docs; pass empty list
                gen.generate_solution(queries[0], [])
            except Exception:
                pass

    except Exception as e:
        logger.warning("Model warmup encountered errors: %s", str(e))

    elapsed = time.time() - start
    logger.info("Model warmup finished (%.1fs)", elapsed)
    try:
        WARMUP_STATUS['in_progress'] = False
        WARMUP_STATUS['last_elapsed'] = elapsed
        WARMUP_STATUS['finished_at'] = time.time()
    except Exception:
        pass


# Warmup status (public endpoint)
WARMUP_STATUS = {
    'in_progress': False,
    'started_at': None,
    'finished_at': None,
    'last_elapsed': None
}


@app.route('/warmup_status', methods=['GET'])
def warmup_status():
    """Get the current status of model warmup"""
    return jsonify({
        'warmup': WARMUP_STATUS,
        'models_initialized': MODEL_STATUS['initialized'],
        'last_error': MODEL_STATUS['last_error']
    })

@app.route('/', methods=['GET'])
def index():
    """Render the main page"""
    return render_template('index.html', 
                         models_ready=MODEL_STATUS['initialized'],
                         model_error=MODEL_STATUS['last_error'])

@app.route('/solve', methods=['POST'])
def solve():
    """Handle problem solving requests"""
    question = request.form.get('question', '').strip()
    mock_mode = request.form.get('mock', 'false').lower() == 'true'
    
    if not question:
        return render_template('index.html', error="Please enter a math problem.")
    
    # Initialize models if needed and not in mock mode
    if not mock_mode and not MODEL_STATUS['initialized']:
        try:
            initialize_models()
        except Exception as e:
            return render_template('index.html', 
                                error=f"Failed to initialize models: {str(e)}",
                                question=question)
    
    try:
        # Process the question
        result = pipeline.process_question(question, mock=mock_mode)
        
        # Check for errors in result
        if 'error' in result:
            return render_template('index.html',
                                error=result.get('details', result['error']),
                                question=question)
        
        # Format MCP envelope for display
        mcp_json = None
        if 'mcp_envelope' in result:
            mcp_json = json.dumps(result['mcp_envelope'], indent=2, ensure_ascii=False)
        
        return render_template('index.html',
                             question=question,
                             solution=result.get('solution'),
                             mcp_json=mcp_json,
                             models_ready=MODEL_STATUS['initialized'])
    except Exception as e:
        logger.exception("Error processing question")
        return render_template('index.html',
                             error=f"An error occurred: {str(e)}",
                             question=question)

if __name__ == '__main__':
    # Configure the application
    app.debug = False
    app.env = 'production'

    # Get port from environment or use default
    port = int(os.environ.get('PORT', 5000))

    preload = os.getenv('PRELOAD_MODELS', os.getenv('PRELOAD', '0')).lower() in ('1', 'true', 'yes')
    preload_blocking = os.getenv('PRELOAD_BLOCKING', '0').lower() in ('1', 'true', 'yes')

    if preload:
        if preload_blocking:
            logger.info('PRELOAD_MODELS=true and PRELOAD_BLOCKING=true -> running blocking warmup')
            try:
                warmup_models(pipeline)
            except Exception:
                logger.exception('Warmup failed')
        else:
            # Start warmup in background so server starts immediately
            logger.info('PRELOAD_MODELS=true -> starting warmup in background')
            t = threading.Thread(target=warmup_models, args=(pipeline,), daemon=True)
            t.start()

    # Run Flask without auto-reloader to avoid restart loops when packages write files during init.
    app.run(host='127.0.0.1', port=port, debug=False, use_reloader=False)
