import json
import argparse
from src.pipeline.enhanced_pipeline import EnhancedPipeline


def run_demo(mock: bool = False):
    pipeline = EnhancedPipeline()
    question = "Solve for x: 2x + 3 = 11"
    result = pipeline.process_question(question, mock=mock)

    envelope = result.get('mcp_envelope')
    if envelope:
        # Print a sanitized, pretty JSON of the MCP envelope
        print(json.dumps(envelope, indent=2, ensure_ascii=False))
    else:
        print('No MCP envelope generated. Full result:')
        print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mock', action='store_true', help='Run demo in mock mode (fast, no models)')
    args = parser.parse_args()
    run_demo(mock=args.mock)
