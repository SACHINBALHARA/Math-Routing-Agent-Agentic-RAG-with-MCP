import os
import json
from datetime import datetime
from typing import Dict, Any


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_mcp_envelope(envelope: Dict[str, Any], out_dir: str = "logs") -> str:
    """Persist an MCP envelope to disk as a JSON-lines entry.

    Returns the path written.
    """
    ensure_dir(out_dir)
    ts = datetime.utcnow().isoformat().replace(':', '-')
    filename = os.path.join(out_dir, "mcp_envelopes.jsonl")
    try:
        with open(filename, 'a', encoding='utf-8') as fh:
            fh.write(json.dumps({"saved_at": datetime.utcnow().isoformat(), "envelope": envelope}, ensure_ascii=False))
            fh.write('\n')
        return filename
    except Exception:
        # best-effort: try to write a timestamped file
        try:
            alt = os.path.join(out_dir, f"mcp_envelope_{ts}.json")
            with open(alt, 'w', encoding='utf-8') as fh:
                json.dump(envelope, fh, ensure_ascii=False, indent=2)
            return alt
        except Exception:
            return ""
