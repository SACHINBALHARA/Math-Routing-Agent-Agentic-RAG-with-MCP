import logging
from datetime import datetime
from typing import List, Dict

try:
    import requests
    from bs4 import BeautifulSoup
except Exception:
    requests = None
    BeautifulSoup = None

logger = logging.getLogger(__name__)


class WebSearcher:
    """Simple web search abstraction.

    This class attempts to perform a lightweight web search and return a list
    of snippet dicts suitable for inclusion in an MCP retrieval block.

    Implementation notes:
    - If `SERPAPI_KEY` or other paid service is preferred, replace the
      implementation here with a SerpAPI / Bing search client.
    - This implementation uses DuckDuckGo HTML results via requests + BeautifulSoup
      when available. If those libraries are not installed, search() will return []
      and the pipeline will continue without web evidence.
    """

    def __init__(self, user_agent: str | None = None):
        self.user_agent = user_agent or "math-agent-web-search/1.0"

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Run a simple web search and return top_k results as dicts.

        Returned dict format (minimal):
        {"source_type": "web", "source_id": <url>, "snippet": <text>, "score": <float>, "extraction_timestamp": <iso>}
        """
        if requests is None or BeautifulSoup is None:
            logger.warning("requests/bs4 not available; web search skipped")
            return []

        try:
            headers = {"User-Agent": self.user_agent}
            resp = requests.post(
                "https://html.duckduckgo.com/html/",
                data={"q": query},
                headers=headers,
                timeout=10,
            )
            if resp.status_code != 200:
                logger.warning(f"DuckDuckGo HTML search returned {resp.status_code}")
                return []

            soup = BeautifulSoup(resp.text, "html.parser")
            results = []
            for a in soup.select("a.result__a")[:top_k]:
                url = a.get("href")
                text = a.get_text(strip=True)
                # Attempt to find a short snippet nearby
                snippet_tag = a.find_parent()
                snippet = text
                if snippet_tag:
                    snippet = snippet_tag.get_text(separator=" ", strip=True)

                results.append(
                    {
                        "source_type": "web",
                        "source_id": url,
                        "snippet": snippet,
                        "score": 0.0,  # placeholder, pipeline can rerank
                        "extraction_timestamp": datetime.now().isoformat(),
                    }
                )

            return results
        except Exception as e:
            logger.exception("Error during web search: %s", e)
            return []
