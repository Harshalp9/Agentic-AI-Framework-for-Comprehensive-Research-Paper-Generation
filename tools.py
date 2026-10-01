import json
import os
import random
import threading
import time

import requests
from dotenv import load_dotenv
from langchain.tools import tool

load_dotenv()

SEMANTIC_SCHOLAR_API_KEY = os.getenv("SEMANTIC_SCHOLAR_API_KEY")


_MIN_INTERVAL = float(
    os.getenv(
        "SEMANTIC_SCHOLAR_MIN_INTERVAL",
        "3.0" if SEMANTIC_SCHOLAR_API_KEY else "10.0",
    )
)
_rate_lock = threading.Lock()
_last_call_at = 0.0


def _throttle() -> None:
    global _last_call_at
    with _rate_lock:
        wait = _last_call_at + _MIN_INTERVAL - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        _last_call_at = time.monotonic()



# RETRY WITH BACKOFF ON 429 / 500

_MAX_RETRIES = 5
_BASE_BACKOFF = 8.0  # seconds


def _request_with_retry(url: str, params: dict, headers: dict) -> requests.Response:
    """
    GET with retry-with-backoff on 429 (rate limited) and 500 (Semantic
    Scholar's docs note this fires almost as often as 429 under load).
    Honors a Retry-After header when the server sends one. Backoff includes
    random jitter so multiple parallel callers hitting 429 together don't
    retry in lockstep and re-collide.

    400s are NOT retried (they're not transient), but the response body is
    logged so a bad param or an exhausted shared-pool rejection is visible
    instead of just "400 Client Error: Bad Request" with no context.
    """
    last_exc = None
    for attempt in range(1, _MAX_RETRIES + 1):
        _throttle()
        try:
            response = requests.get(url, params=params, headers=headers, timeout=10)
        except requests.RequestException as e:
            last_exc = e
            if attempt == _MAX_RETRIES:
                raise
            delay = _BASE_BACKOFF * (2 ** (attempt - 1)) + random.uniform(0, 2)
            time.sleep(delay)
            continue

        if response.status_code in (429, 500):
            if attempt == _MAX_RETRIES:
                response.raise_for_status()
            retry_after = response.headers.get("Retry-After")
            delay = (
                float(retry_after)
                if retry_after
                else _BASE_BACKOFF * (2 ** (attempt - 1)) + random.uniform(0, 2)
            )
            print(
                f"[search_semantic_scholar] {response.status_code} on attempt "
                f"{attempt}/{_MAX_RETRIES} — retrying in {delay:.1f}s"
            )
            time.sleep(delay)
            continue

        if response.status_code == 400:
            print(f"[search_semantic_scholar] 400 body: {response.text[:500]}")

        response.raise_for_status()
        return response

    if last_exc:
        raise last_exc
    raise RuntimeError("Semantic Scholar request failed after all retries.")


@tool
def search_semantic_scholar(query: str) -> str:
    """
    Search Semantic Scholar for relevant academic research papers.
    Returns a JSON object containing verifiable paper metadata. The JSON shape
    is deliberately preserved for the collector in paper.py; prose summaries
    cannot be safely turned into citations.
    """

    url = "https://api.semanticscholar.org/graph/v1/paper/search"

    params = {
        "query": query,
        "limit": 5,
        "fields": (
            "paperId,title,authors,abstract,year,url,externalIds,venue,"
            "publicationVenue,journal,publicationDate,"
            "citationCount,referenceCount,fieldsOfStudy,openAccessPdf"
        ),
    }

    headers = {}
    if SEMANTIC_SCHOLAR_API_KEY:
        headers["x-api-key"] = SEMANTIC_SCHOLAR_API_KEY

    try:
        response = _request_with_retry(url, params, headers)
        data = response.json()
        # Return original API fields instead of a lossy text rendering. This
        # lets the literature collector deduplicate by paperId/DOI and lets
        # the exporter produce real IEEE reference entries.
        return json.dumps({"data": data.get("data", [])}, ensure_ascii=False)

    except requests.RequestException as error:
        note = (
            " Consider requesting a free Semantic Scholar API key "
            "(https://www.semanticscholar.org/product/api#api-key-form) — "
            "unauthenticated search shares a very small global rate-limit "
            "pool, while a key gives you a dedicated 1 request/second."
            if not SEMANTIC_SCHOLAR_API_KEY
            else ""
        )
        return json.dumps(
            {"data": [], "error": f"Semantic Scholar search failed: {error}.{note}"}
        )