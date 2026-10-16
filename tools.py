import json
import os
import random
import threading
import time
from pathlib import Path

import requests
from dotenv import load_dotenv
from langchain.tools import tool

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

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


def _render_architecture_diagram(spec: dict, output_path: str) -> str:
    """Render Qwen's component graph in a restrained, consistent style."""
    components = spec.get("components")
    connections = spec.get("connections")
    if not isinstance(components, list) or not 2 <= len(components) <= 10:
        raise ValueError("Qwen returned an invalid component list.")
    ids = [str(item.get("id", "")) for item in components if isinstance(item, dict)]
    if len(ids) != len(components) or len(set(ids)) != len(ids) or any(not item for item in ids):
        raise ValueError("Qwen returned invalid component identifiers.")
    if not isinstance(connections, list) or len(connections) > 18:
        raise ValueError("Qwen returned an invalid connection list.")
    if any(not isinstance(edge, dict) or edge.get("from") not in ids or edge.get("to") not in ids for edge in connections):
        raise ValueError("Qwen returned a connection to an unknown component.")

    incoming = {item: 0 for item in ids}
    outgoing = {item: [] for item in ids}
    for edge in connections:
        outgoing[edge["from"]].append(edge["to"])
        incoming[edge["to"]] += 1
    roots = [item for item in ids if incoming[item] == 0] or ids[:1]
    layers = {item: 0 for item in roots}
    for _ in ids:
        changed = False
        for source in ids:
            for target in outgoing[source]:
                if source in layers and layers.get(target, -1) < layers[source] + 1:
                    layers[target] = layers[source] + 1
                    changed = True
        if not changed:
            break
    for item in ids:
        layers.setdefault(item, max(layers.values(), default=0))

    groups: dict[int, list[str]] = {}
    for item in ids:
        groups.setdefault(layers[item], []).append(item)
    positions = {
        item: (column, (len(group) - 1) / 2 - row)
        for column, group in groups.items()
        for row, item in enumerate(group)
    }
    max_rows = max(map(len, groups.values()))
    fig, ax = plt.subplots(figsize=(max(8, len(groups) * 2.8), max(3.4, max_rows * 1.45)))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    for edge in connections:
        sx, sy = positions[edge["from"]]
        tx, ty = positions[edge["to"]]
        ax.add_patch(FancyArrowPatch(
            (sx + 0.62, sy), (tx - 0.62, ty), arrowstyle="-|>", mutation_scale=13,
            linewidth=1.1, color="#687589", connectionstyle="arc3,rad=0.04", zorder=1,
        ))
        label = str(edge.get("label", "")).strip()[:28]
        if label:
            ax.text((sx + tx) / 2, (sy + ty) / 2 + 0.12, label, ha="center", va="bottom", fontsize=7, color="#596579")
    for component in components:
        x, y = positions[component["id"]]
        box = FancyBboxPatch(
            (x - 0.62, y - 0.32), 1.24, 0.64,
            boxstyle="round,pad=0.04,rounding_size=0.07",
            facecolor="#f6f8fb", edgecolor="#9aa8ba", linewidth=1, zorder=2,
        )
        ax.add_patch(box)
        ax.text(x, y + 0.08, str(component.get("label", "Component"))[:42], ha="center", va="center",
                fontsize=8.2, weight="semibold", color="#243247", zorder=3)
        detail = str(component.get("detail", "")).strip()[:72]
        if detail:
            ax.text(x, y - 0.13, detail, ha="center", va="center", fontsize=6.5, color="#596579", zorder=3, wrap=True)
    ax.set_xlim(-0.85, max(groups) + 0.85)
    ax.set_ylim(-max_rows / 2 - 0.65, max_rows / 2 + 0.65)
    ax.axis("off")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def generate_qwen_architecture(content: str, output_path: str) -> str:
    """Use local Ollama/Qwen to extract a factual architecture graph and render it."""
    if not content.strip():
        return json.dumps({"ok": False, "error": "Proposed Work is empty."})
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
    model = os.getenv("QWEN_MODEL", "qwen2.5:3b-instruct").strip()
    prompt = (
        "Create an architecture diagram specification from the supplied Proposed Work. "
        "Return JSON only with components [{id,label,detail}] and connections [{from,to,label}]. "
        "Use 2 to 8 components. Include only explicitly described components and relationships; "
        "do not infer missing steps, data sources, technologies, or results. Keep labels concise; "
        "use empty detail or label strings when details are not stated.\n\n"
        f"Proposed Work:\n{content[:10000]}"
    )
    try:
        response = requests.post(
            f"{base_url}/api/chat",
            json={
                "model": model,
                "stream": False,
                "format": "json",
                "messages": [
                    {"role": "system", "content": "You create concise, evidence-grounded architecture specifications."},
                    {"role": "user", "content": prompt},
                ],
                "options": {"temperature": 0.1},
            },
            timeout=180,
        )
        response.raise_for_status()
        spec = json.loads(response.json()["message"]["content"])
        path = _render_architecture_diagram(spec, output_path)
        return json.dumps({"ok": True, "path": path})
    except requests.RequestException as error:
        return json.dumps({"ok": False, "error": f"Local Ollama request failed ({error.__class__.__name__}); start Ollama and check QWEN_MODEL."})
    except (KeyError, TypeError, ValueError, OSError) as error:
        return json.dumps({"ok": False, "error": f"Could not create architecture diagram: {error}"})
