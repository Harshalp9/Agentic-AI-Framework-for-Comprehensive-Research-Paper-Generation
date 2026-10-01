# PaperForge AI

PaperForge AI is a Streamlit application that helps turn a research project description and supporting evidence into an IEEE-style manuscript. A LangGraph workflow plans the paper, searches Semantic Scholar, drafts its sections, checks citations and claims, reviews the manuscript, and exports documents.

## Features

- Multi-stage research and writing workflow with visible progress.
- Semantic Scholar literature search and source metadata collection.
- Human review and approval of the proposed outline before drafting.
- Citation, claim, consistency, and manuscript validation checks.
- Optional project files and experimental CSV/JSON data as evidence.
- DOCX, PDF, and LaTeX exports, plus downloadable Markdown and a verification report.

Generated text should be reviewed before academic or other formal use. Provide measured results and source material yourself; verify all claims, citations, and exports.

## Requirements

- Python 3.10 or newer.
- A Mistral API key for the default model configuration.
- Optional: a Google AI API key for Gemini support, and a Semantic Scholar API key for improved search rate limits.

## Setup

From the project directory, create and activate a virtual environment, then install the dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Create a `.env` file in the project root and add your provider keys:

```dotenv
MISTRAL_API_KEY=your_mistral_api_key

# Optional Gemini support
GOOGLE_API_KEY=your_google_api_key
PRIMARY_LLM=mistral

# Optional Semantic Scholar key (unauthenticated search is also supported)
SEMANTIC_SCHOLAR_API_KEY=your_semantic_scholar_api_key
```

Keep `.env` private and do not commit API keys. `PRIMARY_LLM` can be set to `gemini` when `GOOGLE_API_KEY` is configured. Gemini is optional; Mistral is the default provider.

## Run

Start the Streamlit app from the project root:

```bash
streamlit run app.py
```

Open the local URL printed by Streamlit in your browser.

## Use

1. Describe the research problem, objective, method, dataset, evaluation setup, actual results, and known limitations.
2. Optionally attach project notes or evidence files (`.txt`, `.md`, `.csv`, `.json`, `.py`, or `.tex`). Experimental results can be supplied as CSV/JSON files or pasted as a JSON object.
3. Start generation and review the proposed outline, research gap, and retrieved literature.
4. Approve the outline to continue writing, or reject it with feedback to request a revised plan.
5. Review the manuscript and verification results, then download the available files.

The workflow can take several minutes because it calls language-model and literature-search services. Exported manuscript files are written under `outputs/final/`; uploaded project files are stored under `uploads/` by session. These directories may contain generated or user-provided data, so decide whether they belong in version control before staging the repository.

## Configuration

The application reads environment variables from `.env`:

| Variable | Purpose | Default |
| --- | --- | --- |
| `MISTRAL_API_KEY` | Authenticates Mistral requests | Required for the default provider |
| `GOOGLE_API_KEY` | Enables Gemini support and fallback | Not set |
| `PRIMARY_LLM` | Selects the primary structured-output provider (`mistral` or `gemini`) | `mistral` |
| `SEMANTIC_SCHOLAR_API_KEY` | Authenticates Semantic Scholar search requests | Not set |
| `MAX_REVISIONS` | Maximum automated revision cycles | `2` |
| `MISTRAL_RPS` | Mistral request rate limit | `0.2` |
| `GEMINI_RPS` | Gemini request rate limit | `0.2` |
| `GEMINI_MODEL` | Gemini model name | Defined in `paper.py` |
| `MAX_PROMPT_CHARS` | Maximum characters included per prompt section | `14000` |

The maximum revision cycles can also be adjusted in the app sidebar.

## Project Files

- `app.py` - Streamlit interface and workflow controls.
- `paper.py` - LangGraph research, writing, verification, and export workflow.
- `tools.py` - Semantic Scholar search tool.
- `requirements.txt` - Python dependencies.
- `Config.TOML` - Streamlit-related configuration file included in the project.