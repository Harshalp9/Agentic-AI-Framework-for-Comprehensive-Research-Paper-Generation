import csv
import io
import json
import os
import traceback
import uuid
from pathlib import Path

import streamlit as st
from langgraph.types import Command

st.set_page_config(
    page_title="PaperForge AI",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)


# STYLE


st.markdown(
    """
<style>
#MainMenu, footer {visibility: hidden;}
.block-container {padding-top: 2rem; max-width: 1250px;}

.pf-hero {
    padding: 28px 32px; border-radius: 20px; margin-bottom: 22px;
    background: linear-gradient(120deg, rgba(99,102,241,.22), rgba(236,72,153,.18) 55%, rgba(14,165,233,.16));
    border: 1px solid rgba(128,128,128,.28);
}
.pf-hero h1 {
    margin: 0; font-size: 2.3rem; font-weight: 800; letter-spacing: -.5px;
    background: linear-gradient(90deg, #6366f1, #ec4899);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.pf-hero p {margin: 6px 0 0 0; font-size: 1.02rem; opacity: .85;}

.pf-chips {margin: 6px 0 14px 0;}
.pf-chip {
    display: inline-block; padding: 6px 13px; margin: 3px 4px 3px 0; border-radius: 999px;
    font-size: .82rem; border: 1px solid rgba(128,128,128,.35); opacity: .7;
}
.pf-chip.done {background: rgba(34,197,94,.15); border-color: rgba(34,197,94,.55); opacity: 1;}
.pf-chip.active {
    background: rgba(99,102,241,.22); border-color: rgba(99,102,241,.9); opacity: 1;
    animation: pf-pulse 1.3s ease-in-out infinite;
}
@keyframes pf-pulse {
    0%,100% {box-shadow: 0 0 0 0 rgba(99,102,241,.45);}
    50% {box-shadow: 0 0 0 7px rgba(99,102,241,0);}
}

.pf-card {
    padding: 16px 18px; border-radius: 14px; border: 1px solid rgba(128,128,128,.28);
    background: rgba(128,128,128,.06); margin-bottom: 10px;
}
.pf-tag {
    display: inline-block; padding: 2px 10px; border-radius: 8px; font-size: .75rem;
    background: rgba(99,102,241,.16); margin-right: 6px;
}
.pf-ok {color: #22c55e; font-weight: 600;}
.pf-bad {color: #ef4444; font-weight: 600;}
div[data-testid="stMetric"] {
    border: 1px solid rgba(128,128,128,.28); border-radius: 14px; padding: 12px 16px;
    background: rgba(128,128,128,.05);
}
</style>
""",
    unsafe_allow_html=True,
)


# LOAD THE ENGINE (paper.py)



@st.cache_resource(show_spinner="Loading PaperForge engine...")
def load_engine():
    import paper  # noqa: WPS433  (imports the compiled LangGraph app)

    return paper


try:
    paper = load_engine()
except Exception:  # noqa: BLE001
    st.markdown('<div class="pf-hero"><h1>PaperForge AI</h1></div>', unsafe_allow_html=True)
    st.error("Could not load `paper.py`. Fix the error below and refresh.")
    st.code(traceback.format_exc())
    st.stop()


# CONSTANTS


# NOTE: the "Experimental Setup" writing stage has been removed and the old
# "Results" / "Discussion" writing stages have been merged into a single
# "Result & Discussion" stage. The figure/table (visualization) stages have
# also been removed - the Writer now feeds directly into verification.
STAGES = [
    ("supervisor", "Supervisor", "🧭"),
    ("project_understanding", "Understand", "🧠"),
    ("research_planner", "Plan", "🗺️"),
    ("literature_research", "Literature", "📚"),
    ("paper_filtering", "Filter", "🧹"),
    ("literature_analysis", "Analyze", "🔬"),
    ("literature_synthesis", "Synthesize", "🧾"),
    ("research_gap", "Gap", "🎯"),
    ("paper_outline", "Outline", "📝"),
    ("human_approval", "Approval", "🧑‍⚖️"),
    ("introduction", "Introduction", "✍️"),
    ("literature_review", "Lit. Review", "✍️"),
    ("methodology", "Methodology", "✍️"),
    ("result_and_discussion", "Result & Discussion", "✍️"),
    ("conclusion", "Conclusion", "✍️"),
    ("future_scope", "Future Scope", "✍️"),
    ("abstract", "Abstract", "✍️"),
    ("citation_verification", "Citations", "🔗"),
    ("claim_verification", "Claims", "✅"),
    ("consistency_verification", "Consistency", "🧩"),
    ("review", "Review", "🔎"),
    ("revision", "Revision", "🔁"),
    ("formatting", "Format", "🎨"),
    ("final_validation", "Validate", "🛡️"),
    ("document_export", "Export", "📦"),
]
STAGE_IDS = [s[0] for s in STAGES]

NODE_TO_STAGE = {
    "semantic_scholar": "literature_research",
    "collect_literature": "literature_research",
}
STAY_ON_STAGE = {"literature_research", "semantic_scholar", "writer"}

NODE_LABELS = {
    "supervisor": "Supervisor",
    "project_understanding": "Project understanding",
    "research_planner": "Research planner",
    "literature_research": "Literature agent",
    "semantic_scholar": "Semantic Scholar search",
    "collect_literature": "Literature collector",
    "paper_filtering": "Paper filtering and ranking",
    "literature_analysis": "Paper analysis",
    "literature_synthesis": "Literature synthesis",
    "research_gap": "Research gap analysis",
    "paper_outline": "Paper outline",
    "human_approval": "Human approval",
    "writer": "Writer",
    "citation_verification": "Citation verification",
    "claim_verification": "Claim verification",
    "consistency_verification": "Consistency verification",
    "review": "Academic review",
    "revision": "Revision",
    "formatting": "Formatting",
    "final_validation": "Final validation",
    "document_export": "Document export",
}

EXAMPLE_TEMPLATE = """Research problem: [what problem does your project address?]

Objective: [what do you want to achieve?]

Proposed solution / method: [model, algorithm or system you built]

Dataset: [name, size, source, train/test split]

Setup details (folded into Result & Discussion): [hardware, hyperparameters, baselines, tools]

Evaluation metrics: [e.g. accuracy, F1, latency]

Actual results: [ONLY real numbers you measured, e.g. Accuracy = 91.2%]

Limitations: [known weaknesses]
"""

MIME = {
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".pdf": "application/pdf",
    ".tex": "text/plain",
    ".json": "application/json",
    ".md": "text/markdown",
}


# SESSION STATE


DEFAULTS = {
    "phase": "setup",        # setup | running | awaiting_approval | done | error
    "thread_id": None,
    "pending": None,         # next graph input to run
    "payload": None,         # interrupt payload for approval
    "seen": [],              # nodes completed
    "last": None,            # last completed node
    "log": [],               # (node, message)
    "revisions": 0,
    "error": None,
    "max_rev": 2,
    "description_text": "",
    "paper_metadata": {},
}
for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)
if st.session_state.thread_id is None:
    st.session_state.thread_id = f"paperforge-{uuid.uuid4().hex[:10]}"

ss = st.session_state


def config() -> dict:
    return {"configurable": {"thread_id": ss.thread_id}, "recursion_limit": 200}


def graph_values() -> dict:
    try:
        return dict(paper.app.get_state(config()).values or {})
    except Exception:  # noqa: BLE001
        return {}


def reset_all():
    for k, v in DEFAULTS.items():
        ss[k] = v if not isinstance(v, (list, dict)) else type(v)()
    ss.thread_id = f"paperforge-{uuid.uuid4().hex[:10]}"
    ss.description_text = ""



# HELPERS



def stage_of(node: str) -> str:
    return NODE_TO_STAGE.get(node, node)


def render_chips(seen: list, last: str | None) -> str:
    seen_stages = {stage_of(n) for n in seen}
    active = None
    if last:
        cur = stage_of(last)
        if last in STAY_ON_STAGE and cur in STAGE_IDS:
            active = cur
        elif cur in STAGE_IDS:
            i = STAGE_IDS.index(cur)
            active = STAGE_IDS[i + 1] if i + 1 < len(STAGE_IDS) else None
    else:
        active = STAGE_IDS[0]

    html = ['<div class="pf-chips">']
    for sid, label, icon in STAGES:
        cls = "active" if sid == active else ("done" if sid in seen_stages else "")
        html.append(f'<span class="pf-chip {cls}">{icon} {label}</span>')
    html.append("</div>")
    return "".join(html)


def progress_value(seen: list) -> float:
    idx = [STAGE_IDS.index(stage_of(n)) for n in seen if stage_of(n) in STAGE_IDS]
    return min(1.0, (max(idx) + 1) / len(STAGE_IDS)) if idx else 0.0


def summarize(node: str, upd) -> str:
    upd = upd if isinstance(upd, dict) else {}
    try:
        if node == "collect_literature":
            return f"{len(upd.get('literature_results', []))} papers retrieved"
        if node == "paper_filtering":
            return f"{len(upd.get('literature_results', []))} complete sources selected"
        if node == "literature_analysis":
            return f"{len(upd.get('literature_matrix', []))} papers analysed"
        if node == "literature_synthesis":
            return f"{len(upd.get('literature_synthesis', {}).get('themes', []))} literature themes"
        if node == "writer":
            done = upd.get("completed_sections", [])
            return f"wrote **{done[-1]}**" if done else "writing..."
        if node == "paper_outline":
            t = upd.get("title_candidates", [])
            return f"{len(t)} title candidates, {len(upd.get('paper_outline', {}).get('sections', []))} sections"
        if node == "citation_verification":
            return f"{len(upd.get('citation_issues', []))} citation issues"
        if node == "claim_verification":
            return f"{len(upd.get('claim_issues', []))} claim issues"
        if node == "consistency_verification":
            return f"{len(upd.get('consistency_issues', []))} consistency issues"
        if node == "review":
            return "approved" if upd.get("is_approved") else f"{len(upd.get('review_issues', []))} review issues"
        if node == "revision":
            return f"cycle {upd.get('revision_count', '?')}: {str(upd.get('review_feedback', ''))[:120]}"
        if node == "final_validation":
            return f"status: **{upd.get('final_validation_status', '?')}**"
        if node == "document_export":
            return f"{len(upd.get('output_files', []))} files"
    except Exception:  # noqa: BLE001
        pass
    return "done"


def _coerce(v: str):
    try:
        return float(v) if "." in v else int(v)
    except (ValueError, TypeError):
        return v


def parse_results_files(files) -> dict:
    out = {}
    for f in files or []:
        stem, raw = Path(f.name).stem, f.getvalue()
        if f.name.lower().endswith(".json"):
            out[stem] = json.loads(raw.decode("utf-8", "ignore"))
        else:
            reader = csv.DictReader(io.StringIO(raw.decode("utf-8", "ignore")))
            out[stem] = [{k: _coerce(v) for k, v in row.items()} for row in reader]
    return out


def save_uploads(files) -> list[str]:
    paths = []
    folder = Path("uploads") / ss.thread_id
    folder.mkdir(parents=True, exist_ok=True)
    for f in files or []:
        p = folder / f.name
        p.write_bytes(f.getvalue())
        paths.append(str(p))
    return paths



# SIDEBAR



def render_sidebar():
    with st.sidebar:
        st.markdown("### 📄 PaperForge AI")
        st.caption("Evidence-grounded, multi-agent research paper generator")
        st.divider()

        st.markdown("**Models**")
        mistral_ok = bool(os.getenv("MISTRAL_API_KEY"))
        gemini_ok = getattr(paper, "gemini", None) is not None
        st.markdown(
            f"Mistral: {'<span class=pf-ok>key set</span>' if mistral_ok else '<span class=pf-bad>missing</span>'}<br>"
            f"Gemini: {'<span class=pf-ok>ready</span>' if gemini_ok else '<span class=pf-bad>not configured</span>'}<br>"
            f"Primary: <b>{getattr(paper, 'PRIMARY', 'mistral')}</b>",
            unsafe_allow_html=True,
        )
        st.divider()

        st.markdown("**Settings**")
        ss.max_rev = st.slider(
            "Max revision cycles", 0, 4, ss.max_rev,
            help="How many times the Revision agent may rewrite before the paper is exported anyway.",
            disabled=ss.phase in ("running",),
        )
        paper.MAX_REVISIONS = ss.max_rev
        st.divider()

        if ss.log:
            with st.expander("Activity log", expanded=False):
                for node, msg in ss.log[-40:]:
                    st.markdown(f"**{NODE_LABELS.get(node, node)}** - {msg}")

        if st.button("🔄 New project", use_container_width=True, disabled=ss.phase == "running"):
            reset_all()
            st.rerun()

        st.caption("Tip: don't click anything while the pipeline is running; it would interrupt the run.")



# VIEWS



def render_header():
    st.markdown(
        """
<div class="pf-hero">
  <h1>PaperForge AI</h1>
  <p>Describe your project. Agents plan, search the literature, write, verify every claim and citation, and export an IEEE-style paper.</p>
</div>
""",
        unsafe_allow_html=True,
    )


def view_setup():
    left, right = st.columns([3, 2], gap="large")

    with left:
        st.subheader("1 · Describe your project")
        c1, c2 = st.columns([1, 1])
        if c1.button("📋 Load template", help="Fill the box with a fill-in-the-blanks template"):
            ss.description_text = EXAMPLE_TEMPLATE
            st.rerun()
        if c2.button("🧹 Clear"):
            ss.description_text = ""
            st.rerun()

        ss.description_text = st.text_area(
            "Project description",
            value=ss.description_text,
            height=330,
            placeholder="Problem, method, dataset, setup, metrics, real results, limitations...",
            label_visibility="collapsed",
        )
        st.caption(
            "The more concrete detail you give (especially **real results**), the less the paper "
            "will say \"information is missing\". Nothing is invented by the agents."
        )

    with right:
        st.subheader("2 · Evidence (optional)")
        proj_files = st.file_uploader(
            "Project files (notes, README, code, logs)",
            type=["txt", "md", "csv", "json", "py", "tex"],
            accept_multiple_files=True,
        )
        result_files = st.file_uploader(
            "Experimental data (CSV or JSON), used to ground the Result & Discussion section",
            type=["csv", "json"],
            accept_multiple_files=True,
            help="Each file becomes a named evidence source, e.g. metrics.csv -> 'metrics'.",
        )
        with st.expander("Advanced: paste results as JSON"):
            results_json = st.text_area(
                "actual_results JSON",
                value="",
                height=120,
                placeholder='{"model_metrics": [{"model": "A", "accuracy": 0.91}]}',
                label_visibility="collapsed",
            )
        with st.expander("IEEE author block (optional)"):
            st.caption("Leave blank to export a title-only manuscript. Add one author per line as Name | Department | Institution | City, Country | email.")
            author_lines = st.text_area(
                "Authors", value="", height=100,
                placeholder="A. Researcher | Department of AI | Example University | Pune, India | a@example.edu",
            )

    st.divider()
    go = st.button("🚀 Generate paper", type="primary", use_container_width=True)

    if go:
        if len(ss.description_text.strip()) < 30:
            st.warning("Please describe your project in a bit more detail (at least a couple of sentences).")
            return
        try:
            actual = parse_results_files(result_files)
            if results_json.strip():
                extra = json.loads(results_json)
                if not isinstance(extra, dict):
                    raise ValueError("Pasted results must be a JSON object.")
                actual.update(extra)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Could not read the experimental data: {exc}")
            return

        ss.pending = {
            "project_description": ss.description_text.strip(),
            "uploaded_files": save_uploads(proj_files),
            "messages": [],
            "actual_results": actual,
            "paper_metadata": {"authors": [line.strip() for line in author_lines.splitlines() if line.strip()]},
        }
        ss.seen, ss.last, ss.log, ss.revisions, ss.error = [], None, [], 0, None
        ss.phase = "running"
        st.rerun()


def run_pipeline():
    graph_input = ss.pending
    ss.pending = None
    if graph_input is None:  # e.g. page interaction interrupted a previous run
        ss.phase = "setup"
        st.rerun()

    st.subheader("⚙️ Pipeline running")
    bar = st.progress(progress_value(ss.seen))
    chips = st.empty()
    chips.markdown(render_chips(ss.seen, ss.last), unsafe_allow_html=True)

    ss.phase = "running"
    with st.status("Agents are working... this can take several minutes", expanded=True) as status:
        try:
            for chunk in paper.app.stream(graph_input, config=config(), stream_mode="updates"):
                for node, upd in chunk.items():
                    if node == "__interrupt__":
                        ss.payload = upd[0].value
                        ss.phase = "awaiting_approval"
                        continue
                    # The graph uses one Writer node repeatedly; expose each
                    # completed manuscript section as its own workflow step.
                    display_node = node
                    if node == "writer" and isinstance(upd, dict):
                        completed = upd.get("completed_sections", [])
                        if completed:
                            display_node = completed[-1]
                    ss.seen.append(display_node)
                    ss.last = display_node
                    if node == "revision" and isinstance(upd, dict):
                        ss.revisions = upd.get("revision_count", ss.revisions)
                    msg = summarize(node, upd)
                    ss.log.append((node, msg))
                    st.markdown(f"✔️ **{NODE_LABELS.get(node, node)}** - {msg}")
                    bar.progress(progress_value(ss.seen))
                    chips.markdown(render_chips(ss.seen, ss.last), unsafe_allow_html=True)

            if ss.phase == "awaiting_approval":
                status.update(label="Waiting for your approval of the outline", state="complete")
            else:
                ss.phase = "done"
                bar.progress(1.0)
                status.update(label="Pipeline finished", state="complete")
        except Exception:  # noqa: BLE001
            ss.error = traceback.format_exc()
            ss.phase = "error"
            status.update(label="Pipeline failed", state="error")
    st.rerun()


def view_approval():
    payload = ss.payload or {}
    vals = graph_values()
    outline = payload.get("paper_outline", {})
    titles = payload.get("title_candidates", [])

    st.subheader("🧑‍⚖️ Review the paper outline")
    st.markdown(
        '<div class="pf-card">The Writer will follow this outline exactly. '
        "Approve it, or reject with feedback and the planner will redo the plan and outline.</div>",
        unsafe_allow_html=True,
    )

    missing = vals.get("missing_information", [])
    if missing:
        with st.expander(f"⚠️ {len(missing)} pieces of project information are missing", expanded=False):
            for m in missing:
                st.markdown(f"- {m}")
            st.caption("Consider rejecting and adding these details for a stronger paper.")

    t1, t2, t3 = st.tabs(["📝 Outline", "🎯 Research gap", "📚 Literature found"])

    with t1:
        for s in outline.get("sections", []):
            with st.expander(f"{s.get('section_number', '')} {s.get('title', '')}".strip(), expanded=False):
                st.markdown(f"**Purpose:** {s.get('purpose', '')}")
                for kp in s.get("key_points", []):
                    st.markdown(f"- {kp}")
                if s.get("evidence_needed"):
                    st.markdown("**Evidence needed:** " + "; ".join(s["evidence_needed"]))
                if s.get("related_sources"):
                    st.markdown(
                        "".join(f'<span class="pf-tag">{r}</span>' for r in s["related_sources"]),
                        unsafe_allow_html=True,
                    )
        if outline.get("writing_notes"):
            with st.expander("Writing notes"):
                for n in outline["writing_notes"]:
                    st.markdown(f"- {n}")

    with t2:
        gap = vals.get("research_gap", {})
        if gap:
            st.markdown(f"**Overall gap:** {gap.get('overall_gap') or '_not established_'}")
            st.markdown(f"**Confidence:** {gap.get('confidence') or '_n/a_'}")
            for label, key in [
                ("Identified gaps", "identified_gaps"),
                ("Evidence", "gap_evidence"),
                ("Limitations of existing work", "limitations_of_existing_work"),
            ]:
                if gap.get(key):
                    st.markdown(f"**{label}**")
                    for it in gap[key]:
                        st.markdown(f"- {it}")
        else:
            st.info("No research gap information available.")

    with t3:
        papers = vals.get("literature_results", [])
        if not papers:
            st.warning("No papers were retrieved. Check `tools.py` / Semantic Scholar limits; citations will be unsupported.")
        elif len(papers) < getattr(paper, "MIN_RELEVANT_SOURCES", 15):
            st.warning(f"{len(papers)} complete sources were selected; the target is 15. The system exhausted its retrieval attempts without fabricating references.")
        else:
            st.success(f"{len(papers)} complete, ranked sources are available for the literature review.")
        for p in papers:
            with st.expander(f"[{p.get('source_id')}] {p.get('title')} ({p.get('year')})"):
                st.caption(", ".join(p.get("authors", [])[:6]))
                st.write((p.get("abstract") or "_no abstract_")[:900])
                if p.get("url"):
                    st.markdown(f"[Open source]({p['url']})")

    st.divider()
    st.markdown("#### Your decision")
    chosen = None
    if titles:
        chosen = st.radio("Choose the paper title", titles, index=0)
    feedback = st.text_area(
        "Feedback (only needed if you reject)",
        placeholder="e.g. Add more detail to the Result and Discussion section; focus the gap on scalability...",
        height=90,
    )

    c1, c2 = st.columns(2)
    if c1.button("✅ Approve and write the paper", type="primary", use_container_width=True):
        resume = {"status": "approved", "feedback": ""}
        if chosen:
            resume["selected_title"] = chosen
        ss.pending = Command(resume=resume)
        ss.phase = "running"
        st.rerun()
    if c2.button("↩️ Reject and re-plan", use_container_width=True):
        if not feedback.strip():
            st.warning("Please tell the planner what to change.")
        else:
            ss.pending = Command(resume={"status": "rejected", "feedback": feedback.strip()})
            ss.phase = "running"
            st.rerun()


def _issue_table(label: str, issues: list):
    with st.expander(f"{label} ({len(issues)})", expanded=bool(issues)):
        if issues:
            rows = [i if isinstance(i, dict) else {"issue": str(i)} for i in issues]
            st.dataframe(rows, use_container_width=True, hide_index=True)
        else:
            st.success("No issues.")


def view_results():
    vals = graph_values()
    status = vals.get("final_validation_status", "n/a")
    files = [f for f in vals.get("output_files", []) if isinstance(f, str) and os.path.isfile(f)]

    st.subheader("🎉 Your manuscript")

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Validation", status.replace("_", " ").title() if status else "n/a")
    m2.metric("Papers retrieved", len(vals.get("literature_results", [])))
    m3.metric("Sources cited", len(vals.get("citation_records", [])))
    m4.metric("Revision cycles", vals.get("revision_count", 0))
    m5.metric("Files exported", len(files))

    if status != "pass":
        st.warning(
            "The manuscript was exported with unresolved validation issues. Review the "
            "**Verification** tab before using it."
        )
    if not vals.get("literature_results"):
        st.warning("No literature was retrieved, so citations could not be grounded.")

    tab_ms, tab_ver, tab_lit, tab_dl = st.tabs(["📄 Manuscript", "🔍 Verification", "📚 Literature", "⬇️ Downloads"])

    manuscript = vals.get("final_manuscript", "")
    with tab_ms:
        if manuscript:
            with st.container(border=True):
                st.markdown(manuscript)
        else:
            st.info("No manuscript was produced.")

    with tab_ver:
        _issue_table("Citation issues", vals.get("citation_issues", []))
        _issue_table("Claim issues", vals.get("claim_issues", []))
        _issue_table("Consistency issues", vals.get("consistency_issues", []))
        _issue_table("Review issues", vals.get("review_issues", []))
        _issue_table("Final validation issues", vals.get("final_validation_issues", []))
        if vals.get("final_validation_warnings"):
            st.info("Warnings: " + "; ".join(map(str, vals["final_validation_warnings"])))

    with tab_lit:
        papers = vals.get("literature_results", [])
        if not papers:
            st.info("No literature retrieved.")
        for p in papers:
            with st.expander(f"[{p.get('source_id')}] {p.get('title')} ({p.get('year')})"):
                st.caption(", ".join(p.get("authors", [])[:8]))
                st.write((p.get("abstract") or "_no abstract_")[:1200])
                if p.get("doi"):
                    st.markdown(f"DOI: `{p['doi']}`")
                if p.get("url"):
                    st.markdown(f"[Open source]({p['url']})")

    with tab_dl:
        st.markdown("Download the generated files:")
        cols = st.columns(3)
        i = 0
        for path in files:
            ext = Path(path).suffix.lower()
            if ext in MIME:
                with open(path, "rb") as fh:
                    cols[i % 3].download_button(
                        f"⬇️ {Path(path).name}", fh.read(), file_name=Path(path).name,
                        mime=MIME[ext], use_container_width=True, key=f"dl_{i}",
                    )
                i += 1
        if manuscript:
            cols[i % 3].download_button(
                "⬇️ manuscript.md", manuscript, file_name="manuscript.md",
                mime="text/markdown", use_container_width=True, key="dl_md",
            )
            i += 1
        report = {
            "citation_issues": vals.get("citation_issues", []),
            "claim_issues": vals.get("claim_issues", []),
            "consistency_issues": vals.get("consistency_issues", []),
            "review_issues": vals.get("review_issues", []),
            "final_validation_issues": vals.get("final_validation_issues", []),
            "verification_history": vals.get("verification_results", []),
        }
        cols[i % 3].download_button(
            "⬇️ verification_report.json", json.dumps(report, indent=2, default=str),
            file_name="verification_report.json", mime="application/json",
            use_container_width=True, key="dl_report",
        )
        if not files:
            st.info("DOCX/PDF/LaTeX files were not generated (export needs validation to pass or the revision cap to be reached).")

    st.divider()
    if st.button("✨ Start a new paper", type="primary"):
        reset_all()
        st.rerun()


def view_error():
    st.error("The pipeline stopped because of an error.")
    err = ss.error or ""
    hints = []
    if "429" in err or "rate" in err.lower():
        hints.append("A provider is **rate limiting** you. Wait a minute, lower `MISTRAL_RPS` / `GEMINI_RPS`, or set `PRIMARY_LLM=gemini`.")
    if "NOT_FOUND" in err or "404" in err:
        hints.append("A **model name** is not available to your key. Set `GEMINI_MODEL` in `.env`.")
    if "API key" in err or "401" in err or "api_key" in err.lower():
        hints.append("Check your **API keys** in `.env`.")
    for h in hints:
        st.info(h)
    with st.expander("Technical details", expanded=not hints):
        st.code(err)
    c1, c2 = st.columns(2)
    if c1.button("🔁 Back to setup", use_container_width=True):
        ss.phase = "setup"
        ss.error = None
        st.rerun()
    if c2.button("🔄 New project", use_container_width=True):
        reset_all()
        st.rerun()



# MAIN


render_header()
render_sidebar()

if ss.phase == "setup":
    view_setup()
elif ss.phase == "running":
    run_pipeline()
elif ss.phase == "awaiting_approval":
    st.markdown(render_chips(ss.seen, ss.last), unsafe_allow_html=True)
    view_approval()
elif ss.phase == "done":
    st.markdown(render_chips(ss.seen, "document_export"), unsafe_allow_html=True)
    view_results()
elif ss.phase == "error":
    view_error()