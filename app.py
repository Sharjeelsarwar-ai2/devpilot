from __future__ import annotations
import os
import zipfile
from pathlib import Path
import streamlit as st
from workflow import DEFAULT_MODEL, STAGES, WorkflowEngine


st.set_page_config(
    page_title="DevPilot AI",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] { font-family: Inter, sans-serif; }

    .stApp {
        background:
            radial-gradient(circle at 8% 8%, rgba(105,85,255,.17), transparent 29%),
            radial-gradient(circle at 88% 15%, rgba(0,207,255,.10), transparent 28%),
            radial-gradient(circle at 50% 100%, rgba(160,80,255,.08), transparent 35%),
            #070a12;
        color: #f5f7fb;
    }

    .block-container {
        max-width: 1480px;
        padding-top: 4.8rem;
        padding-bottom: 3rem;
    }

    [data-testid="stHeader"] {
        background: transparent !important;
        box-shadow: none !important;
    }

    [data-testid="stDecoration"] { display: none !important; }

    [data-testid="stSidebar"] {
        background: rgba(10,14,25,.74);
        border-right: 1px solid rgba(255,255,255,.08);
        backdrop-filter: blur(20px);
    }

    .glass {
        background: linear-gradient(135deg,rgba(255,255,255,.075),rgba(255,255,255,.025));
        border: 1px solid rgba(255,255,255,.10);
        box-shadow: 0 24px 70px rgba(0,0,0,.28), inset 0 1px 0 rgba(255,255,255,.05);
        border-radius: 22px;
        padding: 1.25rem;
        backdrop-filter: blur(22px);
    }

    .hero {
        position: relative;
        z-index: 2;
        padding: 1.9rem;
        border-radius: 28px;
        border: 1px solid rgba(255,255,255,.11);
        background: linear-gradient(135deg,rgba(111,90,255,.16),rgba(0,210,255,.06) 55%,rgba(255,255,255,.025));
        box-shadow: 0 24px 80px rgba(0,0,0,.3);
        backdrop-filter: blur(24px);
        margin-bottom: 1.1rem;
    }

    .kicker {
        font-size: .73rem;
        font-weight: 700;
        letter-spacing: .16em;
        text-transform: uppercase;
        color: rgba(255,255,255,.55);
        margin-bottom: .55rem;
    }

    .title {
        font-size: clamp(2.1rem,4vw,3.7rem);
        font-weight: 800;
        line-height: 1.01;
        letter-spacing: -.05em;
        margin: 0;
    }

    .gradient {
        background: linear-gradient(90deg,#fff 0%,#bec8ff 48%,#72dcff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .sub {
        color: rgba(255,255,255,.63);
        max-width: 980px;
        margin-top: .7rem;
        line-height: 1.6;
    }

    .pill {
        display: inline-block;
        padding: .38rem .7rem;
        margin: .85rem .35rem 0 0;
        border-radius: 999px;
        border: 1px solid rgba(255,255,255,.10);
        background: rgba(255,255,255,.045);
        color: rgba(255,255,255,.72);
        font-size: .73rem;
    }

    .section-title {
        font-size: 1.01rem;
        font-weight: 700;
        margin: .1rem 0 .45rem;
    }

    .section-sub {
        font-size: .82rem;
        color: rgba(255,255,255,.47);
        margin-bottom: 1rem;
    }

    .stage {
        display: flex;
        align-items: center;
        gap: .75rem;
        padding: .72rem .8rem;
        margin: .33rem 0;
        border-radius: 14px;
        border: 1px solid rgba(255,255,255,.07);
        background: rgba(255,255,255,.025);
    }

    .stage.active {
        background: rgba(118,100,255,.12);
        border-color: rgba(145,132,255,.25);
    }

    .stage.done { background: rgba(70,190,130,.055); }

    .stage.error {
        background: rgba(255,90,90,.07);
        border-color: rgba(255,100,100,.20);
    }

    .stage.skipped { opacity: .44; }

    .dot {
        width: 10px;
        height: 10px;
        border-radius: 50%;
        flex: 0 0 10px;
        background: rgba(255,255,255,.23);
    }

    .dot.active {
        background: #8e82ff;
        box-shadow: 0 0 16px rgba(142,130,255,.75);
    }

    .dot.done { background: #50d296; }
    .dot.error { background: #ff7777; }
    .dot.skipped { background: rgba(255,255,255,.13); }

    .stage-name {
        font-size: .82rem;
        color: rgba(255,255,255,.80);
    }

    .stage-detail {
        font-size: .69rem;
        color: rgba(255,255,255,.39);
        margin-top: .12rem;
    }

    .metric { text-align: center; padding: .75rem; }
    .metric-value { font-size: 1.38rem; font-weight: 800; }
    .metric-label { font-size: .72rem; color: rgba(255,255,255,.45); }

    [data-testid="stFileUploaderDropzone"] {
        background: rgba(255,255,255,.025) !important;
        border: 1px dashed rgba(255,255,255,.16) !important;
        border-radius: 18px !important;
    }

    textarea, input { border-radius: 14px !important; }
    .stButton > button {
        border-radius: 14px;
        min-height: 47px;
        font-weight: 700;
        border: 1px solid rgba(255,255,255,.10);
        box-shadow: 0 10px 30px rgba(0,0,0,.16);
    }

    footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)


def render_workflow(state, progress_slot, status_slot, timeline_slot):
    total = len(STAGES)
    terminal_ok = state.stage_state.get("final_report") == "done" and not state.aborted

    if terminal_ok:
        pct = 1.0
    else:
        completed = sum(
            1 for key, _, _ in STAGES
            if state.stage_state.get(key) == "done"
        )
        pct = min(0.95, completed / max(total, 1))

    progress_slot.progress(
        pct,
        text=f"Workflow progress · {int(pct * 100)}%"
    )

    rows = []
    for key, label, _ in STAGES:
        value = state.stage_state.get(key, "pending")
        icon = {
            "pending": "○",
            "active": "◉",
            "done": "✓",
            "error": "!",
            "skipped": "—",
        }[value]
        detail = state.stage_detail.get(key, "")
        rows.append(
            f'<div class="stage {value}">'
            f'<div class="dot {value}"></div>'
            f'<div><div class="stage-name">{icon} {label}</div>'
            f'<div class="stage-detail">{detail}</div></div></div>'
        )

    timeline_slot.markdown("".join(rows), unsafe_allow_html=True)

    active = next(
        (key for key, _, _ in STAGES
         if state.stage_state.get(key) == "active"),
        None,
    )

    labels = {key: label for key, label, _ in STAGES}

    if active:
        status_slot.markdown(f"**Currently:** {labels[active]}")
    elif state.aborted:
        status_slot.error(state.abort_reason)
    elif terminal_ok:
        status_slot.success("Development run verified")
    else:
        status_slot.markdown("**Status:** Waiting")


with st.sidebar:
    st.markdown(
        "<div class='section-title'>✦ DevPilot AI</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div class='section-sub'>Multi-stage AI software development workflow</div>",
        unsafe_allow_html=True,
    )

    api_key = st.text_input(
        "Groq API key",
        type="password",
        value=os.getenv("GROQ_API_KEY", ""),
    )

    model = st.text_input("Model", value=DEFAULT_MODEL)

    st.markdown(
        """
        <div class='glass'>
        <div class='section-title'>Execution architecture</div>
        <div class='section-sub'>
        Stage-based LLM reasoning, localized edits, deterministic verification,
        bounded repair and a controlled subprocess layer.
        </div>
        <span class='pill'>Groq</span>
        <span class='pill'>Structured stages</span>
        <span class='pill'>Patch editing</span>
        <span class='pill'>Pytest</span>
        <span class='pill'>Smoke test</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.caption("For Streamlit Cloud, store GROQ_API_KEY in App settings → Secrets.")

st.markdown(
    """
      <div class="hero">
      <div class="kicker">AI SOFTWARE ENGINEERING WORKFLOW</div>
      <div class="title">Build. Test. <span class="gradient">Repair.</span></div>
      <div class="sub">
      A bounded, specialized development workflow that separates requirements,
      inspection, design, implementation, test generation, testing, failure
      analysis, repair, sandbox verification and final reporting.
      </div>
      <div>
        <span class="pill">GPT-OSS 120B</span>
        <span class="pill">Localized edits</span>
        <span class="pill">Bounded retries</span>
        <span class="pill">Requirement tests</span>
        <span class="pill">Streamlit Cloud</span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

left, right = st.columns([1.18, 1], gap="large")

with left:
    st.markdown(
        "<div class='glass'><div class='section-title'>1 · Project</div>"
        "<div class='section-sub'>Upload the application ZIP the workflow should modify.</div>",
        unsafe_allow_html=True,
    )
    uploaded = st.file_uploader(
        "Project ZIP",
        type=["zip"],
        label_visibility="collapsed",
    )
    if uploaded:
        st.success(f"Ready · {uploaded.name}")
    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown(
        "<div class='glass'><div class='section-title'>2 · Requirement</div>"
        "<div class='section-sub'>Describe the outcome you need.</div>",
        unsafe_allow_html=True,
    )
    requirement = st.text_area(
        "Requirement",
        placeholder="Example: Add task priorities and highlight High priority tasks while preserving existing behavior.",
        height=150,
        label_visibility="collapsed",
    )
    st.markdown("</div>", unsafe_allow_html=True)

run_clicked = st.button(
    "✦  Run Multi-Stage Development Workflow",
    type="primary",
    use_container_width=True,
)

if run_clicked:
    if not api_key:
        st.error("Add your Groq API key in the sidebar or Streamlit Cloud Secrets.")
        st.stop()
    if uploaded is None:
        st.error("Upload a project ZIP first.")
        st.stop()
    if not requirement.strip():
        st.error("Describe the software requirement first.")
        st.stop()

    st.markdown("### Agent workflow")
    progress_slot = st.empty()
    status_slot = st.empty()
    timeline_slot = st.empty()

    progress_slot.progress(0, text="Workflow progress · 0%")

    def on_update(state):
        render_workflow(
            state,
            progress_slot,
            status_slot,
            timeline_slot,
        )

    engine = WorkflowEngine(
        api_key=api_key,
        model=model.strip() or DEFAULT_MODEL,
        callback=on_update,
    )

    with st.spinner("Specialized stages are working through the pipeline..."):
        state = engine.run(uploaded, requirement)

    render_workflow(
        state,
        progress_slot,
        status_slot,
        timeline_slot,
    )

    st.divider()

    successful = sum(1 for e in state.events if e["ok"])
    failed = len(state.events) - successful

    c1, c2, c3, c4 = st.columns(4)
    metrics = [
        (c1, len(state.events), "Workflow actions"),
        (c2, successful, "Successful"),
        (c3, failed, "Failed"),
        (c4, state.repair_attempts, "Repair attempts"),
    ]

    for slot, value, label in metrics:
        with slot:
            st.markdown(
                f"<div class='glass'><div class='metric'>"
                f"<div class='metric-value'>{value}</div>"
                f"<div class='metric-label'>{label}</div>"
                f"</div></div>",
                unsafe_allow_html=True,
            )

    st.markdown("### Final report")
    st.markdown(state.final_report)

    if state.workspace and state.workspace.exists():
        st.markdown("### Modified project")

        rows = []
        for path in sorted(state.workspace.rglob("*")):
            if path.is_file():
                try:
                    rows.append({
                        "file": path.relative_to(state.workspace).as_posix(),
                        "size": path.stat().st_size,
                    })
                except OSError:
                    pass

        st.dataframe(rows, use_container_width=True, hide_index=True)

        output_zip = state.workspace.parent / "modified_project.zip"

        with zipfile.ZipFile(
            output_zip,
            "w",
            zipfile.ZIP_DEFLATED,
        ) as zf:
            for path in state.workspace.rglob("*"):
                if path.is_file():
                    zf.write(path, path.relative_to(state.workspace))

        with open(output_zip, "rb") as handle:
            st.download_button(
                "Download modified project",
                data=handle,
                file_name="modified_project.zip",
                mime="application/zip",
                use_container_width=True,
            )

    with st.expander("Detailed workflow activity"):
        for i, event in enumerate(state.events, 1):
            icon = "✅" if event["ok"] else "❌"
            st.markdown(f"**{icon} {i}. {event['detail']}**")
            st.json(event["result"])
