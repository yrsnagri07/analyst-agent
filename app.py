"""Analyst Agent: Autonomous AI Data Analyst Streamlit Application."""

import os
from pathlib import Path
import streamlit as st
import pandas as pd

from agent import run_agent_loop, get_gemini_api_key, get_gemini_model_name
from profiling import load_csv, profile_dataframe, generate_suggested_questions, FileTooLargeError, CSVLoadError
from ui import CUSTOM_CSS, apply_plotly_minimal_theme, render_confidence_badge, generate_markdown_report

# Page configuration
st.set_page_config(
    page_title="Analyst",
    page_icon=None,
    layout="centered",
    initial_sidebar_state="collapsed",
)

# Inject minimal, calm CSS
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def init_session():
    """Initialize Streamlit session state keys."""
    if "df" not in st.session_state:
        st.session_state.df = None
    if "df_profile" not in st.session_state:
        st.session_state.df_profile = None
    if "suggested_questions" not in st.session_state:
        st.session_state.suggested_questions = []
    if "history" not in st.session_state:
        st.session_state.history = []
    if "input_query" not in st.session_state:
        st.session_state.input_query = ""
    if "custom_api_key" not in st.session_state:
        st.session_state.custom_api_key = ""


init_session()

# Header
st.markdown(
    """
    <div class="analyst-header">
        <h1 class="analyst-title">Analyst</h1>
        <p class="analyst-subtitle">Autonomous agentic AI data analyst</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# API Key verification & optional inline configuration
api_key = get_gemini_api_key() or st.session_state.custom_api_key
if not api_key:
    with st.expander("API Key Configuration", expanded=True):
        st.info("No Gemini API key detected in environment or Streamlit secrets.")
        user_key = st.text_input(
            "Enter Gemini API Key",
            type="password",
            placeholder="AIzaSy...",
            help="Your key is kept in memory during this session and never stored to disk."
        )
        if user_key:
            st.session_state.custom_api_key = user_key.strip()
            st.rerun()

# Step 1: File Uploader & Sample Dataset
col_up1, col_up2 = st.columns([3, 1])

with col_up1:
    uploaded_file = st.file_uploader(
        "Upload dataset (CSV, max 20 MB)",
        type=["csv"],
        help="Upload any standard CSV file with headers."
    )

with col_up2:
    st.write("")
    st.write("")
    if st.button("Use sample data", help="Load the 12-month synthetic sales dataset"):
        sample_path = Path("sample_data/sales.csv")
        if sample_path.exists():
            try:
                loaded_df = pd.read_csv(sample_path)
                st.session_state.df = loaded_df
                st.session_state.df_profile = profile_dataframe(loaded_df)
                st.session_state.suggested_questions = generate_suggested_questions(loaded_df, st.session_state.df_profile)
                st.session_state.history = []
                st.rerun()
            except Exception as e:
                st.error(f"Failed to load sample data: {e}")
        else:
            st.error("Sample dataset file not found at sample_data/sales.csv")

if uploaded_file is not None:
    try:
        # Load and profile dataframe
        loaded_df = load_csv(uploaded_file, uploaded_file.name)
        # Check if new file
        if st.session_state.df is None or not loaded_df.equals(st.session_state.df):
            st.session_state.df = loaded_df
            st.session_state.df_profile = profile_dataframe(loaded_df)
            st.session_state.suggested_questions = generate_suggested_questions(loaded_df, st.session_state.df_profile)
            st.session_state.history = []
    except FileTooLargeError as fle:
        st.error(str(fle))
    except CSVLoadError as cle:
        st.error(str(cle))
    except Exception as e:
        st.error(f"Error loading CSV file: {str(e)}")

# Render Dataset Card if data is loaded
if st.session_state.df is not None and st.session_state.df_profile is not None:
    prof = st.session_state.df_profile
    chip_html = "".join(
        f'<span class="col-chip">{col["name"]}<span class="col-type">{col["dtype"]}</span></span>'
        for col in prof["columns_info"]
    )
    
    st.markdown(
        f"""
        <div class="dataset-card">
            <div class="dataset-stats">{prof['rows']:,} rows &times; {prof['columns']} columns &bull; {prof['memory_mb']} MB</div>
            <div class="chip-container">{chip_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("Preview dataset", expanded=False):
        st.dataframe(st.session_state.df.head(10), use_container_width=True)

    # Step 2: Query Input
    st.markdown("### Investigate")

    # Suggested Questions Buttons
    if st.session_state.suggested_questions:
        q_cols = st.columns(len(st.session_state.suggested_questions))
        for idx, sq in enumerate(st.session_state.suggested_questions):
            with q_cols[idx]:
                if st.button(sq, key=f"sq_{idx}", use_container_width=True):
                    st.session_state.input_query = sq

    # Single-line query form
    with st.form(key="query_form", clear_on_submit=False):
        user_query = st.text_input(
            "Ask anything about your data",
            value=st.session_state.input_query,
            placeholder="e.g. Why did revenue drop in March?",
            label_visibility="collapsed"
        )
        col_btn, _ = st.columns([1, 4])
        with col_btn:
            submit_clicked = st.form_submit_button("Analyze", use_container_width=True)

    # Handle Analysis Execution
    if submit_clicked and user_query.strip():
        query_text = user_query.strip()
        st.session_state.input_query = ""

        # Prepare chat context
        prior_context = []
        for turn in st.session_state.history:
            prior_context.append({"role": "user", "content": turn["question"]})
            if turn.get("final") and turn["final"].get("summary"):
                prior_context.append({"role": "model", "content": turn["final"]["summary"]})

        # Containers for streaming status and timeline
        status_box = st.status("Analyzing data...", expanded=True)

        current_turn = {
            "question": query_text,
            "plan": None,
            "steps": [],
            "code_snippets": [],
            "charts": [],
            "final": None,
            "ask_user": None,
        }

        with status_box:
            current_step_item = None
            for event in run_agent_loop(
                df=st.session_state.df,
                question=query_text,
                chat_history=prior_context,
                api_key=api_key,
                df_profile=st.session_state.df_profile,
            ):
                event_type = event.get("type")
                step_num = event.get("step", 1)

                if event_type == "plan":
                    current_turn["plan"] = event["content"]
                    st.markdown(f"**Analysis Plan:**\n{event['content']}")

                elif event_type == "code":
                    code_str = event["content"]
                    current_turn["code_snippets"].append(code_str)
                    st.markdown(f"**Step {step_num}/8: Running query**")
                    with st.expander(f"Code (Step {step_num})", expanded=False):
                        st.code(code_str, language="python")

                elif event_type == "output":
                    out_str = event["content"]
                    st.caption(f"Output (Step {step_num}):")
                    st.code(out_str, language="text")

                elif event_type == "error":
                    err_str = event["content"]
                    st.caption(f"Step {step_num} Retry (Traceback):")
                    st.code(err_str, language="text")

                elif event_type == "chart":
                    fig = event["figure"]
                    styled_fig = apply_plotly_minimal_theme(fig)
                    current_turn["charts"].append(styled_fig)

                elif event_type == "ask_user":
                    clarify_q = event["question"]
                    current_turn["ask_user"] = clarify_q
                    st.warning(f"Clarification needed: {clarify_q}")

                elif event_type == "final":
                    current_turn["final"] = {
                        "summary": event.get("summary", ""),
                        "key_findings": event.get("key_findings", []),
                        "confidence": event.get("confidence", "medium"),
                    }
                    if event.get("charts"):
                        for c in event["charts"]:
                            if c not in current_turn["charts"]:
                                current_turn["charts"].append(apply_plotly_minimal_theme(c))

            status_box.update(label="Analysis complete", state="complete", expanded=False)

        # Store in session state history
        st.session_state.history.append(current_turn)
        st.rerun()

# Render Conversation History & Latest Results
if st.session_state.history:
    for idx, turn in enumerate(reversed(st.session_state.history)):
        turn_num = len(st.session_state.history) - idx
        st.markdown(f"#### Question: {turn['question']}")

        if turn.get("ask_user"):
            st.info(f"Agent requested clarification: {turn['ask_user']}")

        if turn.get("final"):
            final_data = turn["final"]
            confidence_badge = render_confidence_badge(final_data.get("confidence", "medium"))
            
            # Final summary & key findings card
            findings_bullets = "".join(f"<li>{kf}</li>" for kf in final_data.get("key_findings", []))
            
            st.markdown(
                f"""
                <div class="final-card">
                    <div>{confidence_badge}</div>
                    <div class="final-summary">{final_data.get('summary', '')}</div>
                    <div class="findings-title">Key Findings</div>
                    <ul>{findings_bullets}</ul>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Display any interactive charts generated
            if turn.get("charts"):
                for chart_idx, fig in enumerate(turn["charts"]):
                    st.plotly_chart(fig, use_container_width=True, key=f"turn_{turn_num}_chart_{chart_idx}")

            # Download report button
            report_md = generate_markdown_report(
                question=turn["question"],
                summary=final_data.get("summary", ""),
                key_findings=final_data.get("key_findings", []),
                confidence=final_data.get("confidence", "medium"),
                code_snippets=turn.get("code_snippets", []),
            )

            st.download_button(
                label="Download report (.md)",
                data=report_md,
                file_name=f"analyst_report_{turn_num}.md",
                mime="text/markdown",
                key=f"dl_rep_{turn_num}",
            )

        st.markdown("---")
