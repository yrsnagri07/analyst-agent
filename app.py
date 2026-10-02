"""Analyst Agent: Autonomous AI Data Analyst Streamlit Application."""

import os
from pathlib import Path
from dotenv import load_dotenv
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

load_dotenv()

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
    if st.button("Use sample data", help="Load the 12-month synthetic sales dataset", use_container_width=True):
        sample_path = Path("sample_data/sales.csv")
        if sample_path.exists():
            try:
                loaded_df = load_csv(str(sample_path), "sales.csv")
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
        loaded_df = load_csv(uploaded_file, uploaded_file.name)
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


def ensure_contextual_chart(df: pd.DataFrame, question: str, existing_charts: list) -> list:
    """Ensure at least one clean interactive Plotly figure exists for the analysis."""
    if existing_charts:
        return existing_charts

    q = question.lower()
    date_cols = [c for c in df.columns if "date" in c.lower()]
    rev_cols = [c for c in df.columns if any(k in c.lower() for k in ["revenue", "sales", "amount", "total"])]
    cat_cols = [c for c in df.columns if any(k in c.lower() for k in ["category", "product", "region"])]

    try:
        # Case 1: Temporal / March drop query
        if ("march" in q or "month" in q or "drop" in q or "trend" in q) and date_cols and rev_cols:
            temp = df.copy()
            dt_s = pd.to_datetime(temp[date_cols[0]])
            temp["month_name"] = dt_s.dt.strftime("%b")
            temp["month_num"] = dt_s.dt.month
            
            group_keys = ["month_num", "month_name"]
            color_key = None
            if "region" in df.columns:
                group_keys.append("region")
                color_key = "region"
            elif "category" in df.columns:
                group_keys.append("category")
                color_key = "category"

            monthly = temp.groupby(group_keys)[rev_cols[0]].sum().reset_index()
            monthly = monthly.sort_values("month_num")
            
            fig = px.bar(
                monthly,
                x="month_name",
                y=rev_cols[0],
                color=color_key,
                barmode="group",
                title=f"Monthly {rev_cols[0].capitalize()} Distribution ({'by ' + color_key.capitalize() if color_key else ''})",
                labels={rev_cols[0]: rev_cols[0].capitalize(), "month_name": "Month"}
            )
            return [apply_plotly_minimal_theme(fig)]

        # Case 2: Category / Product query
        if ("category" in q or "product" in q or "highest" in q) and cat_cols and rev_cols:
            top_cat = cat_cols[0]
            cat_totals = df.groupby(top_cat)[rev_cols[0]].sum().reset_index().sort_values(rev_cols[0], ascending=False)
            fig = px.bar(
                cat_totals,
                x=top_cat,
                y=rev_cols[0],
                color=top_cat,
                title=f"Total {rev_cols[0].capitalize()} by {top_cat.capitalize()}",
                labels={rev_cols[0]: rev_cols[0].capitalize()}
            )
            return [apply_plotly_minimal_theme(fig)]

        # Case 3: Regional query
        if "region" in q and "region" in df.columns and rev_cols:
            reg_totals = df.groupby("region")[rev_cols[0]].sum().reset_index().sort_values(rev_cols[0], ascending=False)
            fig = px.bar(
                reg_totals,
                x="region",
                y=rev_cols[0],
                color="region",
                title=f"{rev_cols[0].capitalize()} by Region"
            )
            return [apply_plotly_minimal_theme(fig)]

    except Exception:
        pass

    return existing_charts


# Render Dataset Card & Interactive KPIs if data is loaded
if st.session_state.df is not None and st.session_state.df_profile is not None:
    prof = st.session_state.df_profile
    df_active = st.session_state.df

    rev_cols = [c for c in df_active.columns if any(k in c.lower() for k in ["revenue", "sales", "amount"])]
    primary_metric_label = f"Total {rev_cols[0].capitalize()}" if rev_cols else "Total Rows"
    primary_metric_val = f"${df_active[rev_cols[0]].sum():,.0f}" if rev_cols else f"{prof['rows']:,}"

    date_coverage = "N/A"
    if prof.get("date_columns"):
        dc = prof["date_columns"][0]
        date_coverage = f"{dc['min'][:7]} &rarr; {dc['max'][:7]}"

    # Interactive KPI Grid
    st.markdown(
        f"""
        <div class="kpi-grid">
            <div class="kpi-card">
                <div class="kpi-label">Dataset Records</div>
                <div class="kpi-value">{prof['rows']:,} rows</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Columns</div>
                <div class="kpi-value">{prof['columns']} fields</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">{primary_metric_label}</div>
                <div class="kpi-value">{primary_metric_val}</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Date Coverage</div>
                <div class="kpi-value">{date_coverage}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    chip_html = "".join(
        f'<span class="col-chip">{col["name"]}<span class="col-type">{col["dtype"]}</span></span>'
        for col in prof["columns_info"]
    )
    st.markdown(
        f"""
        <div class="dataset-card" style="margin-top: 0; padding-top: 0.75rem;">
            <div class="dataset-stats">Schema Details &bull; Memory: {prof['memory_mb']} MB</div>
            <div class="chip-container">{chip_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("Preview data records (first 10 rows)", expanded=False):
        st.dataframe(df_active.head(10), use_container_width=True)

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
        status_box = st.status("Analyst Agent investigating data...", expanded=True)

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
                    current_turn["steps"].append({"step": step_num, "code": code_str, "output": None, "error": None})
                    st.markdown(f"**Step {step_num}/8: Executing query**")
                    with st.expander(f"Code (Step {step_num})", expanded=False):
                        st.code(code_str, language="python")

                elif event_type == "output":
                    out_str = event["content"]
                    if current_turn["steps"]:
                        current_turn["steps"][-1]["output"] = out_str
                    st.caption(f"Output (Step {step_num}):")
                    st.code(out_str, language="text")

                elif event_type == "error":
                    err_str = event["content"]
                    if current_turn["steps"]:
                        current_turn["steps"][-1]["error"] = err_str
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
                        "confidence": event.get("confidence", "high"),
                    }
                    if event.get("charts"):
                        for c in event["charts"]:
                            if c not in current_turn["charts"]:
                                current_turn["charts"].append(apply_plotly_minimal_theme(c))

            status_box.update(label="Analysis verified and complete", state="complete", expanded=False)

        current_turn["charts"] = ensure_contextual_chart(st.session_state.df, query_text, current_turn["charts"])
        st.session_state.history.append(current_turn)
        st.rerun()

# Render Conversation History & Interactive Results
if st.session_state.history:
    for idx, turn in enumerate(reversed(st.session_state.history)):
        turn_num = len(st.session_state.history) - idx
        st.markdown(f"#### Question: {turn['question']}")

        if turn.get("ask_user"):
            st.info(f"Agent requested clarification: {turn['ask_user']}")

        if turn.get("final"):
            final_data = turn["final"]
            confidence_badge = render_confidence_badge(final_data.get("confidence", "high"))
            findings_html = "".join(f'<div class="finding-item">{kf}</div>' for kf in final_data.get("key_findings", []))
            
            st.markdown(
                f"""
                <div class="final-card">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span style="font-weight: 600; font-size: 0.9rem; text-transform: uppercase; letter-spacing: 0.05em; color: #2563EB;">Executive Summary</span>
                        {confidence_badge}
                    </div>
                    <div class="final-summary">{final_data.get('summary', '')}</div>
                    <div class="findings-title">Key Empirical Findings</div>
                    {findings_html}
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Interactive Plotly Charts
            if turn.get("charts"):
                for chart_idx, fig in enumerate(turn["charts"]):
                    st.plotly_chart(fig, use_container_width=True, key=f"turn_{turn_num}_chart_{chart_idx}")

            # Expandable Reasoning Timeline
            if turn.get("steps") or turn.get("plan"):
                with st.expander(f"Inspect Agent Reasoning & Executed Code ({len(turn['steps'])} step{'s' if len(turn['steps']) > 1 else ''})", expanded=False):
                    if turn.get("plan"):
                        st.markdown(f"**Initial Plan:**\n{turn['plan']}")
                        st.markdown("---")
                    for s in turn.get("steps", []):
                        st.markdown(f"**Step {s['step']} / 8**")
                        if s.get("code"):
                            st.code(s["code"], language="python")
                        if s.get("output"):
                            st.caption("Output:")
                            st.code(s["output"], language="text")
                        if s.get("error"):
                            st.caption("Retry Reason:")
                            st.code(s["error"], language="text")

            col_dl, _ = st.columns([1, 2])
            with col_dl:
                report_md = generate_markdown_report(
                    question=turn["question"],
                    summary=final_data.get("summary", ""),
                    key_findings=final_data.get("key_findings", []),
                    confidence=final_data.get("confidence", "high"),
                    code_snippets=turn.get("code_snippets", []),
                )
                st.download_button(
                    label="Download report (.md)",
                    data=report_md,
                    file_name=f"analyst_report_{turn_num}.md",
                    mime="text/markdown",
                    key=f"dl_rep_{turn_num}",
                    use_container_width=True,
                )

            st.markdown(
                """
                <div class="followup-container">
                    <div class="followup-label">Suggested Follow-ups</div>
                </div>
                """,
                unsafe_allow_html=True
            )
            fu_cols = st.columns(3)
            follow_ups = [
                "Which product suffered the largest drop?",
                "Compare North vs South sales in Q1",
                "Break down March revenue by category"
            ]
            for fu_idx, fu_q in enumerate(follow_ups):
                with fu_cols[fu_idx]:
                    if st.button(fu_q, key=f"fu_{turn_num}_{fu_idx}", use_container_width=True):
                        st.session_state.input_query = fu_q
                        st.rerun()

        st.markdown("---")
