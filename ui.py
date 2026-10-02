"""UI styling, theme utilities, and presentation helpers for Analyst Agent."""

import json
from datetime import datetime
from typing import Any
import pandas as pd
import plotly.graph_objects as go

# Minimal, clean CSS injection
CUSTOM_CSS = """
<style>
/* Hide Streamlit chrome */
#MainMenu, header, footer, div[data-testid="stDecoration"] {
    visibility: hidden !important;
    height: 0 !important;
    margin: 0 !important;
    padding: 0 !important;
}

/* Center single column layout (max width 760px) */
.block-container {
    max-width: 760px !important;
    padding-top: 2.5rem !important;
    padding-bottom: 5rem !important;
    padding-left: 1.25rem !important;
    padding-right: 1.25rem !important;
    margin: 0 auto !important;
}

/* System font stack */
html, body, [class*="css"], .stMarkdown, .stButton, .stTextInput, .stAlert {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif !important;
    letter-spacing: -0.01em;
}

/* Header typography */
.analyst-header {
    margin-bottom: 2rem;
    padding-bottom: 1rem;
    border-bottom: 1px solid rgba(128, 128, 128, 0.15);
}
.analyst-title {
    font-size: 1.75rem;
    font-weight: 300;
    color: inherit;
    margin: 0 0 0.25rem 0;
    letter-spacing: -0.02em;
}
.analyst-subtitle {
    font-size: 0.875rem;
    color: rgba(128, 128, 128, 0.9);
    margin: 0;
    font-weight: 400;
}

/* Dataset Card */
.dataset-card {
    border: 1px solid rgba(128, 128, 128, 0.2);
    border-radius: 8px;
    padding: 1rem 1.25rem;
    margin-top: 1rem;
    margin-bottom: 1.5rem;
    background: transparent;
}
.dataset-stats {
    font-size: 0.8125rem;
    color: rgba(128, 128, 128, 0.85);
    margin-bottom: 0.75rem;
    font-weight: 500;
}
.chip-container {
    display: flex;
    flex-wrap: wrap;
    gap: 0.375rem;
    margin-top: 0.5rem;
}
.col-chip {
    display: inline-flex;
    align-items: center;
    background: rgba(128, 128, 128, 0.08);
    border: 1px solid rgba(128, 128, 128, 0.15);
    border-radius: 4px;
    padding: 0.2rem 0.5rem;
    font-size: 0.75rem;
    color: inherit;
    font-family: monospace;
}
.col-type {
    margin-left: 0.35rem;
    color: rgba(128, 128, 128, 0.75);
    font-size: 0.7rem;
}

/* Badges */
.badge-confidence-high {
    display: inline-block;
    padding: 0.2rem 0.6rem;
    border-radius: 12px;
    font-size: 0.75rem;
    font-weight: 500;
    background: rgba(16, 185, 129, 0.12);
    color: #059669;
    border: 1px solid rgba(16, 185, 129, 0.25);
}
.badge-confidence-medium {
    display: inline-block;
    padding: 0.2rem 0.6rem;
    border-radius: 12px;
    font-size: 0.75rem;
    font-weight: 500;
    background: rgba(245, 158, 11, 0.12);
    color: #D97706;
    border: 1px solid rgba(245, 158, 11, 0.25);
}
.badge-confidence-low {
    display: inline-block;
    padding: 0.2rem 0.6rem;
    border-radius: 12px;
    font-size: 0.75rem;
    font-weight: 500;
    background: rgba(239, 68, 68, 0.12);
    color: #DC2626;
    border: 1px solid rgba(239, 68, 68, 0.25);
}

/* KPI Grid */
.kpi-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
    gap: 0.75rem;
    margin-top: 1rem;
    margin-bottom: 1.25rem;
}
.kpi-card {
    background: rgba(128, 128, 128, 0.05);
    border: 1px solid rgba(128, 128, 128, 0.15);
    border-radius: 8px;
    padding: 0.75rem 1rem;
    transition: transform 0.15s ease, border-color 0.15s ease;
}
.kpi-card:hover {
    border-color: rgba(37, 99, 235, 0.4);
    transform: translateY(-1px);
}
.kpi-label {
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: rgba(128, 128, 128, 0.85);
    margin-bottom: 0.25rem;
    font-weight: 600;
}
.kpi-value {
    font-size: 1.15rem;
    font-weight: 600;
    color: inherit;
    letter-spacing: -0.02em;
}

/* Final Answer Box */
.final-card {
    border: 1px solid rgba(37, 99, 235, 0.3);
    border-radius: 10px;
    padding: 1.5rem;
    margin-top: 1.25rem;
    margin-bottom: 1.5rem;
    background: rgba(37, 99, 235, 0.02);
    box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.05);
}
.final-summary {
    font-size: 1rem;
    line-height: 1.65;
    margin-top: 0.875rem;
    margin-bottom: 1.25rem;
    color: inherit;
}
.findings-title {
    font-size: 0.8rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: rgba(128, 128, 128, 0.9);
    margin-top: 1rem;
    margin-bottom: 0.6rem;
}
.finding-item {
    background: rgba(128, 128, 128, 0.04);
    border-left: 3px solid #2563EB;
    border-radius: 0 6px 6px 0;
    padding: 0.6rem 0.875rem;
    margin-bottom: 0.5rem;
    font-size: 0.9rem;
    line-height: 1.5;
}

/* Follow-up suggestions */
.followup-container {
    margin-top: 1.25rem;
    padding-top: 1rem;
    border-top: 1px dashed rgba(128, 128, 128, 0.2);
}
.followup-label {
    font-size: 0.75rem;
    font-weight: 600;
    color: rgba(128, 128, 128, 0.8);
    text-transform: uppercase;
    margin-bottom: 0.5rem;
}

/* Ghost button styling */
div[data-testid="stHorizontalBlock"] button {
    font-size: 0.8rem !important;
    border-radius: 4px !important;
    padding: 0.35rem 0.6rem !important;
    font-weight: 400 !important;
}

/* Code block aesthetic */
pre {
    border-radius: 6px !important;
    font-size: 0.8rem !important;
}
</style>
"""


def apply_plotly_minimal_theme(fig: go.Figure) -> go.Figure:
    """Apply a calm, minimal theme to Plotly figures with a single accent color and clean axes.

    Args:
        fig: Plotly figure object.

    Returns:
        Styled Plotly figure.
    """
    fig.update_layout(
        font=dict(family="-apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif", size=12),
        margin=dict(l=40, r=20, t=40, b=40),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        hoverlabel=dict(bgcolor="white", font_size=12),
        showlegend=True,
    )
    fig.update_xaxes(
        showgrid=False,
        showline=True,
        linecolor="rgba(128, 128, 128, 0.3)",
        linewidth=1,
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor="rgba(128, 128, 128, 0.1)",
        showline=False,
    )
    return fig


def render_confidence_badge(confidence: str) -> str:
    """Render an HTML confidence pill badge.

    Args:
        confidence: 'low', 'medium', or 'high'.

    Returns:
        HTML string.
    """
    conf = (confidence or "medium").lower()
    if conf not in ("low", "medium", "high"):
        conf = "medium"
    return f'<span class="badge-confidence-{conf}">Confidence: {conf.capitalize()}</span>'


def generate_markdown_report(
    question: str,
    summary: str,
    key_findings: list[str],
    confidence: str,
    code_snippets: list[str],
) -> str:
    """Generate a clean Markdown download report for the analysis.

    Args:
        question: User query.
        summary: Executive summary from agent.
        key_findings: Bullet points.
        confidence: Assessment rating.
        code_snippets: List of python code blocks run during analysis.

    Returns:
        Formatted Markdown text.
    """
    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    lines = [
        f"# Analysis Report: {question}",
        f"\n**Generated**: {timestamp}  ",
        f"**Confidence**: {confidence.capitalize()}  ",
        "\n## Summary",
        summary,
        "\n## Key Findings",
    ]
    for kf in key_findings:
        lines.append(f"- {kf}")

    if code_snippets:
        lines.append("\n## Executed Code & Investigation Steps")
        for i, code in enumerate(code_snippets, start=1):
            lines.append(f"### Step {i}")
            lines.append("```python")
            lines.append(code.strip())
            lines.append("```")

    lines.append("\n---")
    lines.append("*Generated by Analyst Agent*")
    return "\n".join(lines)
