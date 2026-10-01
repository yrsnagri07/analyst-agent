"""Unit tests for Python execution sandbox."""

import pytest
import pandas as pd
from sandbox import run_code_sandbox, SecurityValidationError, validate_code_ast


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "category": ["A", "B", "A", "B"],
        "revenue": [100.0, 200.0, 150.0, 250.0],
        "units": [1, 2, 1, 3]
    })


def test_sandbox_normal_execution(sample_df):
    """Test valid pandas/numpy aggregation and print output."""
    code = """
total_rev = df['revenue'].sum()
print(f"Total: {total_rev}")
result = {'total': total_rev}
"""
    res = run_code_sandbox(sample_df, code)
    assert res["success"] is True
    assert "Total: 700.0" in res["stdout"]
    assert res["result"] == {"total": 700.0}
    assert res["error"] is None


def test_sandbox_blocked_os_import(sample_df):
    """Test that importing or referencing 'os' is blocked."""
    code = "import os; print(os.getcwd())"
    res = run_code_sandbox(sample_df, code)
    assert res["success"] is False
    assert "Security Error" in res["error"]
    assert "os" in res["error"]


def test_sandbox_blocked_sys_import(sample_df):
    """Test that importing or referencing 'sys' is blocked."""
    code = "import sys; sys.exit(1)"
    res = run_code_sandbox(sample_df, code)
    assert res["success"] is False
    assert "Security Error" in res["error"]


def test_sandbox_blocked_subprocess(sample_df):
    """Test that subprocess calls are blocked."""
    code = "import subprocess; subprocess.run(['ls'])"
    res = run_code_sandbox(sample_df, code)
    assert res["success"] is False
    assert "Security Error" in res["error"]


def test_sandbox_blocked_socket(sample_df):
    """Test that socket network access is blocked."""
    code = "import socket; s = socket.socket()"
    res = run_code_sandbox(sample_df, code)
    assert res["success"] is False
    assert "Security Error" in res["error"]


def test_sandbox_blocked_dunder_access(sample_df):
    """Test that accessing dunder attributes like __class__ or __subclasses__ is blocked."""
    code = "x = ().__class__.__bases__[0].__subclasses__()"
    res = run_code_sandbox(sample_df, code)
    assert res["success"] is False
    assert "Security Error" in res["error"]
    assert "dunder" in res["error"].lower()


def test_sandbox_blocked_open_builtin(sample_df):
    """Test that open() is blocked."""
    code = "with open('/tmp/test.txt', 'w') as f: f.write('hello')"
    res = run_code_sandbox(sample_df, code)
    assert res["success"] is False
    assert "Security Error" in res["error"] or "open" in res["error"]


def test_sandbox_timeout(sample_df):
    """Test that infinite loops or long-running execution are killed after timeout."""
    code = "while True: pass"
    res = run_code_sandbox(sample_df, code, timeout=1)
    assert res["success"] is False
    assert "TimeoutError" in res["error"]


def test_sandbox_output_truncation(sample_df):
    """Test that massive outputs are truncated to 4,000 characters."""
    code = "print('X' * 6000)"
    res = run_code_sandbox(sample_df, code)
    assert res["success"] is True
    assert "Output truncated" in res["output_str"]
    assert len(res["output_str"]) <= 4200


def test_sandbox_plotly_figure_generation(sample_df):
    """Test that plotly figures can be created and returned."""
    code = """
import plotly.express as px
fig = px.bar(df, x='category', y='revenue', title='Revenue by Category')
"""
    res = run_code_sandbox(sample_df, code)
    assert res["success"] is True
    assert res["fig"] is not None
