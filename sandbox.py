"""Python execution sandbox for Analyst Agent.

SECURITY NOTICE:
This module provides a best-effort in-process/sub-process Python execution sandbox
designed to prevent accidental system modifications, runaway processes, and common
malicious patterns. It utilizes AST parsing, forbidden token inspection, restricted
namespaces, safe builtins, and process-level timeouts.

HOWEVER, THIS IS A BEST-EFFORT SANDBOX AND DOES NOT PROVIDE AN AIRTIGHT OS-LEVEL
SECURITY GUARANTEE. For multi-tenant hostile production environments, isolated container
runtimes (such as gVisor, Firecracker, or Docker sandboxes) must be used.
"""

import ast
import io
import math
import multiprocessing as mp
import re
import sys
import traceback
from typing import Any, Optional
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

TIMEOUT_SECONDS = 10
MAX_OUTPUT_LENGTH = 4000

# Prohibited module names and tokens
BLOCKED_MODULES = {
    "os", "sys", "subprocess", "socket", "shutil", "requests", "urllib",
    "pathlib", "http", "ftplib", "telnetlib", "posix", "pty", "platform",
    "importlib", "builtins", "_thread", "threading", "multiprocessing",
    "ctypes", "pickle", "shelve", "dbm", "pip", "pkg_resources"
}

SAFE_IMPORT_WHITELIST = {
    "math", "datetime", "pandas", "numpy", "plotly", "json", "re",
    "collections", "itertools", "random", "calendar"
}

# Safe built-in functions
_real_import = __builtins__["__import__"] if isinstance(__builtins__, dict) else __builtins__.__import__


def safe_import(name, globals=None, locals=None, fromlist=(), level=0):
    """Restricted __import__ allowing only whitelisted safe modules."""
    root_pkg = name.split(".")[0]
    if root_pkg in BLOCKED_MODULES or root_pkg not in SAFE_IMPORT_WHITELIST:
        raise ImportError(f"Security Error: Importing '{name}' is prohibited in the sandbox.")
    return _real_import(name, globals, locals, fromlist, level)


SAFE_BUILTINS = {
    "__import__": safe_import,
    "abs": abs,
    "all": all,
    "any": any,
    "bin": bin,
    "bool": bool,
    "bytearray": bytearray,
    "bytes": bytes,
    "chr": chr,
    "complex": complex,
    "dict": dict,
    "divmod": divmod,
    "enumerate": enumerate,
    "filter": filter,
    "float": float,
    "format": format,
    "frozenset": frozenset,
    "hash": hash,
    "hex": hex,
    "int": int,
    "isinstance": isinstance,
    "issubclass": issubclass,
    "iter": iter,
    "len": len,
    "list": list,
    "map": map,
    "max": max,
    "min": min,
    "next": next,
    "oct": oct,
    "ord": ord,
    "pow": pow,
    "print": print,
    "range": range,
    "repr": repr,
    "reversed": reversed,
    "round": round,
    "set": set,
    "slice": slice,
    "sorted": sorted,
    "str": str,
    "sum": sum,
    "tuple": tuple,
    "type": type,
    "zip": zip,
    "None": None,
    "True": True,
    "False": False,
}


class SecurityValidationError(Exception):
    """Raised when code violates security sandbox policies."""
    pass


def validate_code_ast(code: str) -> None:
    """Analyze the Python code AST for blocked modules, dunder access, and dangerous operations.

    Args:
        code: Python source code string.

    Raises:
        SecurityValidationError: If malicious or forbidden constructs are detected.
    """
    # Regex check for obvious dunder access patterns (e.g. __class__, __subclasses__)
    dunder_match = re.search(r"__[a-zA-Z0-9_]+__", code)
    if dunder_match:
        raise SecurityValidationError(
            f"Security Error: Access to private dunder attribute '{dunder_match.group(0)}' is prohibited."
        )

    # Check for direct occurrences of blocked module keywords in words
    for blocked in BLOCKED_MODULES:
        pattern = r"\b" + re.escape(blocked) + r"\b"
        if re.search(pattern, code):
            # Check if it's an import or dangerous reference
            raise SecurityValidationError(
                f"Security Error: Reference to blocked module or keyword '{blocked}' is prohibited."
            )

    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        # Let syntax errors be caught during execution or raised cleanly
        return

    for node in ast.walk(tree):
        # Prohibit Import and ImportFrom if not in whitelist
        if isinstance(node, ast.Import):
            for alias in node.names:
                root_pkg = alias.name.split(".")[0]
                if root_pkg in BLOCKED_MODULES or root_pkg not in SAFE_IMPORT_WHITELIST:
                    raise SecurityValidationError(
                        f"Security Error: Importing module '{alias.name}' is prohibited."
                    )
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                root_pkg = node.module.split(".")[0]
                if root_pkg in BLOCKED_MODULES or root_pkg not in SAFE_IMPORT_WHITELIST:
                    raise SecurityValidationError(
                        f"Security Error: Importing from '{node.module}' is prohibited."
                    )

        # Prohibit access to __ attributes
        elif isinstance(node, ast.Attribute):
            if node.attr.startswith("__"):
                raise SecurityValidationError(
                    f"Security Error: Access to attribute '{node.attr}' is prohibited."
                )

        # Prohibit calls to eval, exec, compile, open, input
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                if node.func.id in {"eval", "exec", "compile", "open", "input", "__import__", "globals", "locals"}:
                    raise SecurityValidationError(
                        f"Security Error: Calling '{node.func.id}' is prohibited."
                    )


def _worker_exec(df: pd.DataFrame, code: str, queue: Any) -> None:
    """Worker function executed in an isolated process.

    Args:
        df: Copy of the target DataFrame.
        code: Python code to execute.
        queue: multiprocessing.Queue to send execution results back.
    """
    stdout_capture = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = stdout_capture

    # Build safe execution namespace
    exec_globals = {
        "__builtins__": SAFE_BUILTINS,
        "pd": pd,
        "np": np,
        "px": px,
        "go": go,
        "math": math,
        "df": df.copy(deep=True),
        "result": None,
        "fig": None,
    }

    error_msg: Optional[str] = None
    result_val: Any = None
    fig_val: Any = None

    try:
        exec(code, exec_globals)
        result_val = exec_globals.get("result")
        fig_obj = exec_globals.get("fig")

        # Verify if fig is a plotly figure
        if fig_obj is not None:
            if isinstance(fig_obj, (go.Figure, dict)) or hasattr(fig_obj, "to_dict"):
                fig_val = fig_obj
            else:
                stdout_capture.write(f"\n[Warning: 'fig' is of type {type(fig_obj).__name__}, expected plotly Figure]\n")

    except Exception:
        error_msg = traceback.format_exc()
    finally:
        sys.stdout = old_stdout

    raw_stdout = stdout_capture.getvalue()
    updated_df = exec_globals.get("df") if error_msg is None and isinstance(exec_globals.get("df"), pd.DataFrame) else None
    queue.put({
        "stdout": raw_stdout,
        "result": result_val,
        "fig": fig_val,
        "error": error_msg,
        "df": updated_df,
    })


def run_code_sandbox(df: pd.DataFrame, code: str, timeout: int = TIMEOUT_SECONDS) -> dict[str, Any]:
    """Execute pandas/numpy/plotly code in a sandboxed process with a strict timeout.

    Args:
        df: DataFrame available as `df` inside the sandbox.
        code: Python code to execute.
        timeout: Maximum execution time in seconds (default: 10).

    Returns:
        Dictionary with keys:
            - success: bool
            - stdout: str (truncated to MAX_OUTPUT_LENGTH)
            - result: str or Any
            - fig: Plotly Figure object or None
            - error: str or None
            - output_str: Combined readable output for the agent
    """
    # Step 1: Static AST & Token Validation
    try:
        validate_code_ast(code)
    except SecurityValidationError as sec_err:
        err_text = str(sec_err)
        return {
            "success": False,
            "stdout": "",
            "result": None,
            "fig": None,
            "error": err_text,
            "output_str": err_text,
        }

    # Step 2: Multi-process execution
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    proc = ctx.Process(target=_worker_exec, args=(df, code, queue))

    proc.start()
    proc.join(timeout=timeout)

    if proc.is_alive():
        proc.terminate()
        proc.join(timeout=1)
        if proc.is_alive():
            proc.kill()
        timeout_err = f"TimeoutError: Code execution exceeded {timeout} seconds limit and was terminated."
        return {
            "success": False,
            "stdout": "",
            "result": None,
            "fig": None,
            "error": timeout_err,
            "output_str": timeout_err,
        }

    if not queue.empty():
        exec_res = queue.get()
        stdout = exec_res.get("stdout", "")
        result = exec_res.get("result")
        fig = exec_res.get("fig")
        error = exec_res.get("error")
        res_df = exec_res.get("df")
    else:
        stdout = ""
        result = None
        fig = None
        error = "Process terminated unexpectedly without returning output."
        res_df = None

    # Format result if provided
    result_str = ""
    if result is not None:
        if isinstance(result, (pd.DataFrame, pd.Series)):
            result_str = str(result)
        else:
            result_str = repr(result)

    # Combine output
    combined = []
    if stdout:
        combined.append(stdout.strip())
    if result_str:
        combined.append(f"Result: {result_str}")

    combined_output = "\n".join(combined)

    # Truncate if exceeds limit
    if len(combined_output) > MAX_OUTPUT_LENGTH:
        truncated_count = len(combined_output) - MAX_OUTPUT_LENGTH
        combined_output = combined_output[:MAX_OUTPUT_LENGTH] + f"\n... [Output truncated; {truncated_count} characters omitted]"

    if error:
        success = False
        final_output = f"Execution Error:\n{error}"
        if len(final_output) > MAX_OUTPUT_LENGTH:
            final_output = final_output[:MAX_OUTPUT_LENGTH] + "\n... [Traceback truncated]"
    else:
        success = True
        final_output = combined_output if combined_output else "[Code executed successfully with no output]"

    return {
        "success": success,
        "stdout": stdout,
        "result": result,
        "fig": fig,
        "error": error,
        "output_str": final_output,
        "df": res_df if res_df is not None else df,
    }
