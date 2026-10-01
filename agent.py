"""Agent core: Hand-crafted ReAct loop with Gemini function calling and sandboxed execution."""

import os
from typing import Any, Generator, Optional
import pandas as pd
from google import genai
from google.genai import types

from profiling import profile_dataframe, format_profile_for_llm
from prompts import SYSTEM_PROMPT, build_initial_prompt
from sandbox import run_code_sandbox

MAX_STEPS = 8
MAX_CONSECUTIVE_ERRORS = 3
DEFAULT_MODEL = "gemini-2.5-flash"


def get_gemini_api_key() -> Optional[str]:
    """Retrieve Gemini API key from Streamlit secrets, then environment variable."""
    # 1. Try Streamlit secrets
    try:
        import streamlit as st
        if hasattr(st, "secrets") and "GEMINI_API_KEY" in st.secrets:
            return st.secrets["GEMINI_API_KEY"]
    except Exception:
        pass

    # 2. Try OS environment variable
    return os.environ.get("GEMINI_API_KEY")


def get_gemini_model_name() -> str:
    """Retrieve configured Gemini model name from secrets, env, or default."""
    try:
        import streamlit as st
        if hasattr(st, "secrets") and "GEMINI_MODEL" in st.secrets:
            return st.secrets["GEMINI_MODEL"]
    except Exception:
        pass

    return os.environ.get("GEMINI_MODEL", DEFAULT_MODEL)


def _get_agent_tools() -> list[types.Tool]:
    """Construct explicit tool declarations for Gemini native function calling."""
    return [
        types.Tool(function_declarations=[
            types.FunctionDeclaration(
                name="run_python",
                description=(
                    "Executes Python data analysis code (pandas, numpy, plotly) in a secure sandbox. "
                    "The DataFrame is available as `df`. You may print stdout, assign a value/summary to `result`, "
                    "or assign a Plotly figure to `fig`. Returns execution output, `result`, or error traceback."
                ),
                parameters=types.Schema(
                    type="OBJECT",
                    properties={
                        "code": types.Schema(
                            type="STRING",
                            description="Python code to execute using df, pd, np, px, go."
                        )
                    },
                    required=["code"]
                )
            ),
            types.FunctionDeclaration(
                name="final_answer",
                description=(
                    "Concludes the analysis when sufficient evidence has been collected. "
                    "All numbers in summary and key_findings must strictly originate from executed code."
                ),
                parameters=types.Schema(
                    type="OBJECT",
                    properties={
                        "summary": types.Schema(
                            type="STRING",
                            description="Executive 1-3 paragraph summary answering the user question."
                        ),
                        "key_findings": types.Schema(
                            type="ARRAY",
                            items=types.Schema(type="STRING"),
                            description="List of 3-6 specific data-backed findings with concrete metrics."
                        ),
                        "confidence": types.Schema(
                            type="STRING",
                            enum=["low", "medium", "high"],
                            description="Assessment of evidence confidence: 'low', 'medium', or 'high'."
                        )
                    },
                    required=["summary", "key_findings", "confidence"]
                )
            ),
            types.FunctionDeclaration(
                name="ask_user",
                description=(
                    "Only for truly ambiguous questions where analysis cannot proceed without human clarification. "
                    "Pauses the agent loop."
                ),
                parameters=types.Schema(
                    type="OBJECT",
                    properties={
                        "question": types.Schema(
                            type="STRING",
                            description="Clarifying question to ask the user."
                        )
                    },
                    required=["question"]
                )
            )
        ])
    ]


def run_agent_loop(
    df: pd.DataFrame,
    question: str,
    chat_history: Optional[list[dict[str, Any]]] = None,
    api_key: Optional[str] = None,
    model_name: Optional[str] = None,
    df_profile: Optional[dict[str, Any]] = None,
) -> Generator[dict[str, Any], None, None]:
    """Execute the ReAct data analyst loop on a DataFrame.

    Yields event dictionaries with types:
        - {"type": "plan", "content": str}
        - {"type": "code", "content": str, "step": int}
        - {"type": "output", "content": str, "step": int}
        - {"type": "error", "content": str, "step": int}
        - {"type": "chart", "figure": Any, "step": int}
        - {"type": "ask_user", "question": str, "step": int}
        - {"type": "final", "summary": str, "key_findings": list[str], "confidence": str, "charts": list[Any]}

    Args:
        df: Pandas DataFrame to analyze.
        question: User query string.
        chat_history: Optional prior chat history turns.
        api_key: Optional Gemini API Key (falls back to secrets / env).
        model_name: Optional Gemini model name (falls back to secrets / env).
        df_profile: Optional precomputed profile dictionary.
    """
    key = api_key or get_gemini_api_key()
    if not key:
        yield {
            "type": "error",
            "step": 0,
            "content": (
                "Missing Gemini API Key. Please provide a valid Gemini API Key in the UI "
                "or set the GEMINI_API_KEY environment variable."
            )
        }
        return

    chosen_model = model_name or get_gemini_model_name()

    # Step 1: Profile the dataframe
    if df_profile is None:
        df_profile = profile_dataframe(df)
    profile_text = format_profile_for_llm(df_profile)

    # Initialize Gemini Client
    client = genai.Client(api_key=key)

    # Configure tool calling
    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        tools=_get_agent_tools(),
        temperature=0.1,
    )

    initial_prompt = build_initial_prompt(profile_text, question, chat_history)

    contents: list[types.Content] = [
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=initial_prompt)]
        )
    ]

    consecutive_errors = 0
    step = 0
    collected_charts = []
    has_planned = False

    while step < MAX_STEPS:
        step += 1

        try:
            response = client.models.generate_content(
                model=chosen_model,
                contents=contents,
                config=config,
            )
        except Exception as e:
            yield {
                "type": "error",
                "step": step,
                "content": f"Gemini API request failed: {str(e)}"
            }
            return

        candidate = response.candidates[0] if response.candidates else None
        if not candidate or not candidate.content:
            yield {
                "type": "error",
                "step": step,
                "content": "Model returned an empty response. Stopping loop."
            }
            return

        model_content = candidate.content
        parts = model_content.parts or []

        # Check for model text (reasoning or plan)
        text_snippets = []
        for part in parts:
            if getattr(part, "text", None):
                text_snippets.append(part.text)

        full_text = "\n".join(text_snippets).strip()

        # If model generated text and we haven't yielded a plan yet, treat as plan
        if full_text and not has_planned:
            has_planned = True
            yield {
                "type": "plan",
                "step": step,
                "content": full_text
            }
        elif full_text and has_planned:
            # Additional thought/reasoning
            yield {
                "type": "thought",
                "step": step,
                "content": full_text
            }

        # Extract function calls from parts
        function_calls = []
        for part in parts:
            if getattr(part, "function_call", None):
                function_calls.append(part.function_call)

        # Append model's turn to conversation history
        contents.append(model_content)

        if not function_calls:
            # If the model didn't call any function, check if it answered directly in text
            if full_text:
                yield {
                    "type": "final",
                    "step": step,
                    "summary": full_text,
                    "key_findings": ["Direct answer provided without additional code execution."],
                    "confidence": "medium",
                    "charts": collected_charts,
                }
                return
            else:
                yield {
                    "type": "error",
                    "step": step,
                    "content": "Model did not invoke a tool and did not provide text output."
                }
                return

        # Process the function calls (one by one or first primary call)
        function_responses = []

        for fc in function_calls:
            fn_name = fc.name
            fn_args = fc.args or {}

            if fn_name == "final_answer":
                summary = fn_args.get("summary", "")
                findings = fn_args.get("key_findings", [])
                confidence = fn_args.get("confidence", "medium")

                yield {
                    "type": "final",
                    "step": step,
                    "summary": summary,
                    "key_findings": findings if isinstance(findings, list) else [str(findings)],
                    "confidence": confidence,
                    "charts": collected_charts,
                }
                return

            elif fn_name == "ask_user":
                clarification_q = fn_args.get("question", "Could you please clarify your question?")
                yield {
                    "type": "ask_user",
                    "step": step,
                    "question": clarification_q,
                }
                return

            elif fn_name == "run_python":
                code = fn_args.get("code", "")
                yield {
                    "type": "code",
                    "step": step,
                    "content": code,
                }

                # Run in sandbox
                exec_result = run_code_sandbox(df, code)

                if exec_result.get("fig") is not None:
                    fig = exec_result["fig"]
                    collected_charts.append(fig)
                    yield {
                        "type": "chart",
                        "step": step,
                        "figure": fig,
                    }

                if exec_result["success"]:
                    consecutive_errors = 0
                    out_content = exec_result["output_str"]
                    yield {
                        "type": "output",
                        "step": step,
                        "content": out_content,
                    }
                    tool_output = {"output": out_content}
                else:
                    consecutive_errors += 1
                    err_content = exec_result["output_str"]
                    yield {
                        "type": "error",
                        "step": step,
                        "content": err_content,
                        "retry_count": consecutive_errors,
                    }
                    tool_output = {"error": err_content}

                    if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                        yield {
                            "type": "final",
                            "step": step,
                            "summary": (
                                "Analysis ended early due to 3 consecutive execution errors. "
                                "Some patterns were inspected, but conclusive validation could not be completed."
                            ),
                            "key_findings": [
                                f"Encountered consecutive errors in code execution: {exec_result.get('error', 'Execution error')}",
                                "Unable to complete all verification steps within sandbox limits."
                            ],
                            "confidence": "low",
                            "charts": collected_charts,
                        }
                        return

                function_responses.append(
                    types.Part.from_function_response(
                        name="run_python",
                        response=tool_output
                    )
                )

        # Feed back function response parts to the model
        contents.append(
            types.Content(
                role="tool",
                parts=function_responses
            )
        )

    # Max steps reached without final answer
    yield {
        "type": "final",
        "step": MAX_STEPS,
        "summary": (
            f"Maximum analysis step limit ({MAX_STEPS} steps) was reached. "
            "Summary is based on the verified intermediate results computed so far."
        ),
        "key_findings": [
            f"Analysis stopped after reaching the maximum budget of {MAX_STEPS} steps.",
            "Please review the step timeline above for intermediate calculation outputs."
        ],
        "confidence": "medium" if collected_charts or consecutive_errors == 0 else "low",
        "charts": collected_charts,
    }
