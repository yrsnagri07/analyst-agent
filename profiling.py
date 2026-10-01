"""Dataframe profiling, validation, and question suggestion utilities."""

import csv
import io
from typing import Any
import pandas as pd
import numpy as np

MAX_FILE_SIZE_BYTES = 20 * 1024 * 1024  # 20 MB limit


class FileTooLargeError(Exception):
    """Raised when uploaded file exceeds max size limit."""
    pass


class CSVLoadError(Exception):
    """Raised when CSV loading fails."""
    pass


def load_csv(file_buffer: Any, file_name: str = "uploaded.csv") -> pd.DataFrame:
    """Safely load a CSV file with automatic delimiter and encoding detection.

    Args:
        file_buffer: File-like object (BytesIO or StringIO) or file path.
        file_name: Original file name.

    Returns:
        Loaded pandas DataFrame.

    Raises:
        FileTooLargeError: If file exceeds 20 MB.
        CSVLoadError: If file cannot be parsed.
    """
    # Check size if bytes-like
    if hasattr(file_buffer, "size") and file_buffer.size > MAX_FILE_SIZE_BYTES:
        raise FileTooLargeError(
            f"File '{file_name}' exceeds the maximum allowed size of 20 MB (size: {file_buffer.size / (1024*1024):.1f} MB)."
        )

    raw_bytes: bytes
    if hasattr(file_buffer, "read"):
        raw_bytes = file_buffer.read()
        if hasattr(file_buffer, "seek"):
            file_buffer.seek(0)
    elif isinstance(file_buffer, str):
        with open(file_buffer, "rb") as f:
            raw_bytes = f.read()
    else:
        raise CSVLoadError("Invalid file buffer or path provided.")

    if len(raw_bytes) > MAX_FILE_SIZE_BYTES:
        raise FileTooLargeError(
            f"File exceeds maximum allowed size of 20 MB ({len(raw_bytes) / (1024*1024):.1f} MB)."
        )

    # Try common encodings
    encodings = ["utf-8", "utf-8-sig", "latin-1", "cp1252"]
    decoded_text: str | None = None
    used_encoding: str = "utf-8"

    for enc in encodings:
        try:
            decoded_text = raw_bytes.decode(enc)
            used_encoding = enc
            break
        except UnicodeDecodeError:
            continue

    if decoded_text is None:
        raise CSVLoadError("Unable to decode CSV file. Please ensure it is saved in UTF-8 or Latin-1 format.")

    # Detect delimiter using sample
    sample = decoded_text[:8192]
    delimiter = ","
    try:
        sniffer = csv.Sniffer()
        dialect = sniffer.sniff(sample, delimiters=[",", ";", "\t", "|"])
        delimiter = dialect.delimiter
    except Exception:
        # Fallback heuristic
        counts = {d: sample.count(d) for d in [",", ";", "\t", "|"]}
        delimiter = max(counts, key=counts.get) if counts else ","

    try:
        df = pd.read_csv(io.StringIO(decoded_text), sep=delimiter)
        if df.empty:
            raise CSVLoadError("The uploaded CSV file is empty.")
        return df
    except Exception as e:
        raise CSVLoadError(f"Failed to parse CSV data: {str(e)}") from e


def profile_dataframe(df: pd.DataFrame) -> dict[str, Any]:
    """Auto-profile a dataframe with shape, dtypes, null counts, stats, and sample rows.

    Args:
        df: Pandas DataFrame to profile.

    Returns:
        Structured dictionary containing dataset profile details.
    """
    rows, cols = df.shape

    columns_info = []
    for col in df.columns:
        null_count = int(df[col].isna().sum())
        null_pct = round((null_count / rows) * 100, 2) if rows > 0 else 0.0
        unique_count = int(df[col].nunique(dropna=True))
        dtype_str = str(df[col].dtype)

        # Get up to 3 non-null sample values
        samples = [str(v) for v in df[col].dropna().head(3).tolist()]

        columns_info.append({
            "name": str(col),
            "dtype": dtype_str,
            "null_count": null_count,
            "null_pct": null_pct,
            "unique_count": unique_count,
            "samples": samples,
        })

    # Numeric summary
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_summary = {}
    if numeric_cols:
        desc = df[numeric_cols].describe().round(2).to_dict()
        numeric_summary = desc

    # Categorical summary
    cat_cols = df.select_dtypes(include=["object", "category", "string"]).columns.tolist()
    categorical_summary = {}
    for col in cat_cols[:5]:  # limit to top 5 categorical columns
        top_counts = df[col].value_counts().head(5).to_dict()
        categorical_summary[col] = top_counts

    # Sample rows
    sample_rows = df.head(5).to_dict(orient="records")

    # Date columns detection
    date_columns = []
    for col in df.columns:
        if "date" in str(col).lower() or "time" in str(col).lower():
            try:
                dt_series = pd.to_datetime(df[col], errors="coerce").dropna()
                if not dt_series.empty:
                    date_columns.append({
                        "name": str(col),
                        "min": str(dt_series.min().date()),
                        "max": str(dt_series.max().date()),
                    })
            except Exception:
                pass

    return {
        "rows": rows,
        "columns": cols,
        "column_names": [str(c) for c in df.columns],
        "columns_info": columns_info,
        "numeric_summary": numeric_summary,
        "categorical_summary": categorical_summary,
        "sample_rows": sample_rows,
        "date_columns": date_columns,
        "memory_mb": round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2),
    }


def format_profile_for_llm(profile: dict[str, Any]) -> str:
    """Format the dataframe profile into a compact, readable context string for Gemini.

    Args:
        profile: Dictionary returned by profile_dataframe.

    Returns:
        Markdown string ready for prompt injection.
    """
    lines = [
        f"- Dataset Dimensions: {profile['rows']} rows x {profile['columns']} columns (Memory: {profile.get('memory_mb', 0)} MB)",
        "\nColumns & Data Types:",
    ]
    for col in profile["columns_info"]:
        sample_str = ", ".join(f"'{s}'" for s in col["samples"])
        lines.append(
            f"  - `{col['name']}` ({col['dtype']}): {col['unique_count']} unique, "
            f"{col['null_count']} nulls ({col['null_pct']}%), samples: [{sample_str}]"
        )

    if profile.get("date_columns"):
        lines.append("\nDate Ranges:")
        for dc in profile["date_columns"]:
            lines.append(f"  - `{dc['name']}`: {dc['min']} to {dc['max']}")

    if profile.get("numeric_summary"):
        lines.append("\nNumeric Column Summaries (describe):")
        for col, stats in profile["numeric_summary"].items():
            stat_str = f"mean={stats.get('mean')}, min={stats.get('min')}, max={stats.get('max')}, std={stats.get('std')}"
            lines.append(f"  - `{col}`: {stat_str}")

    if profile.get("categorical_summary"):
        lines.append("\nCategorical Top Values:")
        for col, counts in profile["categorical_summary"].items():
            count_str = ", ".join(f"'{k}': {v}" for k, v in counts.items())
            lines.append(f"  - `{col}`: [{count_str}]")

    lines.append("\nFirst 5 Sample Rows:")
    sample_df = pd.DataFrame(profile["sample_rows"])
    lines.append(sample_df.to_markdown(index=False))

    return "\n".join(lines)


def generate_suggested_questions(df: pd.DataFrame, profile: dict[str, Any]) -> list[str]:
    """Generate 3 smart, context-aware suggested questions based on dataframe schema and content.

    Args:
        df: Pandas DataFrame.
        profile: Dictionary returned by profile_dataframe.

    Returns:
        List of 3 suggested question strings.
    """
    cols_lower = {str(c).lower(): c for c in df.columns}
    suggestions = []

    # Check for revenue/sales and date
    has_date = any("date" in c or "time" in c for c in cols_lower)
    rev_col = next((cols_lower[c] for c in cols_lower if any(term in c for term in ["revenue", "sales", "amount", "total"])), None)
    cat_col = next((cols_lower[c] for c in cols_lower if any(term in c for term in ["category", "product", "item", "type"])), None)
    region_col = next((cols_lower[c] for c in cols_lower if any(term in c for term in ["region", "country", "city", "location", "segment"])), None)
    units_col = next((cols_lower[c] for c in cols_lower if any(term in c for term in ["unit", "quantity", "volume", "count"])), None)
    disc_col = next((cols_lower[c] for c in cols_lower if "discount" in c or "margin" in c), None)

    # 1. Anomaly / Trend question
    if has_date and rev_col:
        suggestions.append(f"Why did {rev_col} drop in March?")
    elif rev_col and cat_col:
        suggestions.append(f"Which {cat_col} generated the highest {rev_col}?")

    # 2. Breakdown / Comparison question
    if rev_col and region_col and cat_col:
        suggestions.append(f"How does {rev_col} break down by {region_col} and {cat_col}?")
    elif rev_col and region_col:
        suggestions.append(f"Which {region_col} has the highest and lowest total {rev_col}?")
    elif cat_col and units_col:
        suggestions.append(f"Which {cat_col} had the most {units_col} sold?")

    # 3. Correlation / Relationship question
    if disc_col and rev_col:
        suggestions.append(f"What is the relationship between {disc_col} and {rev_col}?")
    elif units_col and rev_col:
        suggestions.append(f"What is the average price per unit across different categories?")
    elif profile.get("numeric_summary") and len(profile["numeric_summary"]) >= 2:
        num_keys = list(profile["numeric_summary"].keys())
        suggestions.append(f"What is the correlation between {num_keys[0]} and {num_keys[1]}?")

    # Fallback to general questions if needed
    fallbacks = [
        "What are the key trends and patterns in this dataset?",
        "Provide a comprehensive summary of total metrics.",
        "Are there any notable anomalies or outliers?",
    ]
    for fb in fallbacks:
        if len(suggestions) < 3 and fb not in suggestions:
            suggestions.append(fb)

    return suggestions[:3]
