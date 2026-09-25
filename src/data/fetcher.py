"""
src/data/fetcher.py
--------------------
Live benchmark data fetching from the HuggingFace Open LLM Leaderboard.

All fetch functions use Streamlit's @st.cache_data with a configurable TTL
so the UI remains snappy while data stays fresh.

NOTE: Actual network calls are stubbed here — full implementation is the
next milestone. Each function returns a typed DataFrame with the schema
that downstream modules expect.
"""

from __future__ import annotations

import logging
from datetime import datetime

import pandas as pd
import streamlit as st

from src.utils.constants import (
    BENCHMARK_COLUMNS,
    CACHE_TTL_SECONDS,
    HF_LEADERBOARD_REPO,
)

logger = logging.getLogger(__name__)

# Expected output schema for all fetcher functions
LEADERBOARD_SCHEMA: dict[str, str] = {
    "model_name":     "object",   # full HF model ID
    "model_type":     "object",   # e.g. "chat", "base", "instruct"
    "params_b":       "float64",  # parameter count in billions
    "licence":        "object",
    "ifeval":         "float64",
    "bbh":            "float64",
    "math_lvl5":      "float64",
    "gpqa":           "float64",
    "musr":           "float64",
    "mmlu_pro":       "float64",
    "average":        "float64",  # official HF average
    "flagged":        "bool",
    "merged":         "bool",
    "fetched_at":     "object",   # ISO timestamp string
}


def _empty_leaderboard() -> pd.DataFrame:
    """Return an empty DataFrame matching LEADERBOARD_SCHEMA."""
    return pd.DataFrame({col: pd.Series(dtype=dtype) for col, dtype in LEADERBOARD_SCHEMA.items()})


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def fetch_leaderboard(
    include_flagged: bool = False,
    include_merged: bool = False,
) -> pd.DataFrame:
    """
    Fetch the Open LLM Leaderboard dataset from HuggingFace Hub.

    Parameters
    ----------
    include_flagged : If True, include models marked as flagged
    include_merged  : If True, include models that are merges of other models

    Returns
    -------
    pd.DataFrame with columns matching LEADERBOARD_SCHEMA.

    Raises
    ------
    RuntimeError if the fetch fails and no cached data is available.

    Notes
    -----
    STUB: Returns a realistic sample DataFrame until the HF dataset
    integration is implemented in the next milestone.
    """
    import requests

    logger.info("fetch_leaderboard called at %s", datetime.utcnow().isoformat())

    rows = []
    try:
        for offset in [0, 100, 200, 300, 400]:
            url = f"https://datasets-server.huggingface.co/rows?dataset=open-llm-leaderboard%2Fcontents&config=default&split=train&offset={offset}&length=100"
            r = requests.get(url, timeout=10)
            r.raise_for_status()
            page_rows = r.json().get("rows", [])
            rows.extend(page_rows)
            if len(page_rows) < 100:
                break
    except Exception as e:
        logger.error(f"Failed to fetch from HuggingFace Datasets API at offset {offset}: {e}")
        if not rows:
            return _empty_leaderboard()

    processed_data = []
    now = datetime.utcnow().isoformat()
    for item in rows:
        row = item.get("row", {})
        
        # Determine model_type
        type_str = str(row.get("Type", "")).lower()
        if "chat" in type_str or "instruct" in type_str:
            model_type = "chat"
        elif "base" in type_str or "pretrained" in type_str:
            model_type = "base"
        else:
            model_type = "unknown"
            
        processed_data.append({
            "model_name": row.get("fullname", ""),
            "model_type": model_type,
            "params_b": float(row.get("#Params (B)", 0)) if pd.notna(row.get("#Params (B)")) else float("nan"),
            "licence": row.get("Hub License", "unknown"),
            "ifeval": float(row.get("IFEval", float("nan"))),
            "bbh": float(row.get("BBH", float("nan"))),
            "math_lvl5": float(row.get("MATH Lvl 5", float("nan"))),
            "gpqa": float(row.get("GPQA", float("nan"))),
            "musr": float(row.get("MUSR", float("nan"))),
            "mmlu_pro": float(row.get("MMLU-PRO", float("nan"))),
            "average": float(row.get("Average ⬆️", float("nan"))),
            "flagged": bool(row.get("Flagged", False)),
            "merged": bool(row.get("Merged", False)),
            "fetched_at": now,
        })

    df = pd.DataFrame(processed_data)
    
    if df.empty:
        return _empty_leaderboard()

    # Enforce schema types strictly
    for col, dtype in LEADERBOARD_SCHEMA.items():
        if col in df.columns:
            df[col] = df[col].astype(dtype, errors='ignore')
        else:
            df[col] = pd.Series(dtype=dtype)

    if not include_flagged:
        df = df[~df["flagged"]]
    if not include_merged:
        df = df[~df["merged"]]

    return df.reset_index(drop=True)


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def fetch_model_details(model_name: str) -> dict:
    """
    Fetch detailed metadata for a single model from the HF Hub.

    Parameters
    ----------
    model_name : Full HuggingFace model ID (e.g. "mistralai/Mistral-7B-v0.1")

    Returns
    -------
    Dictionary with model metadata fields.

    Notes
    -----
    STUB: Returns a placeholder dict until `huggingface_hub` integration.
    """
    logger.info("fetch_model_details called for %s", model_name)
    return {
        "model_name": model_name,
        "description": "Detailed model description — fetched from HF Hub (stub).",
        "downloads_last_month": None,
        "likes": None,
        "tags": [],
        "pipeline_tag": None,
        "created_at": None,
    }
