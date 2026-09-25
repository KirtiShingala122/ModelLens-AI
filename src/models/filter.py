"""
src/models/filter.py
---------------------
Model filtering layer for ModelLens AI.

All functions are **pure** — they accept a DataFrame and keyword arguments,
perform no I/O or Streamlit calls, and return a new DataFrame.  The caller
(UI or recommender) is responsible for providing data and interpreting results.

Filtering pipeline
------------------
1. filter_by_use_case        — keep only models plausibly suited to a task
2. filter_by_size            — parameter-count range (billions)
3. filter_by_language        — required language support
4. filter_by_benchmark_availability — minimum number of non-null benchmark cols
5. filter_by_benchmark_score — per-column or average floor score
6. filter_by_licence         — allowed licence strings / commercial-only flag
7. filter_by_model_type      — model architecture category (chat/base/instruct)
8. filter_by_name_search     — substring search on model ID

Each function:
  • Returns a fresh DataFrame (never mutates the input).
  • Handles missing columns gracefully — if the required column is absent the
    filter is a no-op so the caller still gets data rather than an error.
  • Handles NaN / None values safely using pandas NA-aware comparisons.
  • Attaches a ``filter_log`` attribute to the returned DataFrame so the
    orchestrator can report what was dropped at each stage.

Orchestrator
------------
``apply_filters`` runs the full pipeline and returns a ``FilterResult``
named-tuple carrying the final DataFrame, the total rows dropped, and a
per-stage audit log.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd

from src.utils.constants import (
    BENCHMARK_COLUMNS,
    COMMERCIAL_OK_LICENCES,
    LANGUAGE_KEYWORDS,
    USE_CASE_REQUIRED_BENCHMARKS,
    USE_CASE_WEIGHT_PRESETS,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------

@dataclass
class FilterResult:
    """
    Container returned by ``apply_filters``.

    Attributes
    ----------
    df          : Validated, filtered DataFrame (may be empty).
    rows_in     : Number of rows before any filtering.
    rows_out    : Number of rows after all filters.
    dropped     : Total rows removed (rows_in - rows_out).
    stage_log   : Ordered list of per-stage dicts with keys
                  ``stage``, ``rows_before``, ``rows_after``, ``dropped``,
                  and ``skipped`` (True when the filter was a no-op).
    warnings    : List of human-readable warning strings (missing columns,
                  invalid parameters, etc.).
    """
    df:        pd.DataFrame
    rows_in:   int
    rows_out:  int
    dropped:   int
    stage_log: list[dict] = field(default_factory=list)
    warnings:  list[str]  = field(default_factory=list)

    def __repr__(self) -> str:                              # pragma: no cover
        return (
            f"FilterResult(rows_in={self.rows_in}, rows_out={self.rows_out}, "
            f"dropped={self.dropped}, warnings={len(self.warnings)})"
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _coerce_numeric(series: pd.Series) -> pd.Series:
    """Coerce a Series to float, replacing non-numeric values with NaN."""
    return pd.to_numeric(series, errors="coerce")


def _col_present(df: pd.DataFrame, col: str, warnings: list[str]) -> bool:
    """Return True if col exists; append a warning and return False otherwise."""
    if col in df.columns:
        return True
    warnings.append(f"Column '{col}' not found — filter skipped.")
    return False


def _safe_str_lower(series: pd.Series) -> pd.Series:
    """Lowercase a string Series, converting NaN to empty string."""
    return series.fillna("").astype(str).str.lower().str.strip()


# ---------------------------------------------------------------------------
# 1. Filter by use-case / task
# ---------------------------------------------------------------------------

def filter_by_use_case(
    df: pd.DataFrame,
    use_case: str | None,
    *,
    model_type_col: str = "model_type",
    warnings: list[str] | None = None,
) -> pd.DataFrame:
    """
    Keep models whose ``model_type`` is appropriate for the requested task.

    The mapping is intentionally permissive — a model passes if its type
    matches *any* of the allowed types for that task, or if the ``model_type``
    column is absent / the value is NaN (we never exclude unknowns silently).

    Parameters
    ----------
    df            : Input DataFrame (not mutated).
    use_case      : One of the keys in USE_CASE_WEIGHT_PRESETS, or None / ""
                    to skip this filter.
    model_type_col: Column name holding model type strings.
    warnings      : Mutable list to collect diagnostic messages.

    Returns
    -------
    Filtered DataFrame.
    """
    warnings = warnings if warnings is not None else []
    if not use_case or use_case not in USE_CASE_WEIGHT_PRESETS:
        return df.copy()

    # Task → acceptable model_type values
    TASK_MODEL_TYPES: dict[str, list[str]] = {
        "General Purpose":      ["chat", "instruct", "base"],
        "Coding & Math":        ["chat", "instruct", "base"],
        "Instruction Following": ["chat", "instruct"],
        "Factual QA / RAG":     ["chat", "instruct"],
        "Creative Writing":     ["chat", "instruct"],
        "Reasoning":            ["chat", "instruct", "base"],
    }

    allowed_types = TASK_MODEL_TYPES.get(use_case)
    if not allowed_types or model_type_col not in df.columns:
        # If we can't resolve types or the column is missing, pass everything
        return df.copy()

    lowered = _safe_str_lower(df[model_type_col])
    # A row passes if its type matches OR if its type is unknown (empty / NaN)
    mask = lowered.isin(allowed_types) | (lowered == "")
    return df[mask].reset_index(drop=True)


# ---------------------------------------------------------------------------
# 2. Filter by model size
# ---------------------------------------------------------------------------

def filter_by_size(
    df: pd.DataFrame,
    min_b: float = 0.0,
    max_b: float = 10_000.0,
    *,
    size_col: str = "params_b",
    include_unknown_size: bool = True,
    warnings: list[str] | None = None,
) -> pd.DataFrame:
    """
    Keep rows where parameter count (billions) is within [min_b, max_b].

    Parameters
    ----------
    df                  : Input DataFrame (not mutated).
    min_b               : Minimum size in billions (inclusive). Default 0.
    max_b               : Maximum size in billions (inclusive). Default 10 000.
    size_col            : Column name for parameter count.
    include_unknown_size: If True, rows where size is NaN/None are retained
                          (we never silently drop models we can't size).
    warnings            : Mutable diagnostic list.

    Returns
    -------
    Filtered DataFrame.
    """
    warnings = warnings if warnings is not None else []

    # Validate parameters
    try:
        min_b = float(min_b)
        max_b = float(max_b)
    except (TypeError, ValueError):
        warnings.append(f"filter_by_size: invalid min_b or max_b — filter skipped.")
        return df.copy()

    if min_b < 0:
        warnings.append(f"filter_by_size: min_b={min_b} < 0; clamped to 0.")
        min_b = 0.0
    if max_b < min_b:
        warnings.append(
            f"filter_by_size: max_b={max_b} < min_b={min_b}; swapping."
        )
        min_b, max_b = max_b, min_b

    if size_col not in df.columns:
        warnings.append(f"Column '{size_col}' not found — size filter skipped.")
        return df.copy()

    sizes = _coerce_numeric(df[size_col])
    known_mask  = sizes.notna()
    in_range    = sizes.between(min_b, max_b, inclusive="both")
    unknown_mask = ~known_mask if include_unknown_size else pd.Series(False, index=df.index)

    mask = (known_mask & in_range) | unknown_mask
    return df[mask].reset_index(drop=True)


# ---------------------------------------------------------------------------
# 3. Filter by language
# ---------------------------------------------------------------------------

def filter_by_language(
    df: pd.DataFrame,
    required_languages: Sequence[str] | None,
    *,
    language_col: str = "languages",
    name_col: str = "model_name",
    warnings: list[str] | None = None,
) -> pd.DataFrame:
    """
    Keep models that support *all* of the requested languages.

    Strategy (in priority order):
    1. If the DataFrame has a ``language_col`` column, check it directly.
       The column may contain a comma-separated string, a list, or a
       JSON-stringified list — all are handled.
    2. Otherwise fall back to keyword matching on the model name/ID using
       ``LANGUAGE_KEYWORDS`` from constants.
    3. English (``"en"``) is always assumed supported.
    4. Models with no detectable language information pass through unless
       a *non-English* language was explicitly requested AND the model name
       gives no matching signal.

    Parameters
    ----------
    df                 : Input DataFrame (not mutated).
    required_languages : BCP-47 codes (e.g. ["en", "zh"]). None / [] → no-op.
    language_col       : Optional column with known language lists.
    name_col           : Fallback column for keyword heuristic.
    warnings           : Mutable diagnostic list.

    Returns
    -------
    Filtered DataFrame.
    """
    warnings = warnings if warnings is not None else []

    if not required_languages:
        return df.copy()

    lang_codes = [c.lower().strip() for c in required_languages]

    # English-only → pass everything (all models assumed to support English)
    if set(lang_codes) <= {"en"}:
        return df.copy()

    non_english = [c for c in lang_codes if c != "en"]

    def _model_supports(row: pd.Series) -> bool:
        """Return True if the row's model supports all non-english codes."""
        # ── Strategy 1: dedicated languages column ──────────────────────────
        if language_col in row.index:
            raw = row[language_col]
            if pd.notna(raw) and raw != "":
                # Normalise to a flat list of lower-case codes
                if isinstance(raw, list):
                    known = [str(x).lower().strip() for x in raw]
                else:
                    # Try comma / pipe delimited strings and JSON arrays
                    raw_s = str(raw).strip().strip("[]").replace("'", "").replace('"', "")
                    known = [x.strip().lower() for x in re.split(r"[,|;]", raw_s) if x.strip()]

                # "multilingual" entry satisfies any language code
                if "multilingual" in known:
                    return True
                # Check every non-English code is in the known list
                return all(
                    any(code == k or k.startswith(code) for k in known)
                    for code in non_english
                )

        # ── Strategy 2: keyword heuristic on model name ─────────────────────
        if name_col in row.index:
            name_lower = str(row[name_col]).lower()
            for code in non_english:
                keywords = LANGUAGE_KEYWORDS.get(code, [])
                if not keywords:
                    # No keywords defined → assume supported (be permissive)
                    continue
                if not any(kw in name_lower for kw in keywords):
                    return False
            return True

        # No information → pass through (don't silently exclude)
        return True

    mask = df.apply(_model_supports, axis=1)
    if not any(mask):
        warnings.append(
            f"filter_by_language: no models matched languages {required_languages}. "
            "Returning all rows to avoid empty result."
        )
        return df.copy()

    return df[mask].reset_index(drop=True)


# ---------------------------------------------------------------------------
# 4. Filter by benchmark availability
# ---------------------------------------------------------------------------

def filter_by_benchmark_availability(
    df: pd.DataFrame,
    required_benchmarks: Sequence[str] | None = None,
    use_case: str | None = None,
    min_available: int | None = None,
    *,
    warnings: list[str] | None = None,
) -> pd.DataFrame:
    """
    Require that certain benchmark score columns are non-null (evaluated).

    Three modes (mutually exclusive, applied in priority order):
    A. ``required_benchmarks`` — an explicit list of column names that must
       have non-NaN values.
    B. ``use_case`` — automatically derive the required columns from
       ``USE_CASE_REQUIRED_BENCHMARKS`` in constants.
    C. ``min_available`` — require at least N of the standard BENCHMARK_COLUMNS
       to be non-NaN.

    If none of the three are provided, the filter is a no-op.

    Parameters
    ----------
    df                  : Input DataFrame (not mutated).
    required_benchmarks : Explicit list of column names that must be non-null.
    use_case            : Task key to look up required benchmarks automatically.
    min_available       : Minimum number of any standard benchmarks that must
                          have a value. Ignored if required_benchmarks or
                          use_case is set.
    warnings            : Mutable diagnostic list.

    Returns
    -------
    Filtered DataFrame.
    """
    warnings = warnings if warnings is not None else []

    # Resolve which columns must be non-null
    if required_benchmarks:
        must_have = list(required_benchmarks)
    elif use_case and use_case in USE_CASE_REQUIRED_BENCHMARKS:
        must_have = USE_CASE_REQUIRED_BENCHMARKS[use_case]
    elif min_available is not None:
        # Mode C: count available standard benchmarks
        avail_cols = [c for c in BENCHMARK_COLUMNS if c in df.columns]
        if not avail_cols:
            warnings.append("filter_by_benchmark_availability: no benchmark columns found — skipped.")
            return df.copy()
        available_counts = df[avail_cols].notna().sum(axis=1)
        mask = available_counts >= int(min_available)
        if not any(mask):
            warnings.append(
                f"filter_by_benchmark_availability: min_available={min_available} "
                "matched 0 rows — returning all rows."
            )
            return df.copy()
        return df[mask].reset_index(drop=True)
    else:
        return df.copy()  # no-op

    # Validate requested columns exist in the DataFrame
    missing_cols = [c for c in must_have if c not in df.columns]
    if missing_cols:
        warnings.append(
            f"filter_by_benchmark_availability: columns {missing_cols} not in DataFrame — "
            "they will be ignored in the availability check."
        )
        must_have = [c for c in must_have if c in df.columns]

    if not must_have:
        return df.copy()

    # A row passes if ALL required columns are non-NaN
    mask = df[must_have].notna().all(axis=1)
    if not any(mask):
        warnings.append(
            f"filter_by_benchmark_availability: required benchmarks {must_have} "
            "matched 0 rows — returning all rows to avoid empty result."
        )
        return df.copy()

    return df[mask].reset_index(drop=True)


# ---------------------------------------------------------------------------
# 5. Filter by benchmark score
# ---------------------------------------------------------------------------

def filter_by_benchmark_score(
    df: pd.DataFrame,
    *,
    min_average_score: float | None = None,
    per_benchmark_mins: dict[str, float] | None = None,
    average_col: str = "average",
    warnings: list[str] | None = None,
) -> pd.DataFrame:
    """
    Remove rows that do not meet minimum benchmark score thresholds.

    Parameters
    ----------
    df                 : Input DataFrame (not mutated).
    min_average_score  : If set, rows where ``average_col`` < this are dropped.
                         Rows with a NaN average are **retained** (missing
                         data ≠ bad data).
    per_benchmark_mins : Optional dict mapping individual benchmark column
                         names to their minimum required scores, e.g.
                         {"mmlu": 50.0, "gsm8k": 30.0}.  For each specified
                         column: if the value is NaN the row is retained; if
                         the value is present and below the threshold the row
                         is dropped.
    average_col        : Name of the aggregated average column.
    warnings           : Mutable diagnostic list.

    Returns
    -------
    Filtered DataFrame.
    """
    warnings = warnings if warnings is not None else []
    result = df.copy()

    # ── Average score threshold ──────────────────────────────────────────────
    if min_average_score is not None:
        try:
            floor = float(min_average_score)
        except (TypeError, ValueError):
            warnings.append(
                f"filter_by_benchmark_score: invalid min_average_score "
                f"'{min_average_score}' — skipped."
            )
            floor = None

        if floor is not None and floor > 0.0:
            if average_col not in result.columns:
                warnings.append(
                    f"filter_by_benchmark_score: column '{average_col}' not found "
                    "— average score filter skipped."
                )
            else:
                avg = _coerce_numeric(result[average_col])
                # Keep rows where value >= floor OR value is NaN
                mask = avg.isna() | (avg >= floor)
                result = result[mask].reset_index(drop=True)

    # ── Per-benchmark thresholds ─────────────────────────────────────────────
    if per_benchmark_mins:
        for col, threshold in per_benchmark_mins.items():
            try:
                thr = float(threshold)
            except (TypeError, ValueError):
                warnings.append(
                    f"filter_by_benchmark_score: invalid threshold for '{col}': "
                    f"'{threshold}' — skipped."
                )
                continue

            if thr <= 0.0:
                continue  # No-op: threshold of 0 never filters anything

            if col not in result.columns:
                warnings.append(
                    f"filter_by_benchmark_score: column '{col}' not found — "
                    "per-benchmark threshold skipped."
                )
                continue

            scores = _coerce_numeric(result[col])
            # Retain NaN (unknown) and anything >= threshold
            mask = scores.isna() | (scores >= thr)
            result = result[mask].reset_index(drop=True)

    return result


# ---------------------------------------------------------------------------
# 6. Filter by licence  (extended from original)
# ---------------------------------------------------------------------------

def filter_by_licence(
    df: pd.DataFrame,
    licences: Sequence[str] | None = None,
    *,
    commercial_only: bool = False,
    licence_col: str = "licence",
    include_unknown_licence: bool = True,
    warnings: list[str] | None = None,
) -> pd.DataFrame:
    """
    Keep rows whose licence matches the allowed set.

    Parameters
    ----------
    df                     : Input DataFrame (not mutated).
    licences               : Explicit allowlist.  If None and ``commercial_only``
                             is False the filter is a no-op.
    commercial_only        : If True, override ``licences`` with
                             ``COMMERCIAL_OK_LICENCES``.
    licence_col            : Column name for licence strings.
    include_unknown_licence: If True, rows with NaN / empty licence are always
                             retained (safe default — don't exclude what we
                             don't know).
    warnings               : Mutable diagnostic list.

    Returns
    -------
    Filtered DataFrame.
    """
    warnings = warnings if warnings is not None else []

    if commercial_only:
        allowed = [l.lower() for l in COMMERCIAL_OK_LICENCES]
    elif licences:
        allowed = [l.lower() for l in licences]
    else:
        return df.copy()  # no-op

    if licence_col not in df.columns:
        warnings.append(f"Column '{licence_col}' not found — licence filter skipped.")
        return df.copy()

    lowered = _safe_str_lower(df[licence_col])
    known_mask   = lowered != ""
    allowed_mask = lowered.isin(allowed)
    unknown_mask = ~known_mask if include_unknown_licence else pd.Series(False, index=df.index)

    mask = (known_mask & allowed_mask) | unknown_mask
    return df[mask].reset_index(drop=True)


# ---------------------------------------------------------------------------
# 7. Filter by model type
# ---------------------------------------------------------------------------

def filter_by_model_type(
    df: pd.DataFrame,
    model_types: Sequence[str] | None = None,
    *,
    model_type_col: str = "model_type",
    include_unknown_type: bool = True,
    warnings: list[str] | None = None,
) -> pd.DataFrame:
    """
    Keep rows whose model type is in the allowed set.

    Parameters
    ----------
    df                   : Input DataFrame (not mutated).
    model_types          : List of allowed type strings (case-insensitive).
                           None / [] → no-op.
    model_type_col       : Column name.
    include_unknown_type : If True, NaN / blank types are retained.
    warnings             : Mutable diagnostic list.

    Returns
    -------
    Filtered DataFrame.
    """
    warnings = warnings if warnings is not None else []

    if not model_types:
        return df.copy()

    allowed_lower = {t.lower().strip() for t in model_types}

    if model_type_col not in df.columns:
        warnings.append(f"Column '{model_type_col}' not found — model type filter skipped.")
        return df.copy()

    lowered = _safe_str_lower(df[model_type_col])
    known_mask   = lowered != ""
    allowed_mask = lowered.isin(allowed_lower)
    unknown_mask = ~known_mask if include_unknown_type else pd.Series(False, index=df.index)

    mask = (known_mask & allowed_mask) | unknown_mask
    return df[mask].reset_index(drop=True)


# ---------------------------------------------------------------------------
# 8. Filter by name search
# ---------------------------------------------------------------------------

def filter_by_name_search(
    df: pd.DataFrame,
    query: str,
    *,
    name_col: str = "model_name",
    warnings: list[str] | None = None,
) -> pd.DataFrame:
    """
    Case-insensitive substring / regex search on the model ID column.

    Parameters
    ----------
    df       : Input DataFrame (not mutated).
    query    : Search string.  Leading/trailing whitespace is stripped.
               Empty string or None → no-op.
    name_col : Column to search.
    warnings : Mutable diagnostic list.

    Returns
    -------
    Filtered DataFrame.
    """
    warnings = warnings if warnings is not None else []

    query = (query or "").strip()
    if not query:
        return df.copy()

    if name_col not in df.columns:
        warnings.append(f"Column '{name_col}' not found — name search skipped.")
        return df.copy()

    try:
        mask = df[name_col].astype(str).str.contains(query, case=False, na=False, regex=True)
    except re.error:
        # Fallback to literal match if query is not a valid regex
        mask = df[name_col].astype(str).str.contains(
            re.escape(query), case=False, na=False, regex=True
        )
        warnings.append(
            f"filter_by_name_search: '{query}' is not a valid regex — "
            "treating as a literal string."
        )

    return df[mask].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def apply_filters(
    df: pd.DataFrame,
    *,
    # 1. Use-case
    use_case: str | None = None,
    # 2. Size
    min_params_b: float = 0.0,
    max_params_b: float = 10_000.0,
    include_unknown_size: bool = True,
    # 3. Language
    required_languages: Sequence[str] | None = None,
    # 4. Benchmark availability
    required_benchmarks: Sequence[str] | None = None,
    min_benchmarks_available: int | None = None,
    # 5. Benchmark score
    min_average_score: float | None = None,
    per_benchmark_mins: dict[str, float] | None = None,
    # 6. Licence
    licences: Sequence[str] | None = None,
    commercial_only: bool = False,
    include_unknown_licence: bool = True,
    # 7. Model type
    model_types: Sequence[str] | None = None,
    include_unknown_type: bool = True,
    # 8. Name search
    name_query: str = "",
) -> FilterResult:
    """
    Run the full filtering pipeline and return a validated ``FilterResult``.

    Parameters
    ----------
    df : Raw leaderboard DataFrame from ``src.data.fetcher``.

    All other keyword arguments map 1-to-1 to the individual filter functions
    documented above.

    Returns
    -------
    FilterResult with:
    - ``df``        : Filtered candidate models.
    - ``rows_in``   : Input size.
    - ``rows_out``  : Output size.
    - ``dropped``   : Total rows removed.
    - ``stage_log`` : Per-stage audit trail.
    - ``warnings``  : Accumulated diagnostic messages.
    """
    if df is None or not isinstance(df, pd.DataFrame):
        raise TypeError("apply_filters: 'df' must be a pandas DataFrame.")

    warnings: list[str] = []
    stage_log: list[dict] = []
    rows_in = len(df)
    current = df.copy()

    def _record(stage: str, before: int, after: int, skipped: bool = False) -> None:
        stage_log.append(
            {
                "stage":        stage,
                "rows_before":  before,
                "rows_after":   after,
                "dropped":      before - after,
                "skipped":      skipped,
            }
        )
        if before != after:
            logger.debug("[filter] %-35s  %d → %d  (−%d)", stage, before, after, before - after)

    # ── 1. Use-case ──────────────────────────────────────────────────────────
    b = len(current)
    current = filter_by_use_case(current, use_case, warnings=warnings)
    _record("use_case", b, len(current), skipped=not use_case)

    # ── 2. Size ───────────────────────────────────────────────────────────────
    b = len(current)
    current = filter_by_size(
        current, min_params_b, max_params_b,
        include_unknown_size=include_unknown_size, warnings=warnings,
    )
    _record(
        "size",
        b, len(current),
        skipped=(min_params_b <= 0 and max_params_b >= 10_000),
    )

    # ── 3. Language ───────────────────────────────────────────────────────────
    b = len(current)
    current = filter_by_language(current, required_languages, warnings=warnings)
    _record("language", b, len(current), skipped=not required_languages)

    # ── 4. Benchmark availability ─────────────────────────────────────────────
    b = len(current)
    current = filter_by_benchmark_availability(
        current,
        required_benchmarks=required_benchmarks,
        use_case=use_case,
        min_available=min_benchmarks_available,
        warnings=warnings,
    )
    _record(
        "benchmark_availability",
        b, len(current),
        skipped=not (required_benchmarks or use_case or min_benchmarks_available),
    )

    # ── 5. Benchmark score ────────────────────────────────────────────────────
    b = len(current)
    current = filter_by_benchmark_score(
        current,
        min_average_score=min_average_score,
        per_benchmark_mins=per_benchmark_mins,
        warnings=warnings,
    )
    _record(
        "benchmark_score",
        b, len(current),
        skipped=not (min_average_score or per_benchmark_mins),
    )

    # ── 6. Licence ────────────────────────────────────────────────────────────
    b = len(current)
    current = filter_by_licence(
        current,
        licences=licences,
        commercial_only=commercial_only,
        include_unknown_licence=include_unknown_licence,
        warnings=warnings,
    )
    _record(
        "licence",
        b, len(current),
        skipped=not (licences or commercial_only),
    )

    # ── 7. Model type ─────────────────────────────────────────────────────────
    b = len(current)
    current = filter_by_model_type(
        current,
        model_types=model_types,
        include_unknown_type=include_unknown_type,
        warnings=warnings,
    )
    _record("model_type", b, len(current), skipped=not model_types)

    # ── 8. Name search ────────────────────────────────────────────────────────
    b = len(current)
    current = filter_by_name_search(current, name_query, warnings=warnings)
    _record("name_search", b, len(current), skipped=not name_query.strip())

    rows_out = len(current)

    if rows_out == 0:
        warnings.append(
            "apply_filters: all rows were removed by the filter pipeline. "
            "Check your constraints — they may be too restrictive."
        )

    return FilterResult(
        df=current,
        rows_in=rows_in,
        rows_out=rows_out,
        dropped=rows_in - rows_out,
        stage_log=stage_log,
        warnings=warnings,
    )
