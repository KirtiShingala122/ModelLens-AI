"""
src/models/scorer.py
---------------------
Transparent, deterministic, explainable suitability scoring engine.

Scoring model
-------------
Each model receives an overall suitability score in [0, 100] that is the
weighted sum of five independent components:

    Component            Default weight   What it measures
    ─────────────────────────────────────────────────────────────
    benchmark            55%              Normalized task-weighted benchmark
                                          performance across all available
                                          standard benchmark columns.
    use_case             20%              How well the *distribution* of the
                                          model's benchmark strengths aligns
                                          with the use-case's priority weights
                                          (cosine similarity).
    language             10%              Whether the model is confirmed,
                                          probable, or unknown for the
                                          requested languages.
    size_preference       5%              Linear proximity of params_b to the
                                          caller's preferred size range.
    data_completeness    10%              Fraction of standard benchmark columns
                                          that have non-null values.

All weights are configurable via ``SCORING_CONFIG`` in constants.py.
No model is ever hard-coded as the result.

Public API
----------
score_model(row, ...)        -> ScoringResult   (single row)
score_dataframe(df, ...)     -> pd.DataFrame     (adds score columns)
score_for_use_case(df, ...)  -> pd.DataFrame     (ranked, all components)
top_n_models(df, ...)        -> pd.DataFrame     (top-N shortcut)
"""

from __future__ import annotations

import math
import logging
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd

from src.utils.constants import (
    BENCHMARK_COLUMNS,
    BENCHMARK_LABELS,
    BENCHMARK_SCORE_CEILINGS,
    BENCHMARK_SCORE_FLOORS,
    DEFAULT_BENCHMARK_WEIGHTS,
    LANGUAGE_KEYWORDS,
    SCORING_CONFIG,
    USE_CASE_WEIGHT_PRESETS,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class ScoringResult:
    """
    Full score breakdown for a single model row.

    Attributes
    ----------
    model_name          : HuggingFace model ID.
    overall_score       : Final suitability score in [0, 100].
    benchmark_score     : Normalized, use-case-weighted benchmark performance
                          component (before global weight application), [0, 100].
    use_case_score      : Task-alignment component [0, 100].
    language_score      : Language compatibility component [0, 100].
    size_score          : Size-preference component [0, 100].
    completeness_score  : Data-availability component [0, 100].
    benchmark_contributions : Per-benchmark column contribution to the
                              benchmark component (absolute, [0, 100]).
    weights_used        : The five component weights actually applied.
    benchmark_weights   : Per-column benchmark weights used.
    missing_benchmarks  : Columns that were NaN and excluded.
    notes               : Human-readable explanation list.
    """
    model_name:              str
    overall_score:           float
    benchmark_score:         float
    use_case_score:          float
    language_score:          float
    size_score:              float
    completeness_score:      float
    benchmark_contributions: dict[str, float]  = field(default_factory=dict)
    weights_used:            dict[str, float]  = field(default_factory=dict)
    benchmark_weights:       dict[str, float]  = field(default_factory=dict)
    missing_benchmarks:      list[str]         = field(default_factory=list)
    notes:                   list[str]         = field(default_factory=list)

    def as_dict(self) -> dict:
        """Flat dict suitable for adding to a DataFrame row."""
        return {
            "model_name":            self.model_name,
            "overall_score":         round(self.overall_score, 4),
            "score_benchmark":       round(self.benchmark_score, 4),
            "score_use_case":        round(self.use_case_score, 4),
            "score_language":        round(self.language_score, 4),
            "score_size":            round(self.size_score, 4),
            "score_completeness":    round(self.completeness_score, 4),
        }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_component_weights(config: dict | None = None) -> dict[str, float]:
    """Return the five component weights, normalised to sum to 1.0."""
    cfg = config or SCORING_CONFIG
    raw = cfg["weights"]
    total = sum(raw.values())
    if total <= 0:
        raise ValueError("SCORING_CONFIG['weights'] must have positive values.")
    return {k: v / total for k, v in raw.items()}


def _normalize_benchmark(
    raw: float,
    col: str,
    score_min: float = 0.0,
    score_max: float = 100.0,
) -> float:
    """
    Normalize a single raw benchmark score to [0, 1].

    Applies column-specific floor/ceiling from constants, then min-max
    normalizes using the global reference range.
    """
    floor   = BENCHMARK_SCORE_FLOORS.get(col, score_min)
    ceiling = BENCHMARK_SCORE_CEILINGS.get(col, score_max)
    clipped = max(floor, min(ceiling, raw))
    denom = ceiling - floor
    if denom <= 0:
        return 1.0
    return (clipped - floor) / denom  # in [0, 1]


def _cosine_similarity(a: dict[str, float], b: dict[str, float]) -> float:
    """
    Cosine similarity between two dicts sharing some keys.
    Returns 0.0 if either vector has zero magnitude.
    """
    keys = set(a) & set(b)
    if not keys:
        return 0.0
    dot  = sum(a[k] * b[k] for k in keys)
    mag_a = math.sqrt(sum(v ** 2 for k, v in a.items() if k in keys))
    mag_b = math.sqrt(sum(v ** 2 for k, v in b.items() if k in keys))
    if mag_a * mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


def _detect_language_support(
    row: pd.Series,
    language_codes: Sequence[str],
    language_col: str = "languages",
    name_col:     str = "model_name",
    config:       dict | None = None,
) -> float:
    """
    Return a language compatibility score in [0, 1].

    Confirmed support -> language_confirmed_score (default 1.0)
    Unknown / no info -> language_unknown_score   (default 0.6)
    Confirmed missing -> language_missing_score   (default 0.0)

    English is assumed universal (always confirmed).
    """
    cfg = config or SCORING_CONFIG
    confirmed = cfg.get("language_confirmed_score", 1.0)
    unknown   = cfg.get("language_unknown_score",   0.6)
    missing   = cfg.get("language_missing_score",   0.0)

    non_english = [c.lower() for c in language_codes if c.lower() != "en"]
    if not non_english:
        return confirmed  # English-only request, trivially satisfied

    scores_per_lang: list[float] = []

    for code in non_english:
        # 1. Dedicated language column
        if language_col in row.index and pd.notna(row[language_col]) and row[language_col] != "":
            raw = row[language_col]
            if isinstance(raw, list):
                known = [str(x).lower().strip() for x in raw]
            else:
                import re
                raw_s = str(raw).strip().strip("[]").replace("'", "").replace('"', "")
                known = [x.strip().lower() for x in re.split(r"[,|;]", raw_s) if x.strip()]
            if "multilingual" in known:
                scores_per_lang.append(confirmed)
                continue
            matched = any(code == k or k.startswith(code) for k in known)
            scores_per_lang.append(confirmed if matched else missing)
            continue

        # 2. Keyword heuristic on model name
        if name_col in row.index:
            name_lower = str(row[name_col]).lower()
            keywords = LANGUAGE_KEYWORDS.get(code, [])
            if not keywords:
                scores_per_lang.append(unknown)
            elif any(kw in name_lower for kw in keywords):
                scores_per_lang.append(confirmed)
            else:
                scores_per_lang.append(unknown)
            continue

        # 3. No information
        scores_per_lang.append(unknown)

    return float(np.mean(scores_per_lang)) if scores_per_lang else confirmed


def _size_preference_score(
    params_b: float | None,
    preferred_min_b: float | None,
    preferred_max_b: float | None,
    config: dict | None = None,
) -> float:
    """
    Return a size-preference score in [0, 1].

    Score = 1.0 if params_b is within [preferred_min_b, preferred_max_b].
    Score decays linearly to 0 at ``size_tolerance_b`` outside the range.
    Score = 0.6 if params_b is unknown (don't heavily penalise unknowns).
    No preference given (both None) -> 1.0.
    """
    cfg       = config or SCORING_CONFIG
    tolerance = cfg.get("size_tolerance_b", 30.0)

    if preferred_min_b is None and preferred_max_b is None:
        return 1.0  # No preference expressed

    if params_b is None or (isinstance(params_b, float) and math.isnan(params_b)):
        return 0.6  # Unknown size — mildly penalised but not excluded

    lo = preferred_min_b if preferred_min_b is not None else 0.0
    hi = preferred_max_b if preferred_max_b is not None else 1e6

    if lo <= params_b <= hi:
        return 1.0

    # Distance to nearest boundary
    dist = min(abs(params_b - lo), abs(params_b - hi))
    score = max(0.0, 1.0 - dist / tolerance)
    return score


# ---------------------------------------------------------------------------
# Core single-row scorer
# ---------------------------------------------------------------------------

def score_model(
    row: pd.Series,
    *,
    use_case: str = "General Purpose",
    benchmark_weights: dict[str, float] | None = None,
    required_languages: Sequence[str] | None = None,
    preferred_min_b: float | None = None,
    preferred_max_b: float | None = None,
    config: dict | None = None,
) -> ScoringResult:
    """
    Compute the full suitability score for a single model row.

    Parameters
    ----------
    row                : A pandas Series (one row from the leaderboard DataFrame).
    use_case           : Key from USE_CASE_WEIGHT_PRESETS.  Determines
                         benchmark weights and use-case alignment scoring.
    benchmark_weights  : Override benchmark weights. Falls back to the
                         use-case preset, then DEFAULT_BENCHMARK_WEIGHTS.
    required_languages : BCP-47 language codes to score against.
    preferred_min_b    : Lower bound of preferred model size (billions).
    preferred_max_b    : Upper bound of preferred model size (billions).
    config             : Override SCORING_CONFIG from constants.

    Returns
    -------
    ScoringResult with overall_score and all five component scores.
    """
    cfg         = config or SCORING_CONFIG
    comp_weights = _get_component_weights(cfg)
    notes: list[str] = []

    model_name = str(row.get("model_name", "unknown"))

    # ── Resolve benchmark weights ────────────────────────────────────────────
    if benchmark_weights:
        bw = benchmark_weights
        notes.append("Using custom benchmark weights.")
    elif use_case in USE_CASE_WEIGHT_PRESETS:
        bw = USE_CASE_WEIGHT_PRESETS[use_case]
        notes.append(f"Using '{use_case}' benchmark weight preset.")
    else:
        bw = DEFAULT_BENCHMARK_WEIGHTS
        notes.append("Use case not recognised; using default benchmark weights.")

    # ── 1. Benchmark component ───────────────────────────────────────────────
    score_min = cfg.get("benchmark_score_min", 0.0)
    score_max = cfg.get("benchmark_score_max", 100.0)

    available_cols   = [c for c in BENCHMARK_COLUMNS if c in row.index and pd.notna(row[c])]
    missing_cols     = [c for c in BENCHMARK_COLUMNS if c not in available_cols]
    benchmark_contribs: dict[str, float] = {}

    if available_cols:
        # Normalize and weight each available benchmark
        total_bw = sum(bw.get(c, 0.0) for c in available_cols)
        if total_bw <= 0:
            total_bw = 1.0  # Safety: avoid zero division

        weighted_sum = 0.0
        for col in available_cols:
            raw   = float(row[col])
            norm  = _normalize_benchmark(raw, col, score_min, score_max)   # [0, 1]
            w     = bw.get(col, 0.0) / total_bw                            # re-normalised
            contribution = norm * w * 100.0                                 # contribution in [0, 100]
            benchmark_contribs[col] = round(contribution, 4)
            weighted_sum += norm * w

        benchmark_score = weighted_sum * 100.0  # [0, 100]
    else:
        benchmark_score = 0.0
        notes.append("No benchmark data available; benchmark score = 0.")

    if missing_cols:
        notes.append(f"Missing benchmarks (excluded): {missing_cols}")

    # ── 2. Use-case alignment component (cosine similarity) ──────────────────
    if use_case in USE_CASE_WEIGHT_PRESETS and available_cols:
        task_weights  = {c: bw.get(c, 0.0) for c in available_cols}
        model_scores  = {
            c: _normalize_benchmark(float(row[c]), c, score_min, score_max)
            for c in available_cols
        }
        sim = _cosine_similarity(task_weights, model_scores)
        use_case_score = sim * 100.0
    else:
        use_case_score = 50.0  # neutral when no use-case given
        notes.append("Use-case alignment defaulted to 50 (no preset or no data).")

    # ── 3. Language component ────────────────────────────────────────────────
    langs = list(required_languages) if required_languages else []
    if langs:
        lang_raw     = _detect_language_support(row, langs, config=cfg)
        language_score = lang_raw * 100.0
    else:
        language_score = 100.0  # No constraint — full score
        notes.append("No language constraint; language score = 100.")

    # ── 4. Size preference component ─────────────────────────────────────────
    params_b = row.get("params_b", None)
    if pd.notna(params_b) if params_b is not None else False:
        try:
            params_b = float(params_b)
        except (TypeError, ValueError):
            params_b = None

    size_raw     = _size_preference_score(params_b, preferred_min_b, preferred_max_b, cfg)
    size_score   = size_raw * 100.0

    if preferred_min_b is None and preferred_max_b is None:
        notes.append("No size preference; size score = 100.")
    elif params_b is None:
        notes.append(f"Model size unknown; size score = {size_score:.1f} (penalised).")

    # ── 5. Data completeness component ───────────────────────────────────────
    n_total      = len(BENCHMARK_COLUMNS)
    n_available  = len(available_cols)
    completeness_score = (n_available / n_total) * 100.0 if n_total > 0 else 100.0

    if n_available < n_total:
        notes.append(
            f"Data completeness: {n_available}/{n_total} benchmarks available "
            f"({completeness_score:.0f}%)."
        )

    # ── Overall score ────────────────────────────────────────────────────────
    w = comp_weights
    overall = (
        w["benchmark"]        * benchmark_score  +
        w["use_case"]         * use_case_score   +
        w["language"]         * language_score   +
        w["size_preference"]  * size_score       +
        w["data_completeness"]* completeness_score
    )
    overall = min(100.0, max(0.0, overall))

    return ScoringResult(
        model_name           = model_name,
        overall_score        = round(overall, 4),
        benchmark_score      = round(benchmark_score, 4),
        use_case_score       = round(use_case_score, 4),
        language_score       = round(language_score, 4),
        size_score           = round(size_score, 4),
        completeness_score   = round(completeness_score, 4),
        benchmark_contributions = benchmark_contribs,
        weights_used         = {k: round(v, 4) for k, v in w.items()},
        benchmark_weights    = {k: round(v, 4) for k, v in bw.items()},
        missing_benchmarks   = missing_cols,
        notes                = notes,
    )


# ---------------------------------------------------------------------------
# DataFrame-level scorer
# ---------------------------------------------------------------------------

# Output column names added to the DataFrame
SCORE_COLUMNS = [
    "overall_score",
    "score_benchmark",
    "score_use_case",
    "score_language",
    "score_size",
    "score_completeness",
]


def score_dataframe(
    df: pd.DataFrame,
    *,
    use_case: str = "General Purpose",
    benchmark_weights: dict[str, float] | None = None,
    required_languages: Sequence[str] | None = None,
    preferred_min_b: float | None = None,
    preferred_max_b: float | None = None,
    config: dict | None = None,
) -> pd.DataFrame:
    """
    Score every row in df and append score component columns.

    Parameters
    ----------
    df                 : Leaderboard DataFrame (filtered candidates).
    use_case           : Task key for weight preset selection.
    benchmark_weights  : Optional per-benchmark weight override.
    required_languages : BCP-47 codes to score language compatibility against.
    preferred_min_b    : Lower bound of acceptable model size (billions).
    preferred_max_b    : Upper bound of acceptable model size (billions).
    config             : Override SCORING_CONFIG.

    Returns
    -------
    New DataFrame with ``SCORE_COLUMNS`` appended.  Original columns unchanged.
    Rows are **not** sorted here — use ``rank_scored_df`` for that.
    """
    if df.empty:
        result = df.copy()
        for col in SCORE_COLUMNS:
            result[col] = float("nan")
        return result

    scored_rows: list[dict] = []
    for _, row in df.iterrows():
        result = score_model(
            row,
            use_case=use_case,
            benchmark_weights=benchmark_weights,
            required_languages=required_languages,
            preferred_min_b=preferred_min_b,
            preferred_max_b=preferred_max_b,
            config=config,
        )
        scored_rows.append(result.as_dict())

    scores_df = pd.DataFrame(scored_rows)

    # Drop the model_name duplicate before merging
    scores_df = scores_df.drop(columns=["model_name"], errors="ignore")

    result_df = df.reset_index(drop=True).copy()
    for col in SCORE_COLUMNS:
        result_df[col] = scores_df[col].values

    return result_df


def rank_scored_df(df: pd.DataFrame, score_col: str = "overall_score") -> pd.DataFrame:
    """
    Sort a scored DataFrame by ``score_col`` descending and add a ``rank`` column.

    Parameters
    ----------
    df        : DataFrame that already has score columns (output of score_dataframe).
    score_col : Column to sort by.

    Returns
    -------
    Sorted DataFrame with a prepended ``rank`` column (1-indexed).
    """
    if df.empty:
        df = df.copy()
        df.insert(0, "rank", pd.Series(dtype=int))
        return df

    df = df.sort_values(score_col, ascending=False).reset_index(drop=True)
    df.insert(0, "rank", range(1, len(df) + 1))
    return df


# ---------------------------------------------------------------------------
# Convenience wrappers (backwards-compatible with recommender.py)
# ---------------------------------------------------------------------------

def score_for_use_case(
    df: pd.DataFrame,
    use_case: str,
    *,
    required_languages: Sequence[str] | None = None,
    preferred_min_b: float | None = None,
    preferred_max_b: float | None = None,
    config: dict | None = None,
) -> pd.DataFrame:
    """
    Score and rank models for a given use case.

    Parameters
    ----------
    df       : Filtered leaderboard DataFrame.
    use_case : Key from USE_CASE_WEIGHT_PRESETS.

    Returns
    -------
    Ranked DataFrame with all SCORE_COLUMNS and a ``rank`` column.
    """
    if use_case not in USE_CASE_WEIGHT_PRESETS:
        raise ValueError(
            f"Unknown use case '{use_case}'. "
            f"Valid options: {list(USE_CASE_WEIGHT_PRESETS.keys())}"
        )
    scored = score_dataframe(
        df,
        use_case=use_case,
        required_languages=required_languages,
        preferred_min_b=preferred_min_b,
        preferred_max_b=preferred_max_b,
        config=config,
    )
    return rank_scored_df(scored)


def score_models(
    df: pd.DataFrame,
    weights: dict[str, float] | None = None,
    score_col: str = "overall_score",
) -> pd.DataFrame:
    """
    Backwards-compatible wrapper used by recommender.py.

    Scores and ranks the DataFrame using custom benchmark weights.
    The ``score_col`` argument is accepted but ignored (output column
    is always ``overall_score``).
    """
    scored = score_dataframe(df, benchmark_weights=weights)
    return rank_scored_df(scored)


def top_n_models(
    df: pd.DataFrame,
    n: int = 10,
    *,
    use_case: str = "General Purpose",
    required_languages: Sequence[str] | None = None,
    preferred_min_b: float | None = None,
    preferred_max_b: float | None = None,
    config: dict | None = None,
) -> pd.DataFrame:
    """
    Return the top-N models by overall suitability score.

    Parameters
    ----------
    df       : Filtered leaderboard DataFrame.
    n        : Number of top models to return.
    use_case : Task key for weight preset.

    Returns
    -------
    DataFrame with at most n rows.
    """
    ranked = score_for_use_case(
        df,
        use_case=use_case,
        required_languages=required_languages,
        preferred_min_b=preferred_min_b,
        preferred_max_b=preferred_max_b,
        config=config,
    )
    return ranked.head(n)
