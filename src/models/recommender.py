"""
src/models/recommender.py
--------------------------
Use-case-driven model recommendation engine.

Chains apply_filters() → score_models() and returns both the ranked
DataFrame and the FilterResult audit trail so the UI can report how
many models survived each stage.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from typing import Any

from src.models.filter import FilterResult, apply_filters
from src.models.scorer import score_for_use_case, score_models
from src.utils.constants import USE_CASE_WEIGHT_PRESETS, BENCHMARK_COLUMNS


@dataclass
class ModelRecommendation:
    """Detailed recommendation explanation for a single model."""
    model_name: str
    rank: int
    overall_score: float
    why_this_model: str
    strengths: list[str]
    tradeoffs: list[str]
    raw_data: dict[str, Any]


@dataclass
class RecommendationResult:
    """
    Container returned by ``recommend``.

    Attributes
    ----------
    df                : Ranked DataFrame (top-N rows) with 'overall_score'
                        and 'rank' columns prepended.
    filter_result     : The full FilterResult from the filtering stage.
    use_case          : The use-case key that was scored against.
    recommendations   : List of detailed ModelRecommendation objects.
    benchmark_sources : List of benchmarks used for this recommendation.
    highest_scoring   : The top ModelRecommendation (if any).
    """
    df:                pd.DataFrame
    filter_result:     FilterResult
    use_case:          str
    recommendations:   list[ModelRecommendation]
    benchmark_sources: list[str]
    highest_scoring:   ModelRecommendation | None


def _generate_model_insights(row: pd.Series, use_case: str) -> tuple[str, list[str], list[str]]:
    """Generate deterministic text insights based purely on the scored row data."""
    strengths = []
    tradeoffs = []

    uc_score = row.get("score_use_case", 0.0)
    size_score = row.get("score_size", 0.0)
    lang_score = row.get("score_language", 0.0)
    
    top_bms = []
    low_bms = []
    missing_bms = []
    
    for bm in BENCHMARK_COLUMNS:
        val = row.get(bm, float("nan"))
        if pd.isna(val):
            missing_bms.append(bm)
        elif val >= 70:
            top_bms.append(bm)
        elif val < 50:
            low_bms.append(bm)
            
    if top_bms:
        strengths.append(f"Strong performance on {', '.join(top_bms).upper()}.")
    if uc_score >= 80:
        strengths.append(f"Benchmark profile strongly aligns with the '{use_case}' task.")
    if size_score == 100:
        strengths.append("Perfectly matches the preferred size range.")
    if lang_score == 100:
        strengths.append("Confirmed support for the required languages.")
        
    if missing_bms:
        tradeoffs.append(f"No data available for {', '.join(missing_bms).upper()}.")
    if low_bms:
        tradeoffs.append(f"Weaker performance on {', '.join(low_bms).upper()}.")
    if size_score < 100:
        tradeoffs.append("Falls outside the preferred parameter size constraints.")
    if lang_score < 100:
        tradeoffs.append("Language compatibility is uncertain or incomplete.")

    if not strengths:
        strengths.append("Solid baseline performance across standard metrics.")
    if not tradeoffs:
        tradeoffs.append("No major benchmark weaknesses found based on current data.")

    model_name = row.get("model_name", "This model")
    overall = row.get("overall_score", 0.0)
    
    why = f"{model_name} achieves an overall suitability score of {overall:.1f}/100 for {use_case}. "
    if top_bms:
        why += f"It excels particularly in {top_bms[0].upper()} ({row[top_bms[0]]:.1f}). "
    if missing_bms:
        why += f"Note that it hasn't been evaluated on {len(missing_bms)} standard benchmark(s)."
        
    return why.strip(), strengths, tradeoffs


def recommend(
    df: pd.DataFrame,
    *,
    use_case: str = "General Purpose",
    top_n: int = 5,
    # Size
    min_params_b: float = 0.0,
    max_params_b: float = 10_000.0,
    include_unknown_size: bool = True,
    # Language
    required_languages: list[str] | None = None,
    # Licence
    licences: list[str] | None = None,
    commercial_only: bool = False,
    # Model type
    model_types: list[str] | None = None,
    # Score floor
    min_average_score: float | None = None,
    per_benchmark_mins: dict[str, float] | None = None,
    # Benchmark availability (auto-derived from use_case if not set)
    required_benchmarks: list[str] | None = None,
    min_benchmarks_available: int | None = None,
    # Name search
    name_query: str = "",
) -> RecommendationResult:
    """
    Filter, score, and return the top-N models for a given use case.

    Parameters
    ----------
    df       : Full leaderboard DataFrame from src.data.fetcher.
    use_case : Key from USE_CASE_WEIGHT_PRESETS.
    top_n    : Maximum number of results to return.

    All other parameters are forwarded directly to apply_filters().

    Returns
    -------
    RecommendationResult with ranked df, full filter audit, and use_case label.
    """
    if use_case not in USE_CASE_WEIGHT_PRESETS:
        raise ValueError(
            f"Unknown use case '{use_case}'. "
            f"Valid options: {list(USE_CASE_WEIGHT_PRESETS.keys())}"
        )

    # Step 1 — Filter
    filter_result = apply_filters(
        df,
        use_case=use_case,
        min_params_b=min_params_b,
        max_params_b=max_params_b,
        include_unknown_size=include_unknown_size,
        required_languages=required_languages,
        licences=licences,
        commercial_only=commercial_only,
        model_types=model_types,
        min_average_score=min_average_score,
        per_benchmark_mins=per_benchmark_mins,
        required_benchmarks=required_benchmarks,
        min_benchmarks_available=min_benchmarks_available,
        name_query=name_query,
    )

    candidates = filter_result.df

    # Step 2 — Score + rank
    if candidates.empty:
        ranked = candidates.copy()
    else:
        ranked = score_for_use_case(
            candidates,
            use_case,
            required_languages=required_languages,
            preferred_min_b=min_params_b if min_params_b > 0 else None,
            preferred_max_b=max_params_b if max_params_b < 10_000 else None,
        ).head(top_n)

    # Step 3 — Generate detailed insights
    recommendations = []
    for _, row in ranked.iterrows():
        why, strengths, tradeoffs = _generate_model_insights(row, use_case)
        rec = ModelRecommendation(
            model_name=str(row["model_name"]),
            rank=int(row["rank"]),
            overall_score=float(row["overall_score"]),
            why_this_model=why,
            strengths=strengths,
            tradeoffs=tradeoffs,
            raw_data=row.to_dict(),
        )
        recommendations.append(rec)

    highest = recommendations[0] if recommendations else None
    sources_used = [bm for bm in BENCHMARK_COLUMNS if bm in df.columns]

    return RecommendationResult(
        df=ranked,
        filter_result=filter_result,
        use_case=use_case,
        recommendations=recommendations,
        benchmark_sources=sources_used,
        highest_scoring=highest,
    )


def list_use_cases() -> list[str]:
    """Return all supported use-case preset names."""
    return list(USE_CASE_WEIGHT_PRESETS.keys())
