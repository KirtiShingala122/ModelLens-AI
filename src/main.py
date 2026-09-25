"""
src/main.py
-----------
Main backend entry point for ModelLens AI.

Exposes a clean function to run the full pipeline:
Requirements -> Fetch -> Filter -> Score -> Recommend
"""

from __future__ import annotations

import logging
from typing import Any

from src.data.fetcher import fetch_leaderboard
from src.models.recommender import recommend, RecommendationResult

logger = logging.getLogger(__name__)


def run_pipeline(
    use_case: str = "General Purpose",
    top_n: int = 5,
    min_params_b: float = 0.0,
    max_params_b: float = 10_000.0,
    required_languages: list[str] | None = None,
    commercial_only: bool = False,
    min_average_score: float | None = None,
) -> dict[str, Any]:
    """
    Run the complete ModelLens AI backend pipeline.

    Flow:
    1. Validate requirements.
    2. Fetch live data (with error handling).
    3. Filter and score models based on constraints.
    4. Return structured recommendations.

    Parameters
    ----------
    use_case          : The task preset (e.g., "Coding & Math").
    top_n             : Max number of models to return.
    min_params_b      : Min parameter count in billions.
    max_params_b      : Max parameter count in billions.
    required_languages: List of BCP-47 language codes (e.g., ["en", "zh"]).
    commercial_only   : If True, restricts to commercial-friendly licences.
    min_average_score : Minimum acceptable average benchmark score.

    Returns
    -------
    dict with keys:
      - "status": "success" or "error"
      - "message": Human-readable status or error message
      - "recommendations": List of dicts with detailed model insights
      - "benchmark_sources": List of benchmarks used
      - "audit": Filter and scoring audit trail
    """
    # 1. Validation
    if min_params_b < 0 or max_params_b < min_params_b:
        return {
            "status": "error",
            "message": "Invalid parameter range requirements.",
            "recommendations": [],
            "benchmark_sources": [],
            "audit": []
        }

    # 2. Fetch Live Data
    try:
        df = fetch_leaderboard()
        if df.empty:
            return {
                "status": "error",
                "message": "Unable to fetch live data from the Hugging Face API right now. The server returned an empty dataset or timed out. Please check your network or try again later.",
                "recommendations": [],
                "benchmark_sources": [],
                "audit": []
            }
    except Exception as e:
        logger.error(f"Data fetch failed: {e}")
        return {
            "status": "error",
            "message": f"API/network failure while fetching leaderboard data: {str(e)}",
            "recommendations": [],
            "benchmark_sources": [],
            "audit": []
        }

    # 3. Filter and Score
    try:
        rec_result = recommend(
            df,
            use_case=use_case,
            top_n=top_n,
            min_params_b=min_params_b,
            max_params_b=max_params_b,
            required_languages=required_languages,
            commercial_only=commercial_only,
            min_average_score=min_average_score,
        )
    except ValueError as e:
        # E.g. invalid use_case
        return {
            "status": "error",
            "message": f"Invalid requirements: {str(e)}",
            "recommendations": [],
            "benchmark_sources": [],
            "audit": []
        }
    except Exception as e:
        logger.error(f"Recommendation engine failed: {e}")
        return {
            "status": "error",
            "message": f"Internal error during scoring: {str(e)}",
            "recommendations": [],
            "benchmark_sources": [],
            "audit": []
        }

    # 4. Handle 'no matching models'
    if not rec_result.recommendations:
        warnings = rec_result.filter_result.warnings
        msg = "No matching models found for the given requirements."
        if warnings:
            msg += f" Hints: {' '.join(warnings)}"
        return {
            "status": "error",
            "message": msg,
            "recommendations": [],
            "benchmark_sources": rec_result.benchmark_sources,
            "audit": rec_result.filter_result.stage_log
        }

    # 5. Format Success Result
    recs_out = []
    for rec in rec_result.recommendations:
        recs_out.append({
            "model_name": rec.model_name,
            "rank": rec.rank,
            "suitability_score": rec.overall_score,
            "why_this_model": rec.why_this_model,
            "strengths": rec.strengths,
            "tradeoffs": rec.tradeoffs,
            "score_components": {
                "benchmark": rec.raw_data.get("score_benchmark", 0.0),
                "use_case": rec.raw_data.get("score_use_case", 0.0),
                "language": rec.raw_data.get("score_language", 0.0),
                "size": rec.raw_data.get("score_size", 0.0),
                "completeness": rec.raw_data.get("score_completeness", 0.0),
            },
            "benchmark_data": {
                bm: rec.raw_data.get(bm) for bm in rec_result.benchmark_sources
            }
        })

    return {
        "status": "success",
        "message": f"Successfully generated top {len(recs_out)} recommendations for {use_case}.",
        "recommendations": recs_out,
        "benchmark_sources": rec_result.benchmark_sources,
        "audit": rec_result.filter_result.stage_log
    }

if __name__ == "__main__":
    # Test script execution
    res = run_pipeline(
        use_case="Coding & Math",
        min_params_b=5.0,
        max_params_b=20.0
    )
    import json
    print(json.dumps(res, indent=2))
