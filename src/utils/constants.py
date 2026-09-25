"""
src/utils/constants.py
-----------------------
App-wide constants: feature highlights, quick stats, benchmark source
metadata, model categories, and column name mappings.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Homepage content
# ---------------------------------------------------------------------------

QUICK_STATS: list[dict] = [
    {"label": "Open-Source Models Tracked",  "value": "3,200+",  "delta": "+142 this week"},
    {"label": "Benchmark Evaluations",        "value": "18,000+", "delta": "+850 this week"},
    {"label": "Leaderboard Sources",          "value": "6",       "delta": None},
    {"label": "Supported Use Cases",          "value": "25+",     "delta": None},
]

FEATURE_HIGHLIGHTS: list[dict] = [
    {
        "icon": "LC",
        "title": "Live Benchmark Leaderboard",
        "description": (
            "Real-time scores from HuggingFace Open LLM Leaderboard covering "
            "ARC, HellaSwag, MMLU, TruthfulQA, Winogrande, and GSM8K."
        ),
    },
    {
        "icon": "FL",
        "title": "Smart Model Filtering",
        "description": (
            "Filter by parameter count, licence type, model family, language "
            "support, quantisation, and benchmark task category."
        ),
    },
    {
        "icon": "RC",
        "title": "AI-Powered Recommender",
        "description": (
            "Describe your task in plain English. Our scoring engine ranks "
            "models by weighted benchmark performance for your specific needs."
        ),
    },
    {
        "icon": "VS",
        "title": "Side-by-Side Comparison",
        "description": (
            "Select up to 5 models and compare them across every benchmark "
            "metric with interactive radar and bar charts."
        ),
    },
    {
        "icon": "VZ",
        "title": "Rich Visualizations",
        "description": (
            "Explore score distributions, performance-vs-size scatter plots, "
            "timeline trends, and correlation heat maps built with Plotly."
        ),
    },
    {
        "icon": "RT",
        "title": "Auto-Refreshing Data",
        "description": (
            "Benchmark data is fetched live from the HuggingFace Hub dataset "
            "and cached locally so your leaderboard is always up to date."
        ),
    },
]

BENCHMARK_SOURCES: list[dict] = [
    {
        "icon": "HF",
        "name": "HF Open LLM Leaderboard",
        "description": "Primary benchmark hub — ARC, HellaSwag, MMLU, TruthfulQA, Winogrande, GSM8K",
    },
    {
        "icon": "CB",
        "name": "LMSYS Chatbot Arena",
        "description": "Human preference ELO scores from blind pairwise comparisons",
    },
    {
        "icon": "BB",
        "name": "BIG-bench Hard",
        "description": "Challenging reasoning tasks beyond standard benchmarks",
    },
    {
        "icon": "HL",
        "name": "HELM",
        "description": "Holistic Evaluation of Language Models across 42+ scenarios",
    },
]

# ---------------------------------------------------------------------------
# Benchmark column metadata
# ---------------------------------------------------------------------------

BENCHMARK_COLUMNS: list[str] = [
    "ifeval",
    "bbh",
    "math_lvl5",
    "gpqa",
    "musr",
    "mmlu_pro",
]

BENCHMARK_LABELS: dict[str, str] = {
    "ifeval":    "IFEval",
    "bbh":       "BBH",
    "math_lvl5": "MATH Lvl 5",
    "gpqa":      "GPQA",
    "musr":      "MUSR",
    "mmlu_pro":  "MMLU-PRO",
}

# Weights used when computing the aggregated composite score
DEFAULT_BENCHMARK_WEIGHTS: dict[str, float] = {
    "ifeval":    0.15,
    "bbh":       0.15,
    "math_lvl5": 0.20,
    "gpqa":      0.15,
    "musr":      0.15,
    "mmlu_pro":  0.20,
}

# ---------------------------------------------------------------------------
# Scoring engine configuration
# ---------------------------------------------------------------------------

# High-level weights for the five score components.
# Must sum to 1.0.  All values configurable here — no magic numbers in scorer.
SCORING_CONFIG: dict = {
    # Component weights (must sum to 1.0)
    "weights": {
        "benchmark":       0.55,   # weighted benchmark performance
        "use_case":        0.20,   # how well benchmarks align to the task
        "language":        0.10,   # language support match
        "size_preference": 0.05,   # closeness to the user's preferred size
        "data_completeness": 0.10, # fraction of benchmarks that have values
    },

    # Benchmark score normalisation.
    # Raw HF benchmark scores are already on 0-100 scale, so the reference
    # range is 0-100. Adjust if you add benchmarks with different scales.
    "benchmark_score_min": 0.0,
    "benchmark_score_max": 100.0,

    # Perfect size match: score = 1.0 when preferred_b == model_b.
    # Penalty is linear; at this many billions away the size score = 0.
    "size_tolerance_b": 30.0,

    # Language: score given when the model is *confirmed* to support the
    # language vs. when it is *unknown* (we don't penalise unknowns heavily).
    "language_confirmed_score": 1.0,
    "language_unknown_score":   0.6,
    "language_missing_score":   0.0,
}

# Known benchmark score floors — used to clip pathological raw values.
# Any raw score below the floor is treated as the floor (not zero) to
# avoid unfairly penalising models with a single weak benchmark.
BENCHMARK_SCORE_FLOORS: dict[str, float] = {
    "ifeval":     0.0,
    "bbh":        0.0,
    "math_lvl5":  0.0,
    "gpqa":       0.0,
    "musr":       0.0,
    "mmlu_pro":   0.0,
}

# Ceiling: scores above this are capped at 100 before normalisation.
BENCHMARK_SCORE_CEILINGS: dict[str, float] = {
    "ifeval":    100.0,
    "bbh":       100.0,
    "math_lvl5": 100.0,
    "gpqa":      100.0,
    "musr":      100.0,
    "mmlu_pro":  100.0,
}


# ---------------------------------------------------------------------------
# Use-case benchmark weight presets
# ---------------------------------------------------------------------------

USE_CASE_WEIGHT_PRESETS: dict[str, dict[str, float]] = {
    "General Purpose": {
        "ifeval": 0.20, "bbh": 0.15, "math_lvl5": 0.15,
        "gpqa": 0.15, "musr": 0.15, "mmlu_pro": 0.20,
    },
    "Coding & Math": {
        "ifeval": 0.15, "bbh": 0.10, "math_lvl5": 0.40,
        "gpqa": 0.05, "musr": 0.05, "mmlu_pro": 0.25,
    },
    "Instruction Following": {
        "ifeval": 0.40, "bbh": 0.15, "math_lvl5": 0.05,
        "gpqa": 0.10, "musr": 0.10, "mmlu_pro": 0.20,
    },
    "Factual QA / RAG": {
        "ifeval": 0.10, "bbh": 0.10, "math_lvl5": 0.05,
        "gpqa": 0.35, "musr": 0.10, "mmlu_pro": 0.30,
    },
    "Creative Writing": {
        "ifeval": 0.30, "bbh": 0.10, "math_lvl5": 0.05,
        "gpqa": 0.15, "musr": 0.30, "mmlu_pro": 0.10,
    },
    "Reasoning": {
        "ifeval": 0.10, "bbh": 0.30, "math_lvl5": 0.15,
        "gpqa": 0.10, "musr": 0.15, "mmlu_pro": 0.20,
    },
}

# Benchmarks that are most indicative for each use-case / task.
# Used by filter_by_benchmark_availability to require at least these
# columns to be non-null before a model qualifies.
USE_CASE_REQUIRED_BENCHMARKS: dict[str, list[str]] = {
    "General Purpose":      ["ifeval", "mmlu_pro"],
    "Coding & Math":        ["math_lvl5", "mmlu_pro"],
    "Instruction Following": ["ifeval"],
    "Factual QA / RAG":     ["gpqa", "mmlu_pro"],
    "Creative Writing":     ["ifeval", "musr"],
    "Reasoning":            ["bbh", "mmlu_pro"],
}

# ---------------------------------------------------------------------------
# Model size buckets
# ---------------------------------------------------------------------------

SIZE_BUCKETS: list[dict] = [
    {"label": "Tiny  (< 3B)",   "min": 0,    "max": 3},
    {"label": "Small (3–7B)",   "min": 3,    "max": 7},
    {"label": "Medium (7–14B)", "min": 7,    "max": 14},
    {"label": "Large (14–35B)", "min": 14,   "max": 35},
    {"label": "XL   (35–70B)", "min": 35,   "max": 70},
    {"label": "XXL  (> 70B)",   "min": 70,   "max": 10_000},
]

# ---------------------------------------------------------------------------
# Licence categories
# ---------------------------------------------------------------------------

OPEN_LICENCES: list[str] = [
    "apache-2.0",
    "mit",
    "llama2",
    "llama3",
    "cc-by-4.0",
    "cc-by-sa-4.0",
    "bigscience-openrail-m",
    "openrail",
    "other",
]

COMMERCIAL_OK_LICENCES: list[str] = [
    "apache-2.0",
    "mit",
    "cc-by-4.0",
]

# ---------------------------------------------------------------------------
# Cache settings
# ---------------------------------------------------------------------------

CACHE_TTL_SECONDS: int = 7_200       # 2 hours — mirrors the reference repo
HF_DATASET_PATH: str = "open-llm-leaderboard/results"
HF_LEADERBOARD_REPO: str = "open-llm-leaderboard/open_llm_leaderboard"

# ---------------------------------------------------------------------------
# Language metadata
# ---------------------------------------------------------------------------

# Known multilingual / language-specific model name keywords.
# The language filter uses these to heuristically detect language support
# when a dedicated 'languages' column is absent from the DataFrame.
# Keys are BCP-47 language codes; values are case-insensitive substrings
# that appear in model IDs or tags for models known to support that language.
LANGUAGE_KEYWORDS: dict[str, list[str]] = {
    "en":  [],                               # English — assumed universal
    "zh":  ["qwen", "baichuan", "glm", "chinese", "zh"],
    "de":  ["leo", "german", "deutsch"],
    "fr":  ["croissant", "french", "fr-"],
    "ja":  ["japanese", "swallow", "elyza", "calm"],
    "ko":  ["korean", "ko-", "kollama"],
    "ar":  ["arabic", "ar-", "jais"],
    "es":  ["spanish", "es-"],
    "pt":  ["portuguese", "pt-", "sabia"],
    "ru":  ["russian", "ru-", "rugpt"],
    "hi":  ["hindi", "hi-", "airavata"],
    "multilingual": [
        "multilingual", "multi", "bloom", "xglm",
        "mbert", "xlm", "aya", "qwen",
    ],
}

# All supported language options surfaced in the UI
SUPPORTED_LANGUAGES: list[dict] = [
    {"code": "en",           "label": "English"},
    {"code": "zh",           "label": "Chinese (Mandarin)"},
    {"code": "de",           "label": "German"},
    {"code": "fr",           "label": "French"},
    {"code": "ja",           "label": "Japanese"},
    {"code": "ko",           "label": "Korean"},
    {"code": "ar",           "label": "Arabic"},
    {"code": "es",           "label": "Spanish"},
    {"code": "pt",           "label": "Portuguese"},
    {"code": "ru",           "label": "Russian"},
    {"code": "hi",           "label": "Hindi"},
    {"code": "multilingual", "label": "Multilingual (any)"},
]
