"""
src/ui/components.py
--------------------
Reusable Streamlit UI components (cards, badges, tables, etc.)
shared across multiple pages.
"""

from __future__ import annotations

import streamlit as st
import pandas as pd


def model_card(
    name: str,
    score: float,
    params: str,
    licence: str,
    tags: list[str] | None = None,
    rank: int | None = None,
) -> None:
    """
    Render a single model card.

    Parameters
    ----------
    name    : Model name (e.g. "mistralai/Mistral-7B-Instruct-v0.2")
    score   : Aggregated benchmark score [0–100]
    params  : Parameter count string (e.g. "7B")
    licence : Licence string (e.g. "Apache 2.0")
    tags    : Optional list of tag strings to display as badges
    rank    : Optional rank number to display
    """
    tags_html = "".join(
        f'<span class="tag tag-accent" style="margin-right:4px;">{t}</span>'
        for t in (tags or [])
    )
    rank_html = (
        f'<span style="font-size:1.4rem;font-weight:800;color:var(--clr-primary-lt);">#{rank}</span>'
        if rank is not None
        else ""
    )

    score_color = (
        "var(--clr-success)" if score >= 75
        else "var(--clr-warning)" if score >= 55
        else "var(--clr-danger)"
    )

    st.markdown(
        f"""
        <div class="glass-card" style="margin-bottom:0.75rem;">
            <div style="display:flex;justify-content:space-between;align-items:flex-start;">
                <div>
                    {rank_html}
                    <span style="font-weight:700;font-size:1rem;margin-left:{'0.5rem' if rank else '0'};">
                        {name}
                    </span>
                </div>
                <div style="
                    font-size:1.4rem;font-weight:800;
                    color:{score_color};
                    background:rgba(0,0,0,0.3);
                    border-radius:8px;padding:2px 10px;
                ">
                    {score:.1f}
                </div>
            </div>
            <div style="margin-top:0.5rem;color:var(--clr-muted);font-size:0.82rem;">
                {params} params &nbsp;·&nbsp; Licence: {licence}
            </div>
            <div style="margin-top:0.5rem;">{tags_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def benchmark_table(df: pd.DataFrame, height: int = 400) -> None:
    """
    Display a styled benchmark dataframe.

    Parameters
    ----------
    df     : DataFrame containing model benchmark columns
    height : Height of the table container in pixels
    """
    st.dataframe(
        df,
        use_container_width=True,
        height=height,
        hide_index=True,
    )


def score_badge(score: float, label: str = "Score") -> None:
    """
    Render a coloured circular score badge.

    Parameters
    ----------
    score : Value between 0 and 100
    label : Display label below the badge
    """
    color = (
        "var(--clr-success)" if score >= 75
        else "var(--clr-warning)" if score >= 55
        else "var(--clr-danger)"
    )
    st.markdown(
        f"""
        <div style="text-align:center;">
            <div style="
                width:80px;height:80px;border-radius:50%;
                border:4px solid {color};
                display:inline-flex;align-items:center;justify-content:center;
                font-size:1.4rem;font-weight:800;color:{color};
                margin-bottom:0.25rem;
            ">{score:.0f}</div>
            <div style="color:var(--clr-muted);font-size:0.8rem;">{label}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_header(title: str, subtitle: str | None = None) -> None:
    """
    Render a styled section header with optional subtitle.

    Parameters
    ----------
    title    : Main heading text
    subtitle : Optional secondary description
    """
    sub_html = (
        f'<p style="color:var(--clr-muted);font-size:0.9rem;margin:0.25rem 0 0;">{subtitle}</p>'
        if subtitle
        else ""
    )
    st.markdown(
        f"""
        <div style="margin:1.5rem 0 1rem;">
            <h2 style="font-size:1.4rem;font-weight:700;margin:0;">{title}</h2>
            {sub_html}
        </div>
        <hr style="border-color:var(--clr-border);margin-bottom:1.5rem;">
        """,
        unsafe_allow_html=True,
    )


def info_banner(message: str, kind: str = "info") -> None:
    """
    Render a coloured information banner.

    Parameters
    ----------
    message : HTML-safe message string
    kind    : One of "info" | "success" | "warning" | "error"
    """
    palette = {
        "info":    ("var(--clr-accent)",   "rgba(6,182,212,0.1)",  "ℹ️"),
        "success": ("var(--clr-success)",  "rgba(16,185,129,0.1)", "✅"),
        "warning": ("var(--clr-warning)",  "rgba(245,158,11,0.1)", "⚠️"),
        "error":   ("var(--clr-danger)",   "rgba(239,68,68,0.1)",  "❌"),
    }
    clr, bg, icon = palette.get(kind, palette["info"])
    st.markdown(
        f"""
        <div style="
            background:{bg};border:1px solid {clr};
            border-left:4px solid {clr};
            border-radius:var(--radius-sm);
            padding:0.75rem 1rem;margin:0.5rem 0;
        ">
            {icon} {message}
        </div>
        """,
        unsafe_allow_html=True,
    )
