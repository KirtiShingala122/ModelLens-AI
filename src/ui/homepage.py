"""
src/ui/homepage.py
------------------
Renders the ModelLens AI homepage — hero section, feature cards,
live stats strip, and quick-start guide.
"""

from __future__ import annotations

import streamlit as st

from src.ui.layout import render_sidebar_nav
from src.utils.constants import BENCHMARK_SOURCES, FEATURE_HIGHLIGHTS, QUICK_STATS


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _hero_section() -> None:
    """Animated hero banner with headline and CTA buttons."""
    st.markdown(
        """
        <div style="
            text-align:center;
            padding: 3.5rem 1rem 2rem;
            background: radial-gradient(ellipse at 50% 0%, rgba(124,58,237,0.18) 0%, transparent 70%);
            border-radius: 24px;
            margin-bottom: 2rem;
        ">
            <div style="margin-bottom:0.75rem;">
                <span class="tag tag-success">
                    <span class="pulse-dot"></span>Live Benchmarks
                </span>
                &nbsp;
                <span class="tag tag-accent">Open Source Models</span>
                &nbsp;
                <span class="tag tag-primary">AI-Powered Scoring</span>
            </div>

            <h1 style="
                font-size:clamp(2.2rem,5vw,3.8rem);
                font-weight:800;
                line-height:1.15;
                margin:0.5rem 0;
                letter-spacing:-0.03em;
            ">
                Discover the <span class="gradient-text">Perfect LLM</span><br>for Your Use Case
            </h1>

            <p style="
                color:var(--clr-muted);
                font-size:1.1rem;
                max-width:640px;
                margin:1rem auto 0;
                line-height:1.7;
            ">
                ModelLens AI aggregates live benchmark data from <strong>HuggingFace Open LLM Leaderboard</strong>,
                applies intelligent scoring, and surfaces the best open-source models
                tailored to your task — in seconds.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # CTA buttons
    col_a, col_b, col_c = st.columns([1, 1, 1])
    with col_a:
        if st.button("Find My Model", use_container_width=True, key="cta_recommend"):
            st.session_state.active_page = "recommender"
            st.rerun()
    with col_b:
        if st.button("View Leaderboard", use_container_width=True, key="cta_leaderboard"):
            st.session_state.active_page = "leaderboard"
            st.rerun()
    with col_c:
        if st.button("Compare Models", use_container_width=True, key="cta_compare"):
            st.session_state.active_page = "compare"
            st.rerun()


def _stats_strip() -> None:
    """Horizontal strip of live-style numeric statistics."""
    st.markdown("<br>", unsafe_allow_html=True)
    cols = st.columns(len(QUICK_STATS))
    for col, stat in zip(cols, QUICK_STATS):
        with col:
            st.metric(
                label=stat["label"],
                value=stat["value"],
                delta=stat.get("delta"),
            )
    st.markdown("<br>", unsafe_allow_html=True)


def _feature_cards() -> None:
    """Grid of feature highlight cards."""
    st.markdown(
        "<h2 style='font-size:1.5rem;font-weight:700;margin-bottom:1rem;'>What ModelLens AI Does</h2>",
        unsafe_allow_html=True,
    )

    cols = st.columns(3)
    for idx, feat in enumerate(FEATURE_HIGHLIGHTS):
        with cols[idx % 3]:
            st.markdown(
                f"""
                <div class="glass-card" style="margin-bottom:1rem;min-height:170px;">
                    <div style="font-size:2rem;margin-bottom:0.5rem;">{feat['icon']}</div>
                    <h3 style="font-size:1rem;font-weight:700;margin:0 0 0.4rem;color:var(--clr-primary-lt);">
                        {feat['title']}
                    </h3>
                    <p style="color:var(--clr-muted);font-size:0.875rem;line-height:1.6;margin:0;">
                        {feat['description']}
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )


def _benchmark_sources_section() -> None:
    """Cards listing integrated benchmark sources."""
    st.markdown(
        "<h2 style='font-size:1.5rem;font-weight:700;margin:2rem 0 1rem;'>Benchmark Sources</h2>",
        unsafe_allow_html=True,
    )

    cols = st.columns(len(BENCHMARK_SOURCES))
    for col, source in zip(cols, BENCHMARK_SOURCES):
        with col:
            st.markdown(
                f"""
                <div class="glass-card" style="text-align:center;padding:1.2rem;">
                    <div style="font-size:1.8rem;">{source['icon']}</div>
                    <div style="font-weight:600;font-size:0.9rem;margin:0.4rem 0 0.2rem;
                                color:var(--clr-accent-lt);">
                        {source['name']}
                    </div>
                    <div style="color:var(--clr-muted);font-size:0.78rem;">
                        {source['description']}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def _quickstart_guide() -> None:
    """Numbered quick-start steps."""
    st.markdown(
        "<h2 style='font-size:1.5rem;font-weight:700;margin:2.5rem 0 1rem;'>Get Started in 3 Steps</h2>",
        unsafe_allow_html=True,
    )

    steps = [
        ("1", "Describe Your Task", "Tell ModelLens what you need — coding assistant, summarisation, RAG, multi-lingual support, and more."),
        ("2", "Set Constraints", "Choose your model-size budget, licence preference, and performance priorities."),
        ("3", "Get Recommendations", "Receive AI-scored model rankings with benchmark breakdowns and comparison charts."),
    ]

    cols = st.columns(3)
    for col, (num, title, desc) in zip(cols, steps):
        with col:
            st.markdown(
                f"""
                <div style="
                    background:var(--clr-surface);
                    border:1px solid var(--clr-border);
                    border-radius:var(--radius-lg);
                    padding:1.5rem;
                    position:relative;
                    overflow:hidden;
                ">
                    <div style="
                        position:absolute;top:-10px;right:10px;
                        font-size:5rem;font-weight:900;
                        color:rgba(124,58,237,0.08);
                        line-height:1;pointer-events:none;
                    ">{num}</div>
                    <div style="
                        width:36px;height:36px;
                        background:linear-gradient(135deg,var(--clr-primary),#5B21B6);
                        border-radius:50%;
                        display:flex;align-items:center;justify-content:center;
                        font-weight:700;font-size:1rem;
                        margin-bottom:0.75rem;
                    ">{num}</div>
                    <h3 style="font-size:1rem;font-weight:700;margin:0 0 0.4rem;">{title}</h3>
                    <p style="color:var(--clr-muted);font-size:0.85rem;margin:0;line-height:1.6;">{desc}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )


def _footer() -> None:
    """Minimal branded footer."""
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown(
        """
        <hr style="border-color:rgba(124,58,237,0.2);margin:0 0 1rem;">
        <div style="text-align:center;color:var(--clr-muted);font-size:0.8rem;">
            <strong>ModelLens AI</strong> &middot; Open-source LLM intelligence platform &middot;
            Built with Streamlit, Pandas &amp; Plotly
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def render_homepage() -> None:
    """
    Render the complete homepage.

    Also handles routing: if the user navigated to another page via the
    sidebar or CTA buttons, delegate to that page's render function.
    """
    active = render_sidebar_nav()

    if active == "home":
        _hero_section()
        _stats_strip()
        _feature_cards()
        _benchmark_sources_section()
        _quickstart_guide()
        _footer()

    elif active == "leaderboard":
        st.info("**Leaderboard** — coming soon! Benchmark data fetching module is next.")

    elif active == "explorer":
        st.info("**Model Explorer** — coming soon! Filter and search thousands of open-source LLMs.")

    elif active == "compare":
        st.info("**Compare Models** — coming soon! Side-by-side benchmark comparison with Plotly charts.")

    elif active == "recommender":
        from src.ui.recommender import render_recommender
        render_recommender()

    elif active == "visualizations":
        st.info("**Visualizations** — coming soon! Rich interactive benchmark dashboards.")
