"""
src/ui/layout.py
----------------
Global Streamlit page configuration and shared layout helpers.
"""

import streamlit as st

# ---------------------------------------------------------------------------
# Inline CSS injected once at startup
# ---------------------------------------------------------------------------
_GLOBAL_CSS = """
<style>
/* ── Google Font ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── CSS custom properties ── */
:root {
    --clr-bg:          #0F0F1A;
    --clr-surface:     #1A1A2E;
    --clr-surface-2:   #16213E;
    --clr-border:      rgba(124,58,237,0.25);
    --clr-primary:     #7C3AED;
    --clr-primary-lt:  #A78BFA;
    --clr-accent:      #06B6D4;
    --clr-accent-lt:   #67E8F9;
    --clr-success:     #10B981;
    --clr-warning:     #F59E0B;
    --clr-danger:      #EF4444;
    --clr-text:        #E2E8F0;
    --clr-muted:       #94A3B8;
    --radius-lg:       16px;
    --radius-md:       10px;
    --radius-sm:       6px;
    --shadow-glow:     0 0 40px rgba(124,58,237,0.15);
    --transition:      0.25s cubic-bezier(0.4,0,0.2,1);
}

/* ── Global resets ── */
html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--clr-bg) !important;
    font-family: 'Inter', sans-serif !important;
    color: var(--clr-text) !important;
}

[data-testid="stSidebar"] {
    background-color: var(--clr-surface) !important;
    border-right: 1px solid var(--clr-border) !important;
}

[data-testid="stHeader"] {
    background-color: var(--clr-bg) !important;
    border-bottom: 1px solid var(--clr-border) !important;
}

/* ── Metric cards ── */
[data-testid="metric-container"] {
    background: var(--clr-surface) !important;
    border: 1px solid var(--clr-border) !important;
    border-radius: var(--radius-md) !important;
    padding: 1rem 1.25rem !important;
    transition: transform var(--transition), box-shadow var(--transition) !important;
}
[data-testid="metric-container"]:hover {
    transform: translateY(-2px) !important;
    box-shadow: var(--shadow-glow) !important;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, var(--clr-primary), #5B21B6) !important;
    color: #fff !important;
    border: none !important;
    border-radius: var(--radius-sm) !important;
    font-weight: 600 !important;
    letter-spacing: 0.02em !important;
    transition: transform var(--transition), box-shadow var(--transition) !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 20px rgba(124,58,237,0.4) !important;
}

/* ── Selectboxes & text inputs ── */
[data-baseweb="select"], [data-baseweb="input"] {
    background: var(--clr-surface-2) !important;
    border-color: var(--clr-border) !important;
    border-radius: var(--radius-sm) !important;
}

/* ── Tabs ── */
[data-testid="stTabs"] button {
    font-weight: 500 !important;
    color: var(--clr-muted) !important;
    border-bottom: 2px solid transparent !important;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    color: var(--clr-primary-lt) !important;
    border-bottom-color: var(--clr-primary) !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar        { width: 6px; height: 6px; }
::-webkit-scrollbar-track  { background: var(--clr-bg); }
::-webkit-scrollbar-thumb  { background: var(--clr-primary); border-radius: 99px; }

/* ── Utility classes ── */
.gradient-text {
    background: linear-gradient(135deg, var(--clr-primary-lt), var(--clr-accent-lt));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.glass-card {
    background: rgba(26,26,46,0.7) !important;
    border: 1px solid var(--clr-border) !important;
    border-radius: var(--radius-lg) !important;
    backdrop-filter: blur(12px) !important;
    padding: 1.5rem !important;
}
.pulse-dot {
    display: inline-block;
    width: 8px; height: 8px;
    background: var(--clr-success);
    border-radius: 50%;
    animation: pulse 2s ease-in-out infinite;
    margin-right: 6px;
}
@keyframes pulse {
    0%,100% { opacity: 1; transform: scale(1); }
    50%      { opacity: 0.5; transform: scale(1.4); }
}
.tag {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 99px;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}
.tag-primary { background: rgba(124,58,237,0.2); color: var(--clr-primary-lt); border: 1px solid rgba(124,58,237,0.4); }
.tag-accent  { background: rgba(6,182,212,0.15); color: var(--clr-accent-lt);  border: 1px solid rgba(6,182,212,0.3); }
.tag-success { background: rgba(16,185,129,0.15); color: #6EE7B7;              border: 1px solid rgba(16,185,129,0.3); }
</style>
"""


def render_page_config() -> None:
    """Set Streamlit page metadata and inject global CSS."""
    st.set_page_config(
        page_title="ModelLens AI — LLM Benchmarking & Discovery",
        page_icon="ML",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            "Get Help": "https://github.com/your-org/modellens-ai",
            "Report a bug": "https://github.com/your-org/modellens-ai/issues",
            "About": "# ModelLens AI\nIntelligent LLM discovery and benchmarking platform.",
        },
    )
    st.markdown(_GLOBAL_CSS, unsafe_allow_html=True)


def render_sidebar_nav() -> str:
    """
    Render the global sidebar navigation.

    Returns
    -------
    str
        The currently selected page key.
    """
    with st.sidebar:
        st.markdown(
            """
            <div style="text-align:center;padding:1rem 0 1.5rem;">
                <div style="
                    display:inline-block;
                    background:linear-gradient(135deg,#7C3AED,#06B6D4);
                    border-radius:10px;
                    width:44px;height:44px;
                    line-height:44px;
                    font-size:1.1rem;
                    font-weight:900;
                    color:#fff;
                    letter-spacing:-0.03em;
                    margin-bottom:0.5rem;
                ">ML</div>
                <h2 style="margin:0.25rem 0 0;font-size:1.3rem;font-weight:700;">
                    ModelLens <span class="gradient-text">AI</span>
                </h2>
                <p style="color:var(--clr-muted);font-size:0.78rem;margin:0.2rem 0 0;">
                    LLM Intelligence Platform
                </p>
            </div>
            <hr style="border-color:var(--clr-border);margin-bottom:1rem;">
            """,
            unsafe_allow_html=True,
        )

        pages = {
            "Home": "home",
            "Leaderboard": "leaderboard",
            "Model Explorer": "explorer",
            "Compare Models": "compare",
            "Recommender": "recommender",
            "Visualizations": "visualizations",
        }

        if "active_page" not in st.session_state:
            st.session_state.active_page = "home"

        for label, key in pages.items():
            is_active = st.session_state.active_page == key
            btn_style = (
                "background:rgba(124,58,237,0.2);border-left:3px solid #7C3AED;"
                if is_active
                else "background:transparent;border-left:3px solid transparent;"
            )
            if st.button(
                label,
                key=f"nav_{key}",
                use_container_width=True,
            ):
                st.session_state.active_page = key
                st.rerun()

        st.markdown("<hr style='border-color:var(--clr-border);margin-top:auto;'>", unsafe_allow_html=True)
        st.markdown(
            "<p style='color:var(--clr-muted);font-size:0.72rem;text-align:center;'>"
            "v0.1.0 &middot; Built with Streamlit</p>",
            unsafe_allow_html=True,
        )

    return st.session_state.get("active_page", "home")
