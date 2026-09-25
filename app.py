"""
ModelLens AI — Main Streamlit entry point.

Run with:
    streamlit run app.py
"""

import streamlit as st
from src.ui.layout import render_page_config
from src.ui.homepage import render_homepage


def main() -> None:
    """Bootstrap the Streamlit application."""
    render_page_config()
    render_homepage()


if __name__ == "__main__":
    main()
