import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from src.main import run_pipeline
from src.models.recommender import list_use_cases

def render_recommender() -> None:
    """Render the Recommender page."""
    # 1. Project header and short description
    st.markdown("<h1 style='font-size:2.5rem;font-weight:800;margin-bottom:0.5rem;'>Model Recommender</h1>", unsafe_allow_html=True)
    st.markdown(
        "<p style='color:var(--clr-muted);font-size:1.1rem;margin-bottom:2rem;'>"
        "Discover the perfect open-source LLM tailored to your specific use case, leveraging live benchmarking data."
        "</p>",
        unsafe_allow_html=True
    )

    # 2. User requirements
    with st.form("requirements_form"):
        st.markdown("<h3 style='margin-top:0;'>Your Requirements</h3>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            use_cases = list_use_cases()
            selected_use_case = st.selectbox("Use Case", options=use_cases, index=0)
            
            size_options = {
                "Any": (0.0, 10000.0),
                "Tiny (< 3B)": (0.0, 3.0),
                "Small (3–7B)": (3.0, 7.0),
                "Medium (7–14B)": (7.0, 14.0),
                "Large (14–35B)": (14.0, 35.0),
                "XL/XXL (> 35B)": (35.0, 10000.0)
            }
            size_pref = st.selectbox("Model Size Preference", options=list(size_options.keys()), index=0)
            
        with col2:
            languages_input = st.text_input("Required Languages (e.g., en, fr, zh)", value="en")
            min_score = st.slider(
                "Performance Priority (Minimum Average Score)", 
                min_value=0.0, max_value=100.0, value=0.0, step=5.0,
                help="Only consider models with an average benchmark score above this value."
            )
            
        # 3. "Find Models" button
        st.markdown("<br>", unsafe_allow_html=True)
        submitted = st.form_submit_button("Find Models", use_container_width=True)

    if submitted:
        with st.spinner("Analyzing models and crunching benchmarks..."):
            min_b, max_b = size_options[size_pref]
            langs = [lang.strip() for lang in languages_input.split(",") if lang.strip()]
            
            result = run_pipeline(
                use_case=selected_use_case,
                min_params_b=min_b,
                max_params_b=max_b,
                required_languages=langs,
                min_average_score=min_score if min_score > 0 else None
            )
            
            if result["status"] == "error":
                st.error(result["message"])
                return
                
            st.success(result["message"])
            
            recs = result["recommendations"]
            if not recs:
                st.warning("No recommendations returned.")
                return
                
            st.markdown("<hr style='border-color:var(--clr-border);margin:2rem 0;'>", unsafe_allow_html=True)
            
            # 4. Top 5 recommendations & 5. Explanations
            st.markdown("### Top Recommendations")
            for rec in recs:
                with st.expander(f"Rank {rec['rank']}: {rec['model_name']} (Score: {rec['suitability_score']:.1f})", expanded=(rec['rank'] == 1)):
                    st.markdown(f"**Why this model?** {rec['why_this_model']}")
                    c1, c2 = st.columns(2)
                    with c1:
                        st.markdown("**Strengths:**")
                        for s in rec['strengths']:
                            st.markdown(f"- {s}")
                    with c2:
                        st.markdown("**Trade-offs:**")
                        for t in rec['tradeoffs']:
                            st.markdown(f"- {t}")

            st.markdown("<br>", unsafe_allow_html=True)
            
            # 6. Model comparison table
            st.markdown("### Model Comparison")
            
            table_data = []
            for rec in recs:
                row_data = {"Model": rec["model_name"], "Overall Score": round(rec["suitability_score"], 2)}
                # Add benchmark columns, round them up
                for k, v in rec["benchmark_data"].items():
                    row_data[k.upper()] = round(v, 2) if pd.notna(v) else None
                table_data.append(row_data)
                
            df_table = pd.DataFrame(table_data)
            st.dataframe(df_table, use_container_width=True, hide_index=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # 7. Benchmark comparison charts
            st.markdown("### Benchmark Performance")
            bm_cols = result["benchmark_sources"]
            if bm_cols:
                # Grouped Bar Chart
                bm_upper = [b.upper() for b in bm_cols]
                df_melted = pd.melt(df_table, id_vars=["Model"], value_vars=bm_upper, var_name="Benchmark", value_name="Score")
                fig = px.bar(
                    df_melted, x="Benchmark", y="Score", color="Model", barmode="group",
                    title="Raw Benchmark Scores by Model", template="plotly_white",
                    color_discrete_sequence=px.colors.qualitative.Pastel
                )
                fig.update_layout(
                    legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="right", x=1),
                    margin=dict(l=20, r=20, t=60, b=20)
                )
                st.plotly_chart(fig, use_container_width=True)
                
            # 8. Data source and last-updated information
            st.markdown("<hr style='border-color:var(--clr-border);margin:2rem 0 1rem;'>", unsafe_allow_html=True)
            st.caption(f"**Data Sources:** Hugging Face Open LLM Leaderboard | **Benchmarks Evaluated:** {', '.join(bm_upper)}")
            from datetime import datetime
            st.caption(f"**Last Updated:** {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}")
