#!/usr/bin/env python3
# coding: utf-8

"""
Streamlit-based viewer for citation failure analysis pipeline results.
"""

import streamlit as st
import pandas as pd
import json
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from typing import Dict, List, Any, Optional
import altair as alt

st.set_page_config(
    page_title="Citation Failure Analysis Viewer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data
def load_data():
    """Load all data files from the pipeline."""
    data = {}
    
    # Load extracted data
    with open("data/extracted/unsupported_claims_extracted.json", "r") as f:
        data["unsupported_claims"] = json.load(f)
    
    with open("data/extracted/non_supporting_citations_extracted.json", "r") as f:
        data["non_supporting_citations"] = json.load(f)
    
    # Load classified data
    with open("data/classified/classified_unsupported_claims.json", "r") as f:
        data["classified_claims"] = json.load(f)
    
    with open("data/classified/classified_non_supporting_citations.json", "r") as f:
        data["classified_citations"] = json.load(f)
    
    # Load reports
    with open("reports/failure_mode_analysis_report.json", "r") as f:
        data["failure_mode_report"] = json.load(f)
    
    with open("reports/citation_failure_mode_analysis_report.json", "r") as f:
        data["citation_failure_report"] = json.load(f)
    
    # Load CSV summaries
    data["claims_summary"] = pd.read_csv("reports/classified_claims_summary.csv")
    data["citations_summary"] = pd.read_csv("reports/classified_citations_summary.csv")
    
    return data

def render_overview_dashboard(data: Dict):
    """Render the overview dashboard page."""
    st.title("📊 Citation Failure Analysis Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    # Calculate overall statistics
    systems = list(data["unsupported_claims"]["systems"].keys())
    total_claims = sum(system["total_claims"] for system in data["unsupported_claims"]["systems"].values())
    total_unsupported = sum(system["unsupported_claims_count"] for system in data["unsupported_claims"]["systems"].values())
    total_citations = sum(system["non_supporting_citations_count"] for system in data["non_supporting_citations"]["systems"].values())
    
    with col1:
        st.metric("Systems Analyzed", len(systems))
        st.caption(", ".join(systems))
    
    with col2:
        st.metric("Total Claims", f"{total_claims:,}")
        
    with col3:
        st.metric("Unsupported Claims", f"{total_unsupported:,}")
        st.caption(f"{(total_unsupported/total_claims*100):.1f}% of total")
    
    with col4:
        st.metric("Non-Supporting Citations", f"{total_citations:,}")
    
    st.divider()
    
    # Failure Mode Distribution for Claims
    st.subheader("📈 Claim Failure Mode Distribution")
    
    if "failure_mode_distribution" in data["failure_mode_report"]:
        col1, col2 = st.columns(2)
        
        with col1:
            # Bar chart
            failure_modes = data["failure_mode_report"]["failure_mode_distribution"]
            modes_df = pd.DataFrame([
                {"Mode": v["name"], "Count": v["count"], "Percentage": v["percentage"]}
                for k, v in failure_modes.items()
            ])
            modes_df = modes_df.sort_values("Count", ascending=True)
            
            fig_bar = px.bar(
                modes_df,
                x="Count",
                y="Mode",
                orientation='h',
                title="Failure Modes by Count",
                text="Count",
                color="Count",
                color_continuous_scale="Viridis"
            )
            fig_bar.update_traces(textposition='outside')
            st.plotly_chart(fig_bar, use_container_width=True)
        
        with col2:
            # Pie chart
            fig_pie = px.pie(
                modes_df,
                values="Count",
                names="Mode",
                title="Failure Mode Distribution",
                hole=0.3
            )
            fig_pie.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_pie, use_container_width=True)
    
    st.divider()
    
    # Citation Failure Categories
    st.subheader("📑 Citation Failure Categories")
    
    if "category_statistics" in data["citation_failure_report"]:
        categories_df = pd.DataFrame(data["citation_failure_report"]["category_statistics"])
        categories_df = categories_df.sort_values("count", ascending=False)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Horizontal bar chart
            fig_cat = px.bar(
                categories_df,
                x="count",
                y="name",
                orientation='h',
                title="Citation Failure Categories",
                text="count",
                hover_data=["description"],
                color="percentage",
                color_continuous_scale="Reds"
            )
            fig_cat.update_traces(textposition='outside')
            st.plotly_chart(fig_cat, use_container_width=True)
        
        with col2:
            # Donut chart
            fig_donut = px.pie(
                categories_df,
                values="count",
                names="name",
                title="Category Distribution",
                hole=0.4
            )
            st.plotly_chart(fig_donut, use_container_width=True)

def render_system_analysis(data: Dict):
    """Render system-specific analysis page."""
    st.title("🔍 System-Specific Analysis")
    
    # System selector
    systems = list(data["unsupported_claims"]["systems"].keys())
    selected_system = st.selectbox("Select System", systems)
    
    if selected_system:
        st.subheader(f"Analysis for: {selected_system}")
        
        col1, col2, col3 = st.columns(3)
        
        system_data = data["unsupported_claims"]["systems"][selected_system]
        
        with col1:
            st.metric("Total Rows", f"{system_data.get('total_rows', 'N/A'):,}")
            st.metric("Total Claims", f"{system_data['total_claims']:,}")
        
        with col2:
            st.metric("Unsupported Claims", f"{system_data['unsupported_claims_count']:,}")
            if system_data['total_claims'] > 0:
                pct = (system_data['unsupported_claims_count'] / system_data['total_claims'] * 100)
                st.caption(f"{pct:.1f}% unsupported")
        
        with col3:
            if selected_system in data["non_supporting_citations"]["systems"]:
                citations_count = data["non_supporting_citations"]["systems"][selected_system]["non_supporting_citations_count"]
                st.metric("Non-Supporting Citations", f"{citations_count:,}")
        
        st.divider()
        
        # Failure mode breakdown for this system
        if "system_analysis" in data["failure_mode_report"] and selected_system in data["failure_mode_report"]["system_analysis"]:
            st.subheader("Failure Mode Breakdown")
            
            system_modes = data["failure_mode_report"]["system_analysis"][selected_system]
            if "by_mode" in system_modes:
                modes_data = []
                for mode, count in system_modes["by_mode"].items():
                    mode_name = data["failure_mode_report"]["failure_mode_distribution"].get(mode, {}).get("name", mode)
                    modes_data.append({"Mode": mode_name, "Count": count})
                
                modes_df = pd.DataFrame(modes_data)
                modes_df = modes_df.sort_values("Count", ascending=False)
                
                fig = px.bar(
                    modes_df,
                    x="Mode",
                    y="Count",
                    title=f"Failure Modes for {selected_system}",
                    text="Count",
                    color="Count",
                    color_continuous_scale="Blues"
                )
                fig.update_traces(textposition='outside')
                st.plotly_chart(fig, use_container_width=True)
        
        st.divider()
        
        # Citation failure categories for this system
        if selected_system in data["citation_failure_report"]["system_statistics"]:
            st.subheader("Citation Failure Categories")
            
            system_citations = data["citation_failure_report"]["system_statistics"][selected_system]
            if "categories" in system_citations:
                cat_data = []
                for cat, count in system_citations["categories"].items():
                    # Find the full category name
                    cat_name = cat
                    for cat_info in data["citation_failure_report"]["category_statistics"]:
                        if cat_info["code"] == cat:
                            cat_name = cat_info["name"]
                            break
                    cat_data.append({"Category": cat_name, "Count": count})
                
                cat_df = pd.DataFrame(cat_data)
                cat_df = cat_df.sort_values("Count", ascending=False)
                
                fig = px.bar(
                    cat_df,
                    x="Category",
                    y="Count",
                    title=f"Citation Failure Categories for {selected_system}",
                    text="Count",
                    color="Count",
                    color_continuous_scale="Oranges"
                )
                fig.update_traces(textposition='outside')
                fig.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)

def render_unsupported_claims_explorer(data: Dict):
    """Render the unsupported claims explorer page."""
    st.title("🔎 Unsupported Claims Explorer")
    
    # Create filters
    col1, col2, col3 = st.columns(3)
    
    # Get unique values for filters
    systems = list(data["unsupported_claims"]["systems"].keys())
    failure_modes = []
    if "failure_mode_distribution" in data["failure_mode_report"]:
        failure_modes = [v["name"] for v in data["failure_mode_report"]["failure_mode_distribution"].values()]
    
    with col1:
        selected_system = st.selectbox("Filter by System", ["All"] + systems)
    
    with col2:
        selected_mode = st.selectbox("Filter by Failure Mode", ["All"] + failure_modes)
    
    with col3:
        search_text = st.text_input("Search in claim text", "")
    
    # Prepare claims data
    claims_list = []
    if "claims" in data["classified_claims"]:
        claims_list = data["classified_claims"]["claims"]
    
    # Apply filters
    filtered_claims = claims_list
    
    if selected_system != "All":
        filtered_claims = [c for c in filtered_claims if c.get("system") == selected_system]
    
    if selected_mode != "All":
        # Map display name back to code
        mode_code = None
        for code, info in data["failure_mode_report"]["failure_mode_distribution"].items():
            if info["name"] == selected_mode:
                mode_code = code
                break
        if mode_code:
            filtered_claims = [c for c in filtered_claims if c.get("failure_mode") == mode_code]
    
    if search_text:
        filtered_claims = [c for c in filtered_claims if search_text.lower() in c.get("claim_text", "").lower()]
    
    st.divider()
    st.subheader(f"Found {len(filtered_claims)} claims")
    
    # Display claims
    for idx, claim in enumerate(filtered_claims):
        with st.expander(f"Claim {idx+1}: {claim.get('claim_text', '')[:100]}..."):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown("**Claim Text:**")
                st.write(claim.get("claim_text", ""))
                
                st.markdown("**Question Context:**")
                st.info(claim.get("question", "N/A"))
                
                st.markdown("**Classification Reasoning:**")
                st.write(claim.get("classification_reasoning", "N/A"))
            
            with col2:
                st.markdown("**Metadata:**")
                st.write(f"**System:** {claim.get('system', 'N/A')}")
                
                # Get failure mode name
                mode_code = claim.get("failure_mode", "N/A")
                mode_name = mode_code
                if mode_code in data["failure_mode_report"]["failure_mode_distribution"]:
                    mode_name = data["failure_mode_report"]["failure_mode_distribution"][mode_code]["name"]
                st.write(f"**Failure Mode:** {mode_name}")
                
                supporting_refs = claim.get("supporting_refs", [])
                non_supporting_refs = claim.get("non_supporting_refs", [])
                st.write(f"**Supporting Refs:** {len(supporting_refs)}")
                st.write(f"**Non-Supporting Refs:** {len(non_supporting_refs)}")
            
            # Show cited snippets if available
            if "cited_snippets" in claim and claim["cited_snippets"]:
                st.markdown("**Cited Snippets:**")
                for ref, snippet in claim["cited_snippets"].items():
                    if snippet and snippet != "No snippet found.":
                        with st.container():
                            st.markdown(f"**{ref}:**")
                            st.text(snippet[:500] + "..." if len(snippet) > 500 else snippet)
            
            # Show references
            if non_supporting_refs:
                st.markdown("**Non-Supporting References:**")
                for ref in non_supporting_refs:
                    st.write(f"• {ref}")

def render_definitions(data: Dict):
    """Render the definitions page for failure mode categories."""
    st.title("📖 Category Definitions")
    
    # Claims Failure Modes
    st.header("🔴 Unsupported Claims - Failure Modes")
    
    if "failure_modes_used" in data["classified_claims"]:
        for mode in data["classified_claims"]["failure_modes_used"]:
            with st.expander(f"**{mode['name']}** ({mode['code']})"):
                st.markdown(f"**Description:** {mode['description']}")
                
                if "characteristics" in mode:
                    st.markdown("**Characteristics:**")
                    for char in mode["characteristics"]:
                        st.write(f"• {char}")
                
                if "estimated_percentage" in mode:
                    st.metric("Estimated Percentage", f"{mode['estimated_percentage']}%")
    
    st.divider()
    
    # Citation Failure Categories
    st.header("🔵 Non-Supporting Citations - Categories")
    
    if "category_statistics" in data["citation_failure_report"]:
        for category in data["citation_failure_report"]["category_statistics"]:
            with st.expander(f"**{category['name']}** ({category['code']})"):
                st.markdown(f"**Description:** {category['description']}")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Count", category['count'])
                with col2:
                    st.metric("Percentage", f"{category['percentage']:.1f}%")
    
    st.divider()
    
    # Summary Statistics
    st.header("📊 Summary Statistics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Claims Analysis")
        if "total_claims" in data["classified_claims"]:
            st.metric("Total Claims Classified", data["classified_claims"]["total_claims"])
        if "is_sample" in data["classified_claims"] and data["classified_claims"]["is_sample"]:
            st.metric("Sample Size", data["classified_claims"]["sample_size"])
            st.metric("Total Available", data["classified_claims"]["total_available_claims"])
    
    with col2:
        st.subheader("Citations Analysis")
        if "total_citations_analyzed" in data["citation_failure_report"]:
            st.metric("Total Citations Analyzed", data["citation_failure_report"]["total_citations_analyzed"])

def render_citations_explorer(data: Dict):
    """Render the non-supporting citations explorer page."""
    st.title("📚 Non-Supporting Citations Explorer")
    
    # Create filters
    col1, col2, col3 = st.columns(3)
    
    # Get unique values for filters
    systems = list(data["non_supporting_citations"]["systems"].keys())
    categories = []
    if "category_statistics" in data["citation_failure_report"]:
        categories = [cat["name"] for cat in data["citation_failure_report"]["category_statistics"]]
    
    with col1:
        selected_system = st.selectbox("Filter by System", ["All"] + systems)
    
    with col2:
        selected_category = st.selectbox("Filter by Category", ["All"] + categories)
    
    with col3:
        search_text = st.text_input("Search in claim or citation", "")
    
    # Prepare citations data - it's an array
    citations_list = []
    if isinstance(data["classified_citations"], list):
        citations_list = data["classified_citations"]
    
    # Apply filters
    filtered_citations = citations_list
    
    if selected_system != "All":
        filtered_citations = [c for c in filtered_citations if c.get("system") == selected_system]
    
    if selected_category != "All":
        # Map display name back to code
        cat_code = None
        for cat in data["citation_failure_report"]["category_statistics"]:
            if cat["name"] == selected_category:
                cat_code = cat["code"]
                break
        if cat_code:
            filtered_citations = [c for c in filtered_citations if c.get("assigned_category") == cat_code]
    
    if search_text:
        filtered_citations = [
            c for c in filtered_citations 
            if search_text.lower() in c.get("claim_text", "").lower() 
            or search_text.lower() in c.get("citation_id", "").lower()
            or search_text.lower() in c.get("citation_snippet", "").lower()
        ]
    
    st.divider()
    st.subheader(f"Found {len(filtered_citations)} non-supporting citations")
    
    # Display citations
    for idx, citation in enumerate(filtered_citations):
        with st.expander(f"Citation {idx+1}: {citation.get('citation_id', 'N/A')} - {citation.get('claim_text', '')[:80]}..."):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown("**Claim Text:**")
                st.write(citation.get("claim_text", ""))
                
                st.markdown("**Citation Snippet:**")
                snippet = citation.get("citation_snippet", "N/A")
                if snippet and len(snippet) > 1000:
                    snippet = snippet[:1000] + "..."
                st.info(snippet)
                
                st.markdown("**Classification Reasoning:**")
                st.write(citation.get("classification_reasoning", "N/A"))
            
            with col2:
                st.markdown("**Metadata:**")
                st.write(f"**System:** {citation.get('system', 'N/A')}")
                st.write(f"**Citation ID:** {citation.get('citation_id', 'N/A')}")
                
                # Get category name
                cat_code = citation.get("assigned_category", "N/A")
                cat_name = cat_code
                for cat in data["citation_failure_report"]["category_statistics"]:
                    if cat["code"] == cat_code:
                        cat_name = cat["name"]
                        break
                st.write(f"**Category:** {cat_name}")
                
                st.write(f"**Question:** {citation.get('question', 'N/A')}")
            

def main():
    """Main application function."""
    st.sidebar.title("📊 Citation Failure Analysis")
    st.sidebar.markdown("---")
    
    # Navigation
    page = st.sidebar.radio(
        "Navigation",
        ["Overview Dashboard", "System Analysis", "Unsupported Claims", "Non-Supporting Citations", "Category Definitions"]
    )
    
    # Load data
    try:
        with st.spinner("Loading data..."):
            data = load_data()
    except Exception as e:
        st.error(f"Failed to load data: {str(e)}")
        st.info("Please ensure all data files are present in the expected directories.")
        return
    
    # Render selected page
    if page == "Overview Dashboard":
        render_overview_dashboard(data)
    elif page == "System Analysis":
        render_system_analysis(data)
    elif page == "Unsupported Claims":
        render_unsupported_claims_explorer(data)
    elif page == "Non-Supporting Citations":
        render_citations_explorer(data)
    elif page == "Category Definitions":
        render_definitions(data)
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("### About")
    st.sidebar.info(
        "This viewer provides interactive exploration of citation failure analysis results "
        "from multiple AI research systems."
    )
    
    st.sidebar.markdown("### Data Sources")
    st.sidebar.caption(
        "• Unsupported claims data\n"
        "• Non-supporting citations\n"
        "• Classification reports\n"
        "• Failure mode analysis"
    )

if __name__ == "__main__":
    main()
