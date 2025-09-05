#!/bin/bash
# Launch the Citation Failure Analysis Viewer

echo "Starting Citation Failure Analysis Viewer..."
echo "The app will be available at http://localhost:8501"
echo "Press Ctrl+C to stop the application"
echo ""

uv run --with streamlit --with plotly --with altair streamlit run citation_viewer.py