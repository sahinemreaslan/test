#!/bin/bash
# Start Chart Dashboard

echo "📈 Starting Chart Dashboard..."
echo ""

# Check if streamlit is installed
if ! command -v streamlit &> /dev/null; then
    echo "❌ Streamlit not found!"
    echo "Installing requirements..."
    pip install streamlit plotly
fi

# Create data directory if it doesn't exist
mkdir -p data

echo "✅ Chart Dashboard starting..."
echo "📊 Open your browser and go to: http://localhost:8502"
echo ""
echo "Press Ctrl+C to stop"
echo ""

# Run streamlit on different port
streamlit run chart_dashboard.py --server.port 8502 --server.headless true
