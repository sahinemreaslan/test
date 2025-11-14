#!/bin/bash
# Start Live Trading Dashboard

echo "🚀 Starting Live Trading Dashboard..."
echo ""

# Check if streamlit is installed
if ! command -v streamlit &> /dev/null; then
    echo "❌ Streamlit not found!"
    echo "Installing requirements..."
    pip install streamlit plotly
fi

# Create data directory if it doesn't exist
mkdir -p data

echo "✅ Dashboard starting..."
echo "📊 Open your browser and go to: http://localhost:8501"
echo ""
echo "Press Ctrl+C to stop"
echo ""

# Run streamlit
streamlit run dashboard.py --server.port 8501 --server.headless true
