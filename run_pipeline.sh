#!/bin/bash
set -e

echo "🧭 Research Navigator — Pipeline Runner"
echo "========================================"

echo ""
echo "Step 1/3: Processing with Spark..."
cd src && python3 spark_pipeline.py && cd ..

echo ""
echo "Step 2/3: Generating embeddings + FAISS index..."
cd src && python3 embeddings.py && cd ..

echo ""
echo "Step 3/3: Building citation graph..."
cd src && python3 graph.py && cd ..

echo ""
echo "✅ All done! Launching demo..."
streamlit run app/main.py