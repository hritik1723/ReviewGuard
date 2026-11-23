#!/usr/bin/env bash
# ReviewGuard Launch Script
# Starts the FastAPI uvicorn server on http://localhost:8000

echo "=================================================="
echo " ReviewGuard: Fake Product Review Detection"
echo " Classical ML Intelligence Core"
echo "=================================================="

# Ensure NLTK data is ready
python -c "import nltk; nltk.download('vader_lexicon', quiet=True); nltk.download('stopwords', quiet=True); nltk.download('wordnet', quiet=True)"

echo "Starting server on http://localhost:8000 ..."
python -m uvicorn reviewguard.app.main:app --host 0.0.0.0 --port 8000
