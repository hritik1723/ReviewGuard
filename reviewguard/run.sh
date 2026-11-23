#!/usr/bin/env bash
# ReviewGuard Launch Script
echo "Starting ReviewGuard Intelligence Core..."
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.." || exit 1
exec python -m uvicorn reviewguard.app.main:app --host 0.0.0.0 --port 8000
