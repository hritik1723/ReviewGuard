"""Windows-friendly launcher for the ReviewGuard web application."""

from pathlib import Path
import sys

import uvicorn


PROJECT_PARENT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_PARENT))


if __name__ == "__main__":
    uvicorn.run(
        "reviewguard.app.main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
    )
