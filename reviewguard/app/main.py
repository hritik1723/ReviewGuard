"""FastAPI Main Application entry point for ReviewGuard."""

from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from reviewguard.app.routes import (
    association,
    clustering,
    datasets,
    ensemble,
    evaluation,
    features,
    hmm,
    overview,
    predict,
    train,
)

app = FastAPI(
    title="ReviewGuard Intelligence Core API",
    description="Classical ML End-to-End Fake Product Review Detection System",
    version="2.4.0",
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API Routers
app.include_router(overview.router, prefix="/api", tags=["Overview"])
app.include_router(datasets.router, prefix="/api", tags=["Datasets"])
app.include_router(features.router, prefix="/api", tags=["Features"])
app.include_router(train.router, prefix="/api", tags=["Training"])
app.include_router(evaluation.router, prefix="/api", tags=["Evaluation"])
app.include_router(ensemble.router, prefix="/api", tags=["Ensembles"])
app.include_router(clustering.router, prefix="/api", tags=["Clustering"])
app.include_router(association.router, prefix="/api", tags=["Association Rules"])
app.include_router(hmm.router, prefix="/api", tags=["Sequence Analysis (HMM)"])
app.include_router(predict.router, prefix="/api", tags=["Live Inference"])


@app.get("/api/health")
def healthcheck():
    return {"status": "healthy", "service": "ReviewGuard"}


# Mount frontend static files
frontend_dir = Path(__file__).resolve().parent.parent / "frontend"
if not frontend_dir.exists():
    frontend_dir = Path.cwd() / "frontend"

if frontend_dir.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")

    @app.get("/")
    async def serve_index():
        index_file = frontend_dir / "index.html"
        if index_file.exists():
            return FileResponse(index_file)
        return {"message": "ReviewGuard API is running. Frontend index.html not found."}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("reviewguard.app.main:app", host="0.0.0.0", port=8000, reload=True)
