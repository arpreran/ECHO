"""ECHO backend: FastAPI app hosting two independent services.

  /api/companion/*  ECHO Companion (external conversational AI provider)
  /api/nlp/*        ECHO NLP Lab (classical NLP only; never calls an AI API)

Run from backend/:  ..\\.venv\\Scripts\\uvicorn app.main:app --reload --port 8000
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from companion.providers import Providers
from companion.routes import router as companion_router
from companion.service import CompanionService
from database.db import Database
from nlp_lab.routes import router as nlp_router
from nlp_lab.service import NLPService

from .config import Settings, settings as default_settings

FRONTEND_DIST = Path(__file__).resolve().parents[2] / "frontend" / "dist"


def create_app(settings: Settings | None = None, providers: Providers | None = None,
               nlp: NLPService | None = None) -> FastAPI:
    settings = settings or default_settings

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.settings = settings
        app.state.db = Database(settings.database_path)
        app.state.nlp = nlp or NLPService()
        app.state.providers = providers or Providers(settings)
        # The Companion reuses the NLP Lab's classical LID only to tag message
        # language metadata. The NLP Lab never uses the Companion's provider.
        app.state.companion = CompanionService(app.state.db, app.state.providers, app.state.nlp.lid)
        yield
        app.state.db.conn.close()

    app = FastAPI(title="ECHO API", version="1.0.0", lifespan=lifespan)
    app.add_middleware(CORSMiddleware, allow_origins=list(settings.cors_origins), allow_credentials=False,
                       allow_methods=["GET", "POST", "PATCH", "DELETE"], allow_headers=["Content-Type"])

    @app.get("/api/health")
    def health():
        return {"status": "ok", "database_schema_version": app.state.db.schema_version(),
                "nlp_lab": app.state.nlp.status(), "companion": app.state.providers.status()}

    app.include_router(companion_router)
    app.include_router(nlp_router)

    # Single-server deployment (e.g. the hosted demo): serve the built React app from frontend/dist.
    # In development the Vite dev server serves the frontend instead and this block is skipped.
    if (FRONTEND_DIST / "index.html").is_file():
        app.mount("/assets", StaticFiles(directory=FRONTEND_DIST / "assets"), name="assets")

        @app.get("/{path:path}", include_in_schema=False)
        def spa(path: str):
            if path.startswith("api/"):
                raise HTTPException(404, "Not found")
            f = (FRONTEND_DIST / path).resolve()
            if path and f.is_file() and FRONTEND_DIST.resolve() in f.parents:
                return FileResponse(f)
            return FileResponse(FRONTEND_DIST / "index.html")   # client-side routes (/lab, /companion, ...)
    return app


app = create_app()
