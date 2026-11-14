import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .config import settings
from .llm import LLMService
from .routes import router
from .sessions import SessionStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.llm = LLMService(settings)
    app.state.sessions = SessionStore(settings.session_ttl_s, settings.max_history_messages)
    # Warmup as a background task : the server answers direclty
    warm = asyncio.create_task(app.state.llm.warmup()) if settings.warmup else None
    yield
    if warm:
        warm.cancel()


app = FastAPI(title="Pi LLM Chat Server", version="1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)
app.include_router(router)
# Build at the end : API routes and /docs are the priority
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="ui")