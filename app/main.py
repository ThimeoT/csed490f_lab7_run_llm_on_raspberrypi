import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .config import settings
from .llm import LLMService
from .rag.service import RagService
from .routes import router
from .sessions import SessionStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("main")

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"


async def _background_startup(app: FastAPI) -> None:
    """Background tasks: the server responds immediately, the Pi loads its models afterwards."""
    if settings.warmup:
        await app.state.llm.warmup()
    try:
        await app.state.rag.reindex() # only the new or modified documents are embedded
        log.info("RAG index ready: %s", app.state.rag.status())
    except Exception as exc:
        log.warning("RAG indexing skipped (ollama pull %s ?): %s", settings.embed_model, exc)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.llm = LLMService(settings)
    app.state.sessions = SessionStore(settings.session_ttl_s, settings.max_history_messages)
    app.state.rag = RagService(settings, app.state.llm.client)  # same Ollama client
    task = asyncio.create_task(_background_startup(app))
    yield
    task.cancel()


app = FastAPI(title="Pi LLM Chat + RAG Server", version="2.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)
app.include_router(router)
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="ui")