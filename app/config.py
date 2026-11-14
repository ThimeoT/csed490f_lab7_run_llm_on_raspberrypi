import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Settings:
    # ---------- Project 1 ----------
    ollama_url: str = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")
    model: str = os.getenv("CHAT_MODEL", "llama3.2")
    keep_alive: str = os.getenv("KEEP_ALIVE", "30m")
    warmup: bool = os.getenv("WARMUP", "1") == "1"
    num_predict: int = int(os.getenv("NUM_PREDICT", "512"))
    max_concurrent: int = int(os.getenv("MAX_CONCURRENT", "1"))
    max_queue: int = int(os.getenv("MAX_QUEUE", "4"))
    max_history_messages: int = int(os.getenv("MAX_HISTORY", "12"))
    session_ttl_s: int = int(os.getenv("SESSION_TTL_S", "1800"))
    system_prompt: str = os.getenv(
        "SYSTEM_PROMPT",
        "You are a helpful assistant running locally on a Raspberry Pi. Answer concisely.",
    )

    # ---------- Project 2 : RAG ----------
    embed_model: str = os.getenv("EMBED_MODEL", "nomic-embed-text")
    # The prefixes are specific to nomic-embed-text; set to "" for other models (e.g., bge-m3)
    embed_query_prefix: str = os.getenv("EMBED_QUERY_PREFIX", "search_query: ")
    embed_doc_prefix: str = os.getenv("EMBED_DOC_PREFIX", "search_document: ")
    embed_batch: int = int(os.getenv("EMBED_BATCH", "16"))
    kb_dir: str = os.getenv("KB_DIR", str(ROOT / "knowledge_base"))
    index_dir: str = os.getenv("INDEX_DIR", str(ROOT / "data" / "index"))
    chunk_chars: int = int(os.getenv("CHUNK_CHARS", "800"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "120"))
    top_k: int = int(os.getenv("RAG_TOP_K", "3"))
    min_score: float = float(os.getenv("RAG_MIN_SCORE", "0.45"))  # to calibrate via /api/rag/search
    rag_system_prompt: str = os.getenv(
        "RAG_SYSTEM_PROMPT",
        "You are a helpful assistant running locally on a Raspberry Pi. "
        "Answer from the provided context only. Be concise.",
    )
    rag_no_context_reply: str = os.getenv(
        "RAG_NO_CONTEXT_REPLY",
        "I could not find relevant information in the knowledge base to answer this question.",
    )


settings = Settings()