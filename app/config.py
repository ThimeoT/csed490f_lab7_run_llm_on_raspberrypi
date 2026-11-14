import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    # OLLAMA_URL (not OLLAMA_HOST, already used by Ollama itself)
    ollama_url: str = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")
    model: str = os.getenv("CHAT_MODEL", "llama3.2")
    keep_alive: str = os.getenv("KEEP_ALIVE", "30m")   # keeps the module inside the RAM
    warmup: bool = os.getenv("WARMUP", "1") == "1"
    num_predict: int = int(os.getenv("NUM_PREDICT", "512"))  # max tokens generated
    max_concurrent: int = int(os.getenv("MAX_CONCURRENT", "1"))
    max_queue: int = int(os.getenv("MAX_QUEUE", "4"))         # maximum waiting before 429
    max_history_messages: int = int(os.getenv("MAX_HISTORY", "12"))
    session_ttl_s: int = int(os.getenv("SESSION_TTL_S", "1800"))
    system_prompt: str = os.getenv(
        "SYSTEM_PROMPT",
        "You are a helpful assistant running locally on a Raspberry Pi. Answer concisely.",
    )


settings = Settings()