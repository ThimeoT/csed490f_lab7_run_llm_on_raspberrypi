import asyncio
import logging
import time
from collections.abc import AsyncIterator

from ollama import AsyncClient

from .config import Settings

log = logging.getLogger("llm")


class LLMService:
    """Thin layer around Ollama : pending list, streaming, metrics."""

    def __init__(self, cfg: Settings):
        self.cfg = cfg
        self.client = AsyncClient(host=cfg.ollama_url)
        self._sem = asyncio.Semaphore(cfg.max_concurrent)
        self._pending = 0  # running + pending requests

    @property
    def pending(self) -> int:
        return self._pending

    def is_saturated(self) -> bool:
        return self._pending >= self.cfg.max_concurrent + self.cfg.max_queue

    async def warmup(self) -> None:
        """Loads the model into the memory (empty prompt) to avoid cold start."""
        try:
            t0 = time.perf_counter()
            await self.client.generate(
                model=self.cfg.model, prompt="", keep_alive=self.cfg.keep_alive
            )
            log.info("Model '%s' warmed up in %.1fs", self.cfg.model, time.perf_counter() - t0)
        except Exception as exc:
            log.warning("Warmup failed (Ollama running? model pulled?): %s", exc)

    async def available_models(self) -> list[str]:
        resp = await self.client.list()
        return [m.get("model") or m.get("name") for m in resp["models"]]

    async def stream_chat(
        self,
        messages: list[dict],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[dict]:
        """Yield {'type': 'token', 'content': ...} puis {'type': 'done', 'stats': {...}}."""
        self._pending += 1
        t_enter = time.perf_counter()
        try:
            async with self._sem:
                t_start = time.perf_counter()
                options: dict = {"num_predict": max_tokens or self.cfg.num_predict}
                if temperature is not None:
                    options["temperature"] = temperature

                stream = await self.client.chat(
                    model=self.cfg.model,
                    messages=messages,
                    stream=True,
                    keep_alive=self.cfg.keep_alive,
                    options=options,
                )

                ttft_ms = None
                final = {}
                async for chunk in stream:
                    content = chunk["message"]["content"]
                    if content:
                        if ttft_ms is None:
                            ttft_ms = (time.perf_counter() - t_start) * 1000
                        yield {"type": "token", "content": content}
                    if chunk["done"]:
                        final = chunk

                t_end = time.perf_counter()
                n = final.get("eval_count")
                dur = final.get("eval_duration")
                yield {
                    "type": "done",
                    "stats": {
                        "queue_ms": round((t_start - t_enter) * 1000, 1),
                        "ttft_ms": round(ttft_ms, 1) if ttft_ms is not None else None,
                        "total_ms": round((t_end - t_start) * 1000, 1),
                        "prompt_tokens": final.get("prompt_eval_count"),
                        "completion_tokens": n,
                        "tokens_per_s": round(n / (dur / 1e9), 2) if n and dur else None,
                    },
                }
        finally:
            self._pending -= 1