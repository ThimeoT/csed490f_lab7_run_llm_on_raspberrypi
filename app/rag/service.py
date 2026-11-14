import asyncio
import hashlib
import re
import time
from collections import Counter
from pathlib import Path

import numpy as np
from ollama import AsyncClient

from ..config import Settings
from .chunking import SUPPORTED, chunk_text, load_text
from .store import VectorStore

PROMPT_TEMPLATE = """Answer the question using ONLY the context below. \
If the context does not contain the answer, say you could not find it in the knowledge base. \
Cite the sources you use like [1], [2].

Context:
{context}

Question: {question}"""


class RagService:
    """Ingestion (chunk -> embed -> index), retrieval (top-k) and prompt construction."""

    def __init__(self, cfg: Settings, client: AsyncClient):
        self.cfg = cfg
        self.client = client
        self.store = VectorStore(cfg.index_dir, cfg.embed_model)
        self._lock = asyncio.Lock() # one ingestion at a time : the embbedings run on the CPU
    @property
    def is_ready(self) -> bool:
        return len(self.store.meta) > 0

    def status(self) -> dict:
        return {
            "embed_model": self.cfg.embed_model,
            "documents": len(self.store.doc_hashes),
            "chunks": len(self.store.meta),
        }

    # ---------- embeddings ----------
    async def _embed(self, texts: list[str]) -> np.ndarray:
        rows: list[list[float]] = []
        for i in range(0, len(texts), self.cfg.embed_batch):
            res = await self.client.embed(
                model=self.cfg.embed_model,
                input=texts[i : i + self.cfg.embed_batch],
                keep_alive=self.cfg.keep_alive,
            )
            rows.extend(res["embeddings"])
        arr = np.asarray(rows, dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return arr / norms

    # ---------- ingestion ----------
    async def ingest_text(self, name: str, text: str) -> dict:
        async with self._lock:
            digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
            if self.store.has_current(name, digest):
                n = sum(1 for m in self.store.meta if m["doc"] == name)
                return {"document": name, "status": "unchanged", "chunks": n}
            chunks = chunk_text(text, self.cfg.chunk_chars, self.cfg.chunk_overlap)
            if not chunks:
                raise ValueError(f"'{name}' contains no usable text")
            t0 = time.perf_counter()
            vectors = await self._embed([self.cfg.embed_doc_prefix + c for c in chunks])
            self.store.add_doc(name, digest, chunks, vectors)
            return {
                "document": name,
                "status": "indexed",
                "chunks": len(chunks),
                "embed_ms": round((time.perf_counter() - t0) * 1000),
            }

    async def reindex(self) -> dict:
        """Synchronizes the index with the knowledge_base/ folder (source of truth)."""
        kb = Path(self.cfg.kb_dir)
        kb.mkdir(parents=True, exist_ok=True)
        report, present = [], set()
        for path in sorted(kb.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in SUPPORTED:
                continue
            name = path.relative_to(kb).as_posix()
            present.add(name)
            try:
                text = await asyncio.to_thread(load_text, path)  # PDF = CPU, out of async loop
                report.append(await self.ingest_text(name, text))
            except ValueError as exc:
                report.append({"document": name, "status": "skipped", "reason": str(exc)})
        removed = [d for d in list(self.store.doc_hashes) if d not in present]
        for d in removed:
            self.store.remove_doc(d)
        return {"documents": report, "removed": removed, **self.status()}

    async def add_document(self, name: str, text: str) -> dict:
        safe = re.sub(r"[^\w.\- ]", "_", Path(name).name).strip() or "document"
        if Path(safe).suffix.lower() not in {".txt", ".md"}:
            safe += ".md"
        kb = Path(self.cfg.kb_dir)
        kb.mkdir(parents=True, exist_ok=True)
        (kb / safe).write_text(text, encoding="utf-8")  # le dossier reste la source de vérité
        return await self.ingest_text(safe, text)

    def delete_document(self, name: str) -> bool:
        kb = Path(self.cfg.kb_dir).resolve()
        target = (kb / name).resolve()
        if kb not in target.parents:  # anti path-traversal
            return False
        removed = self.store.remove_doc(name)
        had_file = target.is_file()
        if had_file:
            target.unlink()
        return removed or had_file

    def list_documents(self) -> list[dict]:
        counts = Counter(m["doc"] for m in self.store.meta)
        return [{"name": n, "chunks": counts.get(n, 0)} for n in sorted(self.store.doc_hashes)]

    # ---------- retrieval ----------
    async def retrieve(self, query: str, k: int | None = None, apply_threshold: bool = True) -> list[dict]:
        if not self.is_ready:
            return []
        qvec = (await self._embed([self.cfg.embed_query_prefix + query]))[0]
        hits: list[dict] = []
        for meta, score in self.store.search(qvec, k or self.cfg.top_k):
            if apply_threshold and score < self.cfg.min_score:
                break # results are sorted: the following ones are even worse
            hits.append({
                "id": len(hits) + 1,
                "source": meta["doc"],
                "chunk_id": meta["chunk_id"],
                "score": round(score, 3),
                "text": meta["text"],
            })
        return hits

    @staticmethod
    def public(hit: dict) -> dict:
        text = hit["text"]
        return {
            "id": hit["id"],
            "source": hit["source"],
            "chunk_id": hit["chunk_id"],
            "score": hit["score"],
            "snippet": text[:200] + ("…" if len(text) > 200 else ""),
        }

    # ---------- prompt ----------
    @staticmethod
    def build_messages(system_prompt: str, history: list[dict], hits: list[dict]) -> list[dict]:
        """The context is injected only in the last message (the history remains clean)."""
        context = "\n\n".join(f"[{h['id']}] (source: {h['source']})\n{h['text']}" for h in hits)
        augmented = PROMPT_TEMPLATE.format(context=context, question=history[-1]["content"])
        return [
            {"role": "system", "content": system_prompt},
            *history[:-1],
            {"role": "user", "content": augmented},
        ]