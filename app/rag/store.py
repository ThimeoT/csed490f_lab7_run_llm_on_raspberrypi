import json
import os
from pathlib import Path

import numpy as np


class VectorStore:
    """Vector store in memory (normalized numpy matrix), persisted on the disc."""
    def __init__(self, index_dir: str, model_name: str):
        self.dir = Path(index_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name
        self.vectors = np.zeros((0, 0), dtype=np.float32)
        self.meta: list[dict] = []          # {"doc", "chunk_id", "text"} (one entry per vector line)
        self.doc_hashes: dict[str, str] = {}  # doc -> sha256 of the content (to detect changes)
        self._load()

    # ---------- persistance ----------
    def _load(self) -> None:
        meta_path, vec_path = self.dir / "meta.json", self.dir / "vectors.npy"
        if not (meta_path.exists() and vec_path.exists()):
            return
        data = json.loads(meta_path.read_text(encoding="utf-8"))
        vectors = np.load(vec_path)
        # Another embedding model (different dimension) or an inconsistent index => start from scratch
        if data.get("model") != self.model_name or len(data["chunks"]) != len(vectors):
            return
        self.meta, self.doc_hashes, self.vectors = data["chunks"], data["docs"], vectors

    def save(self) -> None:
        tmp_v, tmp_m = self.dir / "vectors.tmp", self.dir / "meta.tmp"
        with open(tmp_v, "wb") as f:
            np.save(f, self.vectors)
        tmp_m.write_text(
            json.dumps(
                {"model": self.model_name, "chunks": self.meta, "docs": self.doc_hashes},
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        os.replace(tmp_v, self.dir / "vectors.npy")  # atomic write
        os.replace(tmp_m, self.dir / "meta.json")

    # ---------- documents ----------
    def has_current(self, doc: str, digest: str) -> bool:
        return self.doc_hashes.get(doc) == digest

    def remove_doc(self, doc: str, save: bool = True) -> bool:
        if doc not in self.doc_hashes:
            return False
        keep = [i for i, m in enumerate(self.meta) if m["doc"] != doc]
        self.meta = [self.meta[i] for i in keep]
        self.vectors = self.vectors[keep] if keep else np.zeros((0, 0), dtype=np.float32)
        del self.doc_hashes[doc]
        if save:
            self.save()
        return True

    def add_doc(self, doc: str, digest: str, chunks: list[str], vectors: np.ndarray) -> None:
        self.remove_doc(doc, save=False)
        self.vectors = vectors if self.vectors.size == 0 else np.vstack([self.vectors, vectors])
        self.meta += [{"doc": doc, "chunk_id": i, "text": c} for i, c in enumerate(chunks)]
        self.doc_hashes[doc] = digest
        self.save()

    # ---------- search ----------
    def search(self, qvec: np.ndarray, k: int) -> list[tuple[dict, float]]:
        """Normalized vectors: dot product = cosine similarity."""
        n = len(self.meta)
        if n == 0:
            return []
        scores = self.vectors @ qvec
        k = min(k, n)
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [(self.meta[i], float(scores[i])) for i in top]