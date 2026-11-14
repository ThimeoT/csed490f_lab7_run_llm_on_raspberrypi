# Lab 7 Report: Running an LLM on Raspberry Pi (CSED490F)

> Items marked *(extension)* go beyond the assignment and are justified. Values in `[...]` must be filled with measurements.

---

## Page 1 — Architecture

### 1. Problem

The lab slides run Llama 3.2 through `ollama run`, a terminal chat reachable only from the Pi itself. Project 1 asks for a lightweight server that exposes this chatbot to remote clients with minimal latency.

The main constraint is the hardware. Ollama and llama.cpp run the model on the 4 CPU cores of the Pi 5. They cannot use the Hailo-8L NPU, which only runs vision CNNs compiled to HEF files. A full answer therefore takes seconds `[insert measured total time]`, and the CPU can only serve one generation efficiently at a time.

Every choice below follows from this: hide latency (streaming), avoid wasted work (warm model, bounded queue), keep the server small.

### 2. Global architecture

```
Client (browser / curl / Python)
   │ POST /api/chat (JSON)           ▲ NDJSON stream: meta · sources · token… · done
   ▼                                 │
FastAPI + uvicorn (0.0.0.0:8000) ───────────────────────────────
  routes.py    HTTP endpoints, orchestration of one chat turn
  sessions.py  conversation history per session_id
  llm.py       Ollama wrapper: queue, streaming, metrics
  rag/         Project 2: retrieval before generation
   │ ollama-python AsyncClient (localhost only)
   ▼
Ollama daemon (127.0.0.1:11434): llama3.2 (chat) · nomic-embed-text (embeddings)
```

The code is split in layers: HTTP in `routes.py`, the LLM backend in `llm.py`, state in `sessions.py`. Each part can change without touching the others. Project 2 was added without modifying `llm.py` or `sessions.py`.

Ollama listens on localhost only, so FastAPI is the single exposed service, and it validates every input with Pydantic (message ≤ 4000 characters, `session_id` regex).

### 3. Client → FastAPI → Ollama

The client sends one `POST /api/chat` with a `message` and an optional `session_id`. The server calls Ollama with `stream=True` and forwards each chunk immediately as one JSON line (NDJSON): `meta` (session id), `sources` (RAG), `token`, `done` (latency statistics) or `error`.

The client pushes a message and the server pushes tokens on the same connection, while the `session_id` carries the conversation across requests.

### 4. Justification of the choices

- **FastAPI (async) rather than Flask.** The server mostly waits for Ollama. With `async`/`await`, one process serves many connections without one thread per client, which is cheaper on a Pi. `StreamingResponse` is native, Swagger (`/docs`) is generated for free, and we run a **single uvicorn worker** because sessions live in memory and only one model is loaded.

- **Streaming.** Without it, the user waits for the whole answer. With it, the perceived latency becomes the time to first token (TTFT) `[insert measured TTFT vs total time]`. We chose NDJSON over SSE or WebSocket because it works over a plain `POST` (the browser `EventSource` only supports `GET`), needs no handshake, and is trivial to parse with `fetch`, `requests` or `curl`.

- **Sessions.** The LLM is stateless, so the server stores the history under a `session_id` and the client only sends the new message. We keep the last 12 messages, starting with a user message, to stay inside the ~4096-token context window. Sessions expire after 30 minutes. A per-session `asyncio.Lock` prevents two simultaneous requests from interleaving the history, and a failed or cancelled turn is rolled back so the history stays consistent.

- **Latency and load control *(extension)*.** `keep_alive=30m` plus a warm-up request at startup avoid reloading the model on the first request. A semaphore allows one generation at a time and a bounded queue holds 4 more. Beyond that, the server answers `429` with `Retry-After`. An unbounded queue would make latency grow without limit until requests time out, whereas failing fast keeps latency predictable.

```python
async with self._sem:                          # one generation at a time on the Pi CPU
    stream = await self.client.chat(..., stream=True, keep_alive=cfg.keep_alive)
    async for chunk in stream:
        if chunk["message"]["content"] and ttft_ms is None:
            ttft_ms = (time.perf_counter() - t_start) * 1000   # time to first token
        yield {"type": "token", "content": chunk["message"]["content"]}
```

---

## Page 2 — RAG implementation

### Overview

Documents are split, embedded and indexed once. At question time, the question is embedded, the most similar chunks are retrieved, and they are injected into the prompt. Everything runs on the Pi, with no external API.

### 1. Document ingestion

The `knowledge_base/` folder (`.txt`, `.md`, `.pdf` via `pypdf`) is the single source of truth. It is indexed at startup in a background task, so the server answers immediately, or on demand with `POST /api/rag/reindex`. Documents can also be added through the API, which writes them to the same folder.

Three choices go beyond the brief *(extension)*:

- **Incremental indexing.** A SHA-256 hash is stored per document, and only new or modified files are re-embedded. Embedding competes with the LLM for the CPU, so avoiding useless work keeps restarts fast. Deleted files are removed from the index.
- **Atomic persistence.** The index (`vectors.npy` + `meta.json`) is written to temporary files and swapped with `os.replace`, so an abrupt power-off of the Pi cannot leave a corrupted index.
- **Model tag.** The embedding model name is stored in the index, which is rebuilt automatically if the model changes (different vector dimension).

### 2. Chunking and embeddings

Text is split on paragraph boundaries into chunks of about 800 characters with a 120-character overlap (`chunking.py`). Chunks must carry one idea: too large, and the embedding is diluted and wastes context window; too small, and the meaning is lost. The overlap prevents an answer from being cut at a boundary. With `top_k=3`, the injected context is about 600–750 tokens, which leaves room for the history in a 4096-token window.

Embeddings come from `nomic-embed-text` through the same Ollama daemon (`/api/embed`). This adds no new runtime dependency, the model is small, and it runs on CPU. We use the `search_document:` / `search_query:` prefixes the model expects, and L2-normalise every vector.

### 3. Cosine similarity search

Because vectors are normalised at ingestion, cosine similarity is a plain dot product, so the whole search is one matrix–vector product. `argpartition` extracts the top-k without sorting everything:

```python
scores = self.vectors @ qvec                 # cosine similarity (L2-normalised vectors)
top = np.argpartition(-scores, k - 1)[:k]    # top-k without a full sort
top = top[np.argsort(-scores[top])]
```

A vector database (Chroma, FAISS) would add dependencies and RAM for no gain at this scale. The search time is negligible compared with embedding the question `[insert measured retrieval_ms]`.

### 4. Prompt augmentation

The retrieved chunks are numbered and injected into the **last user message only**, with an instruction to answer from the context and cite sources:

```python
PROMPT_TEMPLATE = """Answer the question using ONLY the context below. \
If the context does not contain the answer, say you could not find it in the knowledge base. \
Cite the sources you use like [1], [2].

Context:
{context}

Question: {question}"""
```

The session history stores the raw question, not the augmented prompt. Otherwise the chunks would pile up in the history and overflow the context window after a few turns. The sources are sent to the client in a `sources` event before the first token, so every answer is traceable to a document, which supports the "accurate responses" goal.

### 5. Out-of-scope questions *(extension)*

Retrieval always returns something, even for an unrelated question, and a small 3B model asked to answer from irrelevant context tends to invent an answer. We therefore apply a minimum similarity score (`RAG_MIN_SCORE`, calibrated with `/api/rag/search` on relevant and irrelevant questions `[final value]`).

If no chunk passes, the server returns a fixed "not found in the knowledge base" message **without calling the LLM**:

```python
if use_rag and not hits:
    source = _canned_stream(settings.rag_no_context_reply)   # no LLM call
```

This guarantees no hallucination in that case and saves a full generation of several CPU seconds, which is consistent with the latency goal. The prompt instruction is a second line of defence when a borderline chunk passes the threshold. The `use_rag` flag forces RAG on or off per request, which allows controlled with/without comparisons.

### 6. Evaluation *(extension)*

To check that the pipeline really uses the knowledge base, we indexed fictional documents (a lighthouse, a bakery, a space probe, a chess club) that the model cannot know. `rag_eval.py` measures hit@1 and hit@3 of the retrieval and the rejection rate of out-of-scope questions, and compares the answers with and without RAG.