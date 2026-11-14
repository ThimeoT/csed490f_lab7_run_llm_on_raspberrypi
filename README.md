# Lab 7 - Run LLM on Raspberry Pi
CSED490F - Deep Learning Implementations

A lightweight REST server running on a Raspberry Pi 5 that exposes a local chatbot (Ollama + Llama 3.2) to external clients (**Project 1**), extended with a Retrieval-Augmented Generation pipeline over a custom knowledge base (**Project 2**).

- **Project 1**: web server (FastAPI) between the local LLM and clients: REST API, token streaming, multi-turn sessions, load control, minimal latency.
- **Project 2**: RAG built on top of Project 1: ingestion (chunk -> embed -> index), retrieval (top-k cosine similarity), prompt augmentation, source citation.

> **Note on the NPU.** The Raspberry Pi 5 has a Hailo-8L NPU (AI Kit, Lab 6), but it is **not used here**. It runs vision CNNs compiled to HEF files. Ollama / llama.cpp run the LLM and the embedding model on the **CPU** (4 cores), so generation and embeddings share the same resources.

---

## 1. Architecture

```
 Clients (browser UI, curl, chat_cli.py, benchmark.py)
        |  HTTP / NDJSON streaming (same Wi-Fi as the Pi)
        v
 +--------------------------------------------------------+
 |  FastAPI server (uvicorn, 1 worker)         app/       |
 |                                                        |
 |  routes.py   -> endpoints, orchestrates one chat turn  |
 |  sessions.py -> in-memory conversations (TTL, trim)    |
 |  llm.py      -> Ollama chat: queue, streaming, stats   |
 |  rag/        -> chunking, vector store, RagService     |
 +-----------------------+--------------------------------+
                         |  ollama-python (AsyncClient)
                         v
 +--------------------------------------------------------+
 |  Ollama daemon (127.0.0.1:11434)                       |
 |  - llama3.2        (chat)                              |
 |  - nomic-embed-text (embeddings)                       |
 +--------------------------------------------------------+
```

Flow of one chat turn (`routes.py::_run_turn`):

1. If RAG is active: embed the question -> top-k chunks above a similarity threshold -> emit a `sources` event.
2. If RAG is active but nothing relevant is found: answer with a fixed "not found" message **without calling the LLM**.
3. Otherwise build the prompt (context injected only into the *last* user message), then stream tokens from Ollama.
4. On success, store the raw question + the answer in the session history (never the retrieved chunks).

### Design choices (and why)

| Choice | Reason |
|---|---|
| FastAPI + uvicorn, **single worker** | Async + native streaming. Sessions live in memory and the Pi runs one loaded model, so more workers would add nothing. |
| NDJSON streaming | The first token arrives quickly (low time-to-first-token) instead of waiting for the full answer. `"stream": false` is also supported. |
| `keep_alive` + warmup at startup | Avoids loading the model on the first request (cold start). |
| Semaphore + bounded queue -> `429` | The Pi can generate one answer at a time; extra requests wait, and beyond the queue limit the server answers `429` with `Retry-After`. |
| Per-session lock | Two concurrent requests on the same session cannot corrupt the history. |
| No LangChain, only `ollama` + `numpy` | The assignment allows "basic API integrations". Fewer heavy dependencies on the Pi, and the pipeline stays small and readable. |
| numpy brute-force cosine search | A few thousand chunks = one matrix product (< 1 ms). Chroma / FAISS would be overkill. Index persisted on disk, one hash per document (only new or modified files are re-embedded). |
| Similarity threshold + canned reply | Avoids letting a small model hallucinate when the knowledge base has nothing relevant. |
| Context injected only in the last message | Keeps the history small (context window ~4096 tokens by default in Ollama). |

---

## 2. Repository structure

```
lab7_project1/
├── requirements.txt
├── run.sh                    # creates ./venv, installs deps, starts uvicorn on 0.0.0.0:8000
├── knowledge_base/           # RAG source of truth (.txt / .md / .pdf)
├── data/index/               # generated: vectors.npy + meta.json (can be deleted, it is rebuilt)
├── app/
│   ├── main.py               # app factory, lifespan, background warmup + reindex
│   ├── config.py             # all settings, overridable with environment variables
│   ├── schemas.py            # Pydantic request / response models
│   ├── routes.py             # REST endpoints + chat-turn orchestration
│   ├── llm.py                # Ollama chat wrapper: queue, streaming, metrics
│   ├── sessions.py           # in-memory session store (TTL, history trimming)
│   └── rag/
│       ├── chunking.py       # file loading (txt/md/pdf) + paragraph-aware chunking with overlap
│       ├── store.py          # numpy vector store + atomic persistence
│       └── service.py        # embeddings, ingestion, retrieval, prompt building
├── static/index.html         # minimal chat UI (streaming, RAG toggle, sources, stats)
└── client/
    ├── chat_cli.py           # terminal client
    ├── benchmark.py          # latency / throughput benchmark
    ├── rag_eval.py           # retrieval evaluation + with/without RAG comparison
    └── eval_questions.json   # evaluation questions
```

---

## 3. Requirements

+ fastapi>=0.115
+ uvicorn[standard]>=0.30
+ ollama>=0.4
+ requests>=2.31
+ numpy>=1.26
+ pypdf>=4.0 (only used to read PDF files in `knowledge_base/`; can be removed if you only use `.md` / `.txt`)

Hardware / OS (same as Lab 6): Raspberry Pi 5 + AI Kit, Raspberry Pi OS **64-bit** (Bookworm). Python 3.11 is fine (the code needs >= 3.10).

---

## 4. Setup on the Raspberry Pi

```bash
# 1. Ollama (once)
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3.2               # chat model
ollama pull nomic-embed-text       # embedding model (Project 2)
ollama list                        # verify both models are here

# 2. Put the project on the Pi (scp or VS Code), then:
cd lab7_project1
chmod +x run.sh
./run.sh                           # or: CHAT_MODEL=llama3.2:1b ./run.sh
```

- `run.sh` creates its **own** virtual environment (`lab7_project1/venv`). Do **not** reuse `~/hailo-venv` from Lab 6 (it pins `torch==2.4.1` and `numpy<2`, which this project does not need).
- Always run a **single** uvicorn worker (sessions are in memory).
- At startup the server warms up the chat model and indexes `knowledge_base/` in the background. It answers immediately; check `/health` to see when the index is ready.
- The Pi and the client computer must be on the **same Wi-Fi network** (as for SSH). Use `http://raspberrypi.local:8000` (or the custom hostname you set; `hostname -I` gives the IP).
- If a network blocks device-to-device traffic, use an SSH tunnel: `ssh -L 8000:localhost:8000 <user>@raspberrypi.local`, then use `http://localhost:8000`.

---

## 5. Usage

```bash
curl http://raspberrypi.local:8000/health
curl -N -X POST http://raspberrypi.local:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"why is the sky blue?"}'
curl -X POST http://raspberrypi.local:8000/api/chat -H "Content-Type: application/json" \
  -d '{"message":"hi","stream":false}'
python client/chat_cli.py --url http://raspberrypi.local:8000            # add --no-rag to disable RAG
python client/benchmark.py --url http://raspberrypi.local:8000 -n 10 -c 1
python client/benchmark.py --url http://raspberrypi.local:8000 -n 10 -c 3   # load test
```

In a web browser, `http://raspberrypi.local:8000/` opens the chat (with a RAG toggle and the retrieved sources) and `http://raspberrypi.local:8000/docs` the Swagger documentation.

### RAG

```bash
# Drop documents in knowledge_base/ (.txt, .md, .pdf), then re-index:
curl -X POST http://raspberrypi.local:8000/api/rag/reindex

# Or upload text through the API (also written to knowledge_base/):
curl -X POST http://raspberrypi.local:8000/api/rag/documents -H "Content-Type: application/json" \
  -d '{"name":"notes.md","text":"..."}'

# Inspect retrieval only (no threshold): used to calibrate RAG_MIN_SCORE
curl -G http://raspberrypi.local:8000/api/rag/search --data-urlencode "q=your question" -d k=5

# Ask with / without RAG
curl -N -X POST http://raspberrypi.local:8000/api/chat -H "Content-Type: application/json" \
  -d '{"message":"your question","use_rag":true}'
```

`use_rag` is `null` by default (automatic: RAG is active when the knowledge base is not empty). Set it to `true` / `false` to force it, which is useful for with / without comparisons.

---

## 6. API reference

| Method | Path | Description |
|---|---|---|
| POST | `/api/chat` | Send a message (streaming NDJSON by default). |
| GET | `/api/sessions/{id}` | Conversation history of a session. |
| DELETE | `/api/sessions/{id}` | Delete a session. |
| POST | `/api/rag/documents` | Add / update a document `{name, text}`. |
| GET | `/api/rag/documents` | List indexed documents and their chunk counts. |
| DELETE | `/api/rag/documents/{name}` | Remove a document (index + file). |
| POST | `/api/rag/reindex` | Sync the index with `knowledge_base/` (only new / changed files are embedded). |
| GET | `/api/rag/search?q=&k=` | Retrieval only, no threshold (debug / calibration). |
| GET | `/health` | Ollama reachability, models available, index size, active sessions, pending requests. |

**`POST /api/chat` body**: `message` (required), `session_id` (optional, reuse it to continue a conversation), `stream` (default `true`), `temperature`, `max_tokens`, `use_rag` (`null` / `true` / `false`), `top_k`.

**Streaming events** (one JSON object per line, `application/x-ndjson`):

```
{"type":"meta","session_id":"...","model":"llama3.2"}
{"type":"sources","sources":[{"id":1,"source":"doc.md","chunk_id":0,"score":0.71,"snippet":"..."}]}
{"type":"token","content":"..."}
{"type":"done","stats":{"queue_ms":..,"ttft_ms":..,"total_ms":..,"prompt_tokens":..,"completion_tokens":..,"tokens_per_s":..,"retrieval_ms":..}}
{"type":"error","detail":"..."}
```

Status codes: `429` when the server is saturated (`Retry-After` header), `502` when Ollama fails (non-streaming mode).

---

## 7. Configuration

All settings are environment variables (see `app/config.py`).

| Variable | Default | Meaning |
|---|---|---|
| `OLLAMA_URL` | `http://127.0.0.1:11434` | Ollama address (not `OLLAMA_HOST`, which Ollama itself uses). |
| `CHAT_MODEL` | `llama3.2` | Chat model (try `llama3.2:1b` if too slow). |
| `KEEP_ALIVE` | `30m` | How long Ollama keeps the model in RAM. |
| `WARMUP` | `1` | Load the chat model at startup. |
| `NUM_PREDICT` | `512` | Max generated tokens per answer. |
| `MAX_CONCURRENT` | `1` | Simultaneous generations. |
| `MAX_QUEUE` | `4` | Waiting requests before `429`. |
| `MAX_HISTORY` | `12` | Messages kept per session. |
| `SESSION_TTL_S` | `1800` | Session expiry (seconds). |
| `EMBED_MODEL` | `nomic-embed-text` | Embedding model. The index is rebuilt automatically if it changes. |
| `EMBED_QUERY_PREFIX` / `EMBED_DOC_PREFIX` | `search_query: ` / `search_document: ` | Prefixes required by `nomic-embed-text`. Set both to empty for other models (e.g. `bge-m3`). |
| `CHUNK_CHARS` / `CHUNK_OVERLAP` | `800` / `120` | Chunk size and overlap (characters). |
| `RAG_TOP_K` | `3` | Chunks injected in the prompt. |
| `RAG_MIN_SCORE` | `0.45` | Minimum cosine similarity. **Needs calibration** (see below). |
| `KB_DIR` / `INDEX_DIR` | `./knowledge_base` / `./data/index` | Documents and persisted index. |
| `RAG_SYSTEM_PROMPT` / `RAG_NO_CONTEXT_REPLY` | see code | Prompt used with RAG and the fixed "not found" answer. |

---

## 8. Evaluation

```bash
python client/rag_eval.py --url http://raspberrypi.local:8000 --no-generate   # retrieval only (fast)
python client/rag_eval.py --url http://raspberrypi.local:8000                 # + answers with / without RAG
python client/benchmark.py --url http://raspberrypi.local:8000 -n 10 -c 1     # latency (TTFT, tokens/s)
```

- `client/eval_questions.json`: questions with the expected source file (`expected_source`) and the expected answer. Questions with `"expected_source": null` are **out of scope** and must be rejected ("not found").
- Test documents are fictional (lighthouse, bakery, space probe, chess club) in `knowledge_base/`, so the model cannot know the answers without RAG. Keep the file names identical to the `expected_source` values.
- Metrics: **hit@1 / hit@3** (is the right document ranked first / in the top 3?) and the out-of-scope rejection rate. The with / without RAG comparison is **qualitative** (read the answers).
- **Calibrating `RAG_MIN_SCORE`**: compare, with `/api/rag/search`, the top score of relevant questions and of out-of-scope ones, then set the threshold between the two.
- RAG overhead: `retrieval_ms` in the `done` event; compare TTFT with and without RAG using the same benchmark.

---

## 9. Status and next steps

**Implemented**: everything described above (Project 1 server, Project 2 RAG, UI, clients, evaluation scripts).

**To verify on the Pi** (the code was written without being run against a real Ollama): `/health` shows both models available, indexing works, the UI streams correctly, `rag_eval.py` gives sensible scores. Fix whatever the first run reveals.

**Suggested next steps**
1. Calibrate `RAG_MIN_SCORE` and run `rag_eval.py` / `benchmark.py` to get the numbers and screenshots for the report (1B vs 3B, `-c 1` vs `-c 3`, with vs without RAG).
2. Report (max 4 pages): architecture diagram, justification of the choices above, demo screenshots (UI with sources, "not found" case), measurements, limitations.
3. Optional improvements:
   - **Query rewriting**: retrieval only looks at the last question, so follow-ups ("and the other one?") retrieve poorly. Rewrite the question using the history before embedding.
   - French documents: try `EMBED_MODEL=bge-m3` with empty prefixes (heavier, ~1.2 GB); `nomic-embed-text` is mostly English.
   - Re-ranking or hybrid (BM25 + vectors) search.
   - API key / rate limiting if the server is exposed beyond the local network.
   - Persist sessions (currently lost on restart).

**Known limitations**
- Single worker, in-memory sessions.
- LLM and embeddings share the CPU: ingesting documents while a chat is running slows it down. Let indexing finish before benchmarking.
- Chat and embedding models are both kept in RAM (comfortable on 8 GB; use `llama3.2:1b` on smaller boards).
- Scanned PDFs (no text layer) are ignored; there is no OCR.
- The similarity threshold is not a perfect relevance detector (an out-of-scope question sharing a keyword with a document can pass).

---