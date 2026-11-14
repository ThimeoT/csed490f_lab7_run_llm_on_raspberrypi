import json
import logging
import time

from fastapi import APIRouter, HTTPException, Query, Request, Response
from fastapi.responses import StreamingResponse

from .config import settings
from .schemas import ChatRequest, ChatResponse, DocumentIn

log = logging.getLogger("routes")
router = APIRouter()


def _line(event: dict) -> str:
    return json.dumps(event, ensure_ascii=False) + "\n"


async def _canned_stream(text: str):
    """Fixed response emitted in the same format as the LLM (no model call)."""
    yield {"type": "token", "content": text}
    yield {"type": "done", "stats": {"queue_ms": 0.0, "ttft_ms": 0.0, "total_ms": 0.0,
                                     "completion_tokens": 0}}


async def _run_turn(request: Request, session, req: ChatRequest):
    """One conversation turn: retrieval (optional) -> generation -> history."""
    llm = request.app.state.llm
    store = request.app.state.sessions
    rag = request.app.state.rag
    use_rag = req.use_rag if req.use_rag is not None else rag.is_ready

    # 1) Retrieval, outside the session lock (it only needs the question)
    hits: list[dict] = []
    retrieval_ms = None
    if use_rag:
        try:
            t0 = time.perf_counter()
            hits = await rag.retrieve(req.message, req.top_k)
            retrieval_ms = round((time.perf_counter() - t0) * 1000, 1)
        except Exception as exc:
            log.exception("Retrieval failed")
            yield {"type": "error", "detail": f"Retrieval failed: {type(exc).__name__}: {exc}"}
            return
        yield {"type": "sources", "sources": [rag.public(h) for h in hits]}

    # 2) Generation. No error yield while the lock is held (to avoid interleaving with other clients).
    final_ev, error = None, None
    async with session.lock:
        session.messages.append({"role": "user", "content": req.message})  # question brute

        if use_rag and not hits:
            source = _canned_stream(settings.rag_no_context_reply)
        elif hits:
            payload = rag.build_messages(settings.rag_system_prompt, session.messages, hits)
            source = llm.stream_chat(payload, req.temperature, req.max_tokens)
        else:  # RAG disabled : Project 1 behavior (no context, no system prompt)
            payload = [{"role": "system", "content": settings.system_prompt}, *session.messages]
            source = llm.stream_chat(payload, req.temperature, req.max_tokens)

        parts: list[str] = []
        completed = False
        try:
            async for ev in source:
                if ev["type"] == "done":
                    final_ev = ev
                    continue
                parts.append(ev["content"])
                yield ev
            session.messages.append({"role": "assistant", "content": "".join(parts)})
            store.trim(session)
            completed = True
        except Exception as exc:
            log.exception("Chat failed")
            error = f"{type(exc).__name__}: {exc}"
        finally:
            if not completed:  # client error or disconnected client, cancel the run
                session.messages.pop()

    if error:
        yield {"type": "error", "detail": error}
    else:
        final_ev = final_ev or {"type": "done", "stats": {}}
        final_ev["stats"]["retrieval_ms"] = retrieval_ms
        yield final_ev


@router.post("/api/chat")
async def chat(req: ChatRequest, request: Request):
    if request.app.state.llm.is_saturated():
        raise HTTPException(429, "Server busy, retry shortly", headers={"Retry-After": "5"})

    session = request.app.state.sessions.get_or_create(req.session_id)

    if req.stream:
        async def gen():
            yield _line({"type": "meta", "session_id": session.id, "model": settings.model})
            async for ev in _run_turn(request, session, req):
                yield _line(ev)

        return StreamingResponse(
            gen(),
            media_type="application/x-ndjson",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    reply: list[str] = []
    stats: dict = {}
    sources: list[dict] = []
    async for ev in _run_turn(request, session, req):
        t = ev["type"]
        if t == "token":
            reply.append(ev["content"])
        elif t == "sources":
            sources = ev["sources"]
        elif t == "done":
            stats = ev["stats"]
        elif t == "error":
            raise HTTPException(502, ev["detail"])
    return ChatResponse(session_id=session.id, model=settings.model,
                        reply="".join(reply), stats=stats, sources=sources)


# ---------------- sessions ----------------
@router.get("/api/sessions/{sid}")
async def get_session(sid: str, request: Request):
    session = request.app.state.sessions.get(sid)
    if session is None:
        raise HTTPException(404, "Unknown session")
    return {"session_id": sid, "messages": session.messages}


@router.delete("/api/sessions/{sid}")
async def delete_session(sid: str, request: Request):
    if not request.app.state.sessions.delete(sid):
        raise HTTPException(404, "Unknown session")
    return Response(status_code=204)


# ---------------- base de connaissances (RAG) ----------------
@router.post("/api/rag/documents", status_code=201)
async def rag_add(doc: DocumentIn, request: Request):
    try:
        return await request.app.state.rag.add_document(doc.name, doc.text)
    except ValueError as exc:
        raise HTTPException(422, str(exc))
    except Exception as exc:
        log.exception("Ingestion failed")
        raise HTTPException(502, f"Embedding failed: {type(exc).__name__}: {exc}")


@router.post("/api/rag/reindex")
async def rag_reindex(request: Request):
    try:
        return await request.app.state.rag.reindex()
    except Exception as exc:
        log.exception("Reindex failed")
        raise HTTPException(502, f"Embedding failed: {type(exc).__name__}: {exc}")


@router.get("/api/rag/documents")
async def rag_list(request: Request):
    return request.app.state.rag.list_documents()


@router.delete("/api/rag/documents/{name:path}")
async def rag_delete(name: str, request: Request):
    if not request.app.state.rag.delete_document(name):
        raise HTTPException(404, "Unknown document")
    return Response(status_code=204)


@router.get("/api/rag/search")
async def rag_search(
    request: Request,
    q: str = Query(..., min_length=1, max_length=500),
    k: int = Query(5, ge=1, le=20),
):
    """Retrieval only, no threshold: used to calibrate RAG_MIN_SCORE and evaluate the index."""
    rag = request.app.state.rag
    try:
        hits = await rag.retrieve(q, k, apply_threshold=False)
    except Exception as exc:
        raise HTTPException(502, f"Embedding failed: {type(exc).__name__}: {exc}")
    return {"query": q, "min_score": settings.min_score, "results": [rag.public(h) for h in hits]}


# ---------------- health ----------------
@router.get("/health")
async def health(request: Request):
    llm = request.app.state.llm
    try:
        models = await llm.available_models()
        reachable = True
    except Exception:
        models, reachable = [], False

    def has(name: str) -> bool:
        return name in models or f"{name}:latest" in models

    return {
        "status": "ok" if reachable else "degraded",
        "model": settings.model,
        "ollama_reachable": reachable,
        "model_available": has(settings.model),
        "embed_model_available": has(settings.embed_model),
        "rag": request.app.state.rag.status(),
        "active_sessions": len(request.app.state.sessions),
        "pending_requests": llm.pending,
    }