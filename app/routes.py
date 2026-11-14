import json
import logging

from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.responses import StreamingResponse

from .config import settings
from .schemas import ChatRequest, ChatResponse

log = logging.getLogger("routes")
router = APIRouter()


def _line(event: dict) -> str:
    return json.dumps(event, ensure_ascii=False) + "\n"


async def _run_turn(request: Request, session, req: ChatRequest):
    """One conversation run. No error yeld since the lock is held."""
    llm = request.app.state.llm
    store = request.app.state.sessions
    final_ev, error = None, None

    async with session.lock:
        session.messages.append({"role": "user", "content": req.message})
        payload = [{"role": "system", "content": settings.system_prompt}, *session.messages]
        parts: list[str] = []
        completed = False
        try:
            async for ev in llm.stream_chat(payload, req.temperature, req.max_tokens):
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
            if not completed:  # error or client disconnected : we cancel the run
                session.messages.pop()

    if error:
        yield {"type": "error", "detail": error}
    else:
        yield final_ev or {"type": "done", "stats": {}}


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
    async for ev in _run_turn(request, session, req):
        if ev["type"] == "token":
            reply.append(ev["content"])
        elif ev["type"] == "done":
            stats = ev["stats"]
        elif ev["type"] == "error":
            raise HTTPException(502, ev["detail"])
    return ChatResponse(
        session_id=session.id, model=settings.model, reply="".join(reply), stats=stats
    )


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


@router.get("/health")
async def health(request: Request):
    llm = request.app.state.llm
    try:
        models = await llm.available_models()
        reachable = True
    except Exception:
        models, reachable = [], False
    wanted = settings.model
    return {
        "status": "ok" if reachable else "degraded",
        "model": wanted,
        "ollama_reachable": reachable,
        "model_available": wanted in models or f"{wanted}:latest" in models,
        "active_sessions": len(request.app.state.sessions),
        "pending_requests": llm.pending,
    }