import asyncio
import time
import uuid
from dataclasses import dataclass, field


@dataclass
class Session:
    id: str
    messages: list[dict] = field(default_factory=list)
    last_used: float = field(default_factory=time.monotonic)
    # Prevents two simultaneous request corrupting the history of a same session
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


class SessionStore:
    def __init__(self, ttl_s: int, max_messages: int):
        self._ttl = ttl_s
        self._max = max_messages
        self._sessions: dict[str, Session] = {}

    def get_or_create(self, session_id: str | None = None) -> Session:
        self._purge()
        sid = session_id or uuid.uuid4().hex
        session = self._sessions.get(sid)
        if session is None:
            session = self._sessions[sid] = Session(id=sid)
        session.last_used = time.monotonic()
        return session

    def get(self, sid: str) -> Session | None:
        return self._sessions.get(sid)

    def delete(self, sid: str) -> bool:
        return self._sessions.pop(sid, None) is not None

    def trim(self, session: Session) -> None:
        """Keeps the N-last messages, by starting with a 'user' message."""
        if len(session.messages) > self._max:
            del session.messages[: -self._max]
        while session.messages and session.messages[0]["role"] != "user":
            session.messages.pop(0)

    def _purge(self) -> None:
        now = time.monotonic()
        expired = [
            sid for sid, s in self._sessions.items()
            if now - s.last_used > self._ttl and not s.lock.locked()
        ]
        for sid in expired:
            del self._sessions[sid]

    def __len__(self) -> int:
        return len(self._sessions)