from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000)
    session_id: str | None = Field(None, pattern=r"^[A-Za-z0-9_-]{1,64}$")
    stream: bool = True
    temperature: float | None = Field(None, ge=0, le=2)
    max_tokens: int | None = Field(None, ge=1, le=2048)


class ChatStats(BaseModel):
    queue_ms: float | None = None
    ttft_ms: float | None = None
    total_ms: float | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    tokens_per_s: float | None = None


class ChatResponse(BaseModel):
    session_id: str
    model: str
    reply: str
    stats: ChatStats = ChatStats()