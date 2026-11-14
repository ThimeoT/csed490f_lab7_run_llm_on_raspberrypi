from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000)
    session_id: str | None = Field(None, pattern=r"^[A-Za-z0-9_-]{1,64}$")
    stream: bool = True
    temperature: float | None = Field(None, ge=0, le=2)
    max_tokens: int | None = Field(None, ge=1, le=2048)
    # None = automatic (RAG active if the knowledge base is not empty)
    use_rag: bool | None = None
    top_k: int | None = Field(None, ge=1, le=8)


class ChatStats(BaseModel):
    queue_ms: float | None = None
    ttft_ms: float | None = None
    total_ms: float | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    tokens_per_s: float | None = None
    retrieval_ms: float | None = None


class Source(BaseModel):
    id: int
    source: str
    chunk_id: int
    score: float
    snippet: str


class ChatResponse(BaseModel):
    session_id: str
    model: str
    reply: str
    stats: ChatStats = ChatStats()
    sources: list[Source] = []


class DocumentIn(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    text: str = Field(..., min_length=1, max_length=200_000)