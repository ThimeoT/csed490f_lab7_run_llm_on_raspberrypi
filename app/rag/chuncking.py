import re
from pathlib import Path

SUPPORTED = {".txt", ".md", ".pdf"}


def load_text(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        from pypdf import PdfReader  # lazy import : only for PDFs

        reader = PdfReader(str(path))
        return "\n\n".join((page.extract_text() or "") for page in reader.pages)
    return path.read_text(encoding="utf-8", errors="ignore")


def chunk_text(text: str, size: int = 800, overlap: int = 120) -> list[str]:
    """Sliding window that respects paragraphs, with overlap."""
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()

    pieces: list[str] = []
    for paragraph in (p.strip() for p in text.split("\n\n")):
        if not paragraph:
            continue
        pieces.extend([paragraph] if len(paragraph) <= size else _split_long(paragraph, size))

    chunks: list[str] = []
    current = ""
    for piece in pieces:
        if not current:
            current = piece
        elif len(current) + len(piece) + 2 <= size:
            current = f"{current}\n\n{piece}"
        else:
            chunks.append(current)
            current = _tail(current, overlap) + piece
    if current:
        chunks.append(current)
    return [c for c in chunks if len(c) >= 20]


def _split_long(paragraph: str, size: int) -> list[str]:
    out: list[str] = []
    cur = ""
    for sentence in re.split(r"(?<=[.!?])\s+", paragraph):
        while len(sentence) > size:  # too long sentence : hard cut
            if cur:
                out.append(cur)
                cur = ""
            out.append(sentence[:size])
            sentence = sentence[size:]
        if cur and len(cur) + len(sentence) + 1 > size:
            out.append(cur)
            cur = sentence
        else:
            cur = f"{cur} {sentence}".strip()
    if cur:
        out.append(cur)
    return out


def _tail(text: str, overlap: int) -> str:
    """End of the previous chunk, starting on a word boundary."""
    if overlap <= 0:
        return ""
    tail = text[-overlap:]
    cut = tail.find(" ")
    if cut != -1:
        tail = tail[cut + 1:]
    return tail + "\n\n" if tail else ""