"""API FastAPI para RAG Scouts — Asistente de Especialidades ASMAC 2026."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Iterator

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from src.ingesta import COLLECTION_NAME
from src.rag import get_rag


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Inicializa ChromaDB y el cliente de Gemini una sola vez al arrancar el servidor."""
    get_rag()
    yield


app = FastAPI(title="RAG Scouts API", version="0.2.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Rama-Detectada"],
)


class HistoryTurn(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str
    rama: str | None = None
    history: list[HistoryTurn] | None = None


class SourceItem(BaseModel):
    tipo: str
    rama: str
    area: str
    especialidad: str
    tema_relacionado: str
    fuente: str
    score: float


class ChatResponse(BaseModel):
    reply: str
    rama_detectada: str | None = None
    sources: list[SourceItem] = Field(default_factory=list)


def _serializar_historial(history: list[HistoryTurn] | None) -> list[dict[str, str]] | None:
    if not history:
        return None
    return [{"role": turn.role, "content": turn.content} for turn in history]


@app.get("/health")
def health():
    try:
        rag = get_rag()
        return {
            "status": "ok",
            "collection": COLLECTION_NAME,
            "indexed_chunks": rag.collection.count(),
            "model": rag.llm_model,
        }
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"RAG no disponible: {exc}")


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    user_message = req.message.strip()
    if not user_message:
        raise HTTPException(status_code=400, detail="El mensaje no puede estar vacío.")

    try:
        rag = get_rag()
        resultado = rag.consultar(
            consulta=user_message,
            rama=req.rama,
            historial=_serializar_historial(req.history),
        )
        return ChatResponse(
            reply=resultado.reply,
            rama_detectada=resultado.rama_detectada,
            sources=[SourceItem(**f.to_source_dict()) for f in resultado.fuentes],
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error en consulta RAG: {exc}")


@app.post("/chat/stream")
def chat_stream(req: ChatRequest):
    user_message = req.message.strip()
    if not user_message:
        raise HTTPException(status_code=400, detail="El mensaje no puede estar vacío.")

    # Recuperar contexto antes de iniciar el stream para devolver HTTP 500 limpio si falla
    try:
        rag = get_rag()
        raw_stream, _fuentes, rama_detectada = rag.consultar_stream(
            consulta=user_message,
            rama=req.rama,
            historial=_serializar_historial(req.history),
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error al preparar contexto RAG: {exc}")

    def token_generator() -> Iterator[str]:
        try:
            yield from raw_stream
        except Exception as exc:
            yield f"\n[Error al generar respuesta con Gemini: {exc}]"

    headers = {}
    if rama_detectada:
        headers["X-Rama-Detectada"] = rama_detectada

    return StreamingResponse(
        token_generator(),
        media_type="text/plain; charset=utf-8",
        headers=headers,
    )
