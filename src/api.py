import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from google import genai
from google.genai import types
from pydantic import BaseModel

load_dotenv()

API_KEY = os.getenv("API_KEY")
if not API_KEY:
    raise RuntimeError("No se encontró API_KEY en el archivo .env")

client = genai.Client(api_key=API_KEY)

SYSTEM_INSTRUCTION = (
    "Eres el asistente de RAG Scouts para el sistema de especialidades de la ASMAC 2026. "
    "Responde en español de forma clara, breve y amigable. "
    "Ayuda al usuario a analizar acciones o proyectos de jóvenes Scouts para orientarlo "
    "sobre qué área de desempeño o especialidad podría corresponder. "
    "Nota: el módulo RAG aún está en construcción, así que aclara cuando sea una orientación preliminar."
)

app = FastAPI(title="RAG Scouts API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    reply: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    user_message = req.message.strip()
    if not user_message:
        raise HTTPException(status_code=400, detail="El mensaje no puede estar vacío.")

    try:
        response = client.models.generate_content(
            model="gemini-3.1-flash-lite",
            contents=user_message,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTION,
            ),
        )
        return ChatResponse(
            reply=response.text or "No se recibió respuesta del modelo."
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error al consultar Gemini: {exc}")


@app.post("/chat/stream")
def chat_stream(req: ChatRequest):
    user_message = req.message.strip()
    if not user_message:
        raise HTTPException(status_code=400, detail="El mensaje no puede estar vacío.")

    def token_generator():
        try:
            response_stream = client.models.generate_content_stream(
                model="gemini-3.1-flash-lite",
                contents=user_message,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_INSTRUCTION,
                ),
            )
            for chunk in response_stream:
                if chunk.text:
                    yield chunk.text
        except Exception as exc:
            yield f"\n[Error al consultar Gemini: {exc}]"

    return StreamingResponse(token_generator(), media_type="text/plain; charset=utf-8")
