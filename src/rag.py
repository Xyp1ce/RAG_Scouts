"""Módulo RAG (Recuperación y Generación) para RAG Scouts — ASMAC 2026.

Conecta ChromaDB (poblado por src/ingesta.py) con Google Gemini para:
1. Detectar opcionalmente la sección (rama) mencionada en la consulta.
2. Generar el embedding de consulta (`gemini-embedding-001`, task_type="RETRIEVAL_QUERY").
3. Ejecutar recuperación dual en ChromaDB:
   - Normativa oficial (`tipo="normativa"`) para el sustento formal.
   - Fichas de especialidades (`tipo="especialidad_ejemplo"`) como puente semántico.
4. Construir un prompt fundamentado exclusivamente en el contexto recuperado y
   generar la respuesta (completa o en streaming) con Gemini.

Uso por CLI para pruebas:
    python -m src.rag "Quiero hacer una actividad de primeros auxilios en campamento"
    python -m src.rag --rama Tropa "Un scout quiere programar un videojuego"
    python -m src.rag --solo-buscar "reforestación y reciclaje"
"""

from __future__ import annotations

import argparse
import os
import re
from dataclasses import dataclass
from typing import Any, Iterator

from src.ingesta import (
    BASE_DIR,
    CHROMA_DIR,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    RAMA_CANONICA,
)

DEFAULT_LLM_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")
DEFAULT_K_NORMATIVA = 3
DEFAULT_K_EJEMPLOS = 4
MIN_SIMILARITY_SCORE = 0.35

SYSTEM_INSTRUCTION = """
Eres el asistente oficial de RAG Scouts para orientar a dirigentes y jóvenes sobre el sistema nacional de especialidades de la Asociación de Scouts de México (ASMAC).

Reglas estrictas de respuesta:
1. Fundamenta TODA tu respuesta exclusivamente en el CONTEXTO RECUPERADO proporcionado en el mensaje del usuario (normativa oficial y fichas de referencia de ASMAC).
2. NO inventes áreas de desempeño, nombres oficiales de insignias ni requisitos que no aparezcan en el contexto recuperado.
3. Las únicas 4 áreas oficiales de desempeño de ASMAC son:
   - Habilidades para la vida
   - Medio Ambiente y Sustentabilidad
   - Paz y Acción Comunitaria
   - Salud y Bienestar
4. Si la actividad descrita podría encajar en más de un área según el enfoque, indica el área principal recomendada y menciona brevemente la alternativa posible según la documentación.
5. Si el contexto recuperado NO contiene información suficiente para clasificar la consulta o el tema está fuera del sistema de especialidades ASMAC, dilo con honestidad y sugiere al usuario dar más detalles sobre la acción o proyecto Scout, no prosigas con la respuesta a menos que el usuario aclare lo necesario.
6. No ofrezcas ayuda con los planes de desarrollo ni a como llenar una ficha ya que no tienes la información necesaria para eso.

Formato recomendado (claro, conciso y amigable en español):
- **Área de desempeño recomendada:** <Nombre oficial del área> 
- **Categoría / Tema relacionado:** <Sub-especialidad o tema oficial del contexto, si aplica> 
- **Justificación (Normativa ASMAC)**: Explicación breve basada en la normativa recuperada.
- **Especialidades de referencia similares:** Menciona 1 a 3 fichas del contexto (indicando su sección) que respalden la clasificación.
- **Recomendación práctica:** Si el usuario no indicó la sección (Manada, Tropa, Comunidad o Clan), ofrece ajustar la orientación si especifica la rama del joven.
  """

# Patrones cuidadosos para detectar la rama en lenguaje natural sin falsos positivos
# (evita confundir "para su comunidad" con la sección "Comunidad de Caminantes").
_PATRONES_RAMA: list[tuple[str, re.Pattern[str]]] = [
    ("Manada", re.compile(r"\b(manada|lobatos?|lobeznas?|seisena)\b", re.IGNORECASE)),
    ("Tropa", re.compile(r"\b(tropa|patrulla)\b", re.IGNORECASE)),
    (
        "Comunidad",
        re.compile(
            r"\b(caminantes?|comunidad\s+de\s+caminantes|(?:secci[oó]n|rama)\s+(?:de\s+)?comunidad|(?:soy\s+de|en|para)\s+comunidad\b(?!\s+(?:local|escolar|ind[ií]gena|rural|urbana|vecinal|scout)))",
            re.IGNORECASE,
        ),
    ),
    ("Clan", re.compile(r"\b(clan|rovers?)\b", re.IGNORECASE)),
]


@dataclass
class FragmentoRecuperado:
    """Fragmento recuperado de ChromaDB con metadatos y similitud coseno."""

    chunk_id: str
    text: str
    score: float
    tipo: str
    rama: str
    area: str
    especialidad: str
    tema_relacionado: str
    fuente: str

    def to_source_dict(self) -> dict[str, Any]:
        """Serialización compacta para exponer fuentes en la API."""
        return {
            "tipo": self.tipo,
            "rama": self.rama,
            "area": self.area,
            "especialidad": self.especialidad,
            "tema_relacionado": self.tema_relacionado,
            "fuente": self.fuente,
            "score": self.score,
        }


@dataclass
class RespuestaRAG:
    """Resultado completo de una consulta RAG."""

    reply: str
    rama_detectada: str | None
    fuentes: list[FragmentoRecuperado]


def normalizar_rama(rama: str | None) -> str | None:
    """Normaliza una rama explícita al nombre canónico o devuelve None."""
    if not rama:
        return None
    limpia = rama.strip().lower()
    if limpia in RAMA_CANONICA:
        return RAMA_CANONICA[limpia]
    for clave, valor in RAMA_CANONICA.items():
        if clave in limpia:
            return valor
    return None


def detectar_rama(consulta: str) -> str | None:
    """Detecta si la consulta menciona explícitamente una única sección Scout."""
    coincidencias = [rama for rama, patron in _PATRONES_RAMA if patron.search(consulta)]
    if len(coincidencias) == 1:
        return coincidencias[0]
    return None


def _construir_where(tipo: str, ramas: list[str] | None = None) -> dict[str, Any]:
    """Construye un filtro `where` válido para ChromaDB usando `$and` si hay >1 condición."""
    condiciones: list[dict[str, Any]] = [{"tipo": tipo}]
    if ramas:
        if len(ramas) == 1:
            condiciones.append({"rama": ramas[0]})
        else:
            condiciones.append({"rama": {"$in": ramas}})

    if len(condiciones) == 1:
        return condiciones[0]
    return {"$and": condiciones}


def _ramas_para_normativa(rama: str | None) -> list[str] | None:
    """Mapea la rama solicitada hacia las etiquetas de rama existentes en `tipo="normativa"`."""
    if not rama:
        return None
    if rama in ("Comunidad", "Clan"):
        return [rama, "Comunidad y Clan"]
    if rama == "Manada":
        # En docs/insignias.md Manada tiene la tabla de nombres y Tropa detalla los temas
        # relacionados de las mismas 12 categorías; incluir ambas mejora la justificación.
        return ["Manada", "Tropa"]
    return [rama]


def _extraer_fragmentos_resultado(
    resultado: dict[str, Any],
) -> list[FragmentoRecuperado]:
    """Convierte el diccionario crudo de `collection.query()` en una lista de FragmentoRecuperado."""
    ids = (resultado.get("ids") or [[]])[0]
    docs = (resultado.get("documents") or [[]])[0]
    metas = (resultado.get("metadatas") or [[]])[0]
    dists = (resultado.get("distances") or [[]])[0]

    fragmentos: list[FragmentoRecuperado] = []
    for chunk_id, doc, meta, dist in zip(ids, docs, metas, dists):
        meta = meta or {}
        # Con hnsw:space="cosine", ChromaDB devuelve distancia coseno = 1 - similitud
        similitud = round(max(0.0, min(1.0, 1.0 - float(dist))), 4)
        fragmentos.append(
            FragmentoRecuperado(
                chunk_id=str(chunk_id),
                text=str(doc),
                score=similitud,
                tipo=str(meta.get("tipo", "")),
                rama=str(meta.get("rama", "")),
                area=str(meta.get("area", "")),
                especialidad=str(meta.get("especialidad", "")),
                tema_relacionado=str(meta.get("tema_relacionado", "")),
                fuente=str(meta.get("fuente", "")),
            )
        )
    return fragmentos


class RAGScouts:
    """Orquestador RAG que integra ChromaDB y Google Gemini."""

    def __init__(
        self,
        api_key: str | None = None,
        llm_model: str = DEFAULT_LLM_MODEL,
        embedding_model: str = EMBEDDING_MODEL,
    ) -> None:
        import chromadb
        from dotenv import load_dotenv
        from google import genai

        load_dotenv(BASE_DIR / ".env")
        resolved_key = api_key or os.getenv("API_KEY") or os.getenv("GEMINI_API_KEY")
        if not resolved_key:
            raise RuntimeError(
                "No se encontró API_KEY (ni GEMINI_API_KEY) en el archivo .env"
            )

        if not CHROMA_DIR.exists():
            raise RuntimeError(
                f"No existe la base vectorial en {CHROMA_DIR}. Ejecuta primero: python -m src.ingesta --reset"
            )

        self.llm_model = llm_model
        self.embedding_model = embedding_model
        self.genai_client = genai.Client(api_key=resolved_key)

        self.chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        try:
            self.collection = self.chroma_client.get_collection(name=COLLECTION_NAME)
        except Exception as exc:
            raise RuntimeError(
                f"No se encontró la colección '{COLLECTION_NAME}' en {CHROMA_DIR}. "
                "Ejecuta primero: python -m src.ingesta --reset"
            ) from exc

        if self.collection.count() == 0:
            raise RuntimeError(
                f"La colección '{COLLECTION_NAME}' está vacía. Ejecuta: python -m src.ingesta --reset"
            )

    def _embed_query(self, consulta: str) -> list[float]:
        """Genera el embedding de la consulta del usuario con task_type RETRIEVAL_QUERY."""
        from google.genai import types

        respuesta = self.genai_client.models.embed_content(
            model=self.embedding_model,
            contents=consulta,
            config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
        )
        if not respuesta.embeddings:
            raise RuntimeError(
                "Gemini no devolvió ningún vector de embedding para la consulta."
            )
        return list(respuesta.embeddings[0].values)

    def _query_seguro(
        self,
        query_embedding: list[float],
        where: dict[str, Any],
        n_results: int,
    ) -> list[FragmentoRecuperado]:
        """Consulta ChromaDB ajustando n_results al total real disponible para ese filtro."""
        if n_results <= 0:
            return []
        try:
            # no entiendo porque los errores pero funciona (se arreglará en el futuro)
            resultado = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=where,
                include=["documents", "metadatas", "distances"],
            )
            return _extraer_fragmentos_resultado(resultado)
        except Exception:
            # Si el filtro tiene menos elementos que n_results en alguna versión de Chroma,
            # consultamos el conteo exacto y reintentamos.
            existentes = self.collection.get(where=where, include=[])
            disponibles = len(existentes.get("ids") or [])
            if disponibles == 0:
                return []
            resultado = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=min(n_results, disponibles),
                where=where,
                include=["documents", "metadatas", "distances"],
            )
            return _extraer_fragmentos_resultado(resultado)

    def recuperar(
        self,
        consulta: str,
        rama: str | None = None,
        k_normativa: int = DEFAULT_K_NORMATIVA,
        k_ejemplos: int = DEFAULT_K_EJEMPLOS,
    ) -> tuple[list[FragmentoRecuperado], str | None]:
        """Recuperación dual: normativa oficial + fichas de especialidad de referencia."""
        rama_activa = normalizar_rama(rama) or detectar_rama(consulta)
        query_embedding = self._embed_query(consulta)

        # Recuperar normativa oficial (docs/insignias.md)
        where_norm = _construir_where("normativa", _ramas_para_normativa(rama_activa))
        normativa = self._query_seguro(
            query_embedding, where=where_norm, n_results=k_normativa
        )

        # Recuperar fichas de especialidades de referencia (puente semántico)
        if rama_activa:
            where_esp_rama = _construir_where("especialidad_ejemplo", [rama_activa])
            ejemplos = self._query_seguro(
                query_embedding, where=where_esp_rama, n_results=k_ejemplos
            )
            # Si hay pocos resultados relevantes en esa rama específica (ej. áreas con 1 ficha),
            # complementamos con ejemplos de otras ramas para no perder el puente semántico.
            if len(ejemplos) < k_ejemplos:
                where_esp_todas = _construir_where("especialidad_ejemplo", None)
                extra = self._query_seguro(
                    query_embedding, where=where_esp_todas, n_results=k_ejemplos + 2
                )
                vistos = {f.chunk_id for f in ejemplos}
                for frag in extra:
                    if frag.chunk_id not in vistos:
                        ejemplos.append(frag)
                        vistos.add(frag.chunk_id)
                    if len(ejemplos) >= k_ejemplos:
                        break
        else:
            where_esp_todas = _construir_where("especialidad_ejemplo", None)
            ejemplos = self._query_seguro(
                query_embedding, where=where_esp_todas, n_results=k_ejemplos + 2
            )

        # Deduplicar y limitar a máximo 2 chunks por archivo fuente (evita que un anexo domine Top-K)
        ejemplos_filtrados: list[FragmentoRecuperado] = []
        conteo_fuente: dict[str, int] = {}
        for frag in ejemplos:
            if conteo_fuente.get(frag.fuente, 0) >= 2:
                continue
            ejemplos_filtrados.append(frag)
            conteo_fuente[frag.fuente] = conteo_fuente.get(frag.fuente, 0) + 1
            if len(ejemplos_filtrados) >= k_ejemplos:
                break

        # Combinar normativa primero y luego ejemplos, ambos ordenados por similitud descendente
        normativa.sort(key=lambda f: f.score, reverse=True)
        ejemplos_filtrados.sort(key=lambda f: f.score, reverse=True)
        return normativa + ejemplos_filtrados, rama_activa

    @staticmethod
    def construir_prompt(
        consulta: str,
        fragmentos: list[FragmentoRecuperado],
        rama_detectada: str | None = None,
    ) -> str:
        """Arma el prompt aumentado con los fragmentos recuperados y sus metadatos."""
        bloques_normativa: list[str] = []
        bloques_ejemplos: list[str] = []

        for idx, frag in enumerate(fragmentos, start=1):
            cabecera = (
                f"[Fuente {idx} | tipo={frag.tipo} | rama={frag.rama} | "
                f"área={frag.area} | especialidad={frag.especialidad} | similitud={frag.score:.2f}]"
            )
            bloque = f"{cabecera}\n{frag.text}"
            if frag.tipo == "normativa":
                bloques_normativa.append(bloque)
            else:
                bloques_ejemplos.append(bloque)

        secciones_ctx: list[str] = []
        if bloques_normativa:
            secciones_ctx.append(
                "### NORMATIVA OFICIAL ASMAC 2026\n\n"
                + "\n\n---\n\n".join(bloques_normativa)
            )
        if bloques_ejemplos:
            secciones_ctx.append(
                "### FICHAS DE ESPECIALIDADES DE REFERENCIA\n\n"
                + "\n\n---\n\n".join(bloques_ejemplos)
            )

        contexto = (
            "\n\n===\n\n".join(secciones_ctx)
            if secciones_ctx
            else "(Sin fragmentos recuperados)"
        )
        mejor_score = max((f.score for f in fragmentos), default=0.0)
        aviso_similitud = ""
        if mejor_score < MIN_SIMILARITY_SCORE:
            aviso_similitud = (
                "\nNOTA DEL SISTEMA: La similitud máxima recuperada es baja "
                f"({mejor_score:.2f}). Si la consulta no corresponde a especialidades Scouts "
                "o falta información en el contexto, indícalo con claridad.\n"
            )

        linea_rama = (
            f"Sección (rama) identificada en la consulta: {rama_detectada}"
            if rama_detectada
            else "Sección (rama) identificada en la consulta: No especificada"
        )

        return (
            f"CONTEXTO RECUPERADO DE LA BASE ASMAC:\n\n"
            f"{contexto}\n"
            f"{aviso_similitud}\n"
            f"---\n"
            f"{linea_rama}\n"
            f"CONSULTA DEL USUARIO:\n{consulta}\n\n"
            "Responde al usuario basándote únicamente en el contexto recuperado."
        )

    def _construir_contents(
        self,
        prompt_aumentado: str,
        historial: list[dict[str, str]] | None = None,
    ) -> Any:
        """Prepara el parámetro `contents` de Gemini (soporta historial conversacional opcional)."""
        if not historial:
            return prompt_aumentado

        from google.genai import types

        contents: list[Any] = []
        for turno in historial[-6:]:
            rol = "model" if turno.get("role") in ("assistant", "model") else "user"
            texto = (turno.get("content") or "").strip()
            if texto:
                contents.append(
                    types.Content(role=rol, parts=[types.Part.from_text(text=texto)])
                )
        contents.append(
            types.Content(
                role="user", parts=[types.Part.from_text(text=prompt_aumentado)]
            )
        )
        return contents

    def consultar(
        self,
        consulta: str,
        rama: str | None = None,
        historial: list[dict[str, str]] | None = None,
    ) -> RespuestaRAG:
        """Ejecuta el flujo RAG completo y devuelve la respuesta junto con las fuentes."""
        from google.genai import types

        fragmentos, rama_detectada = self.recuperar(consulta, rama=rama)
        prompt = self.construir_prompt(
            consulta, fragmentos, rama_detectada=rama_detectada
        )
        contents = self._construir_contents(prompt, historial=historial)

        respuesta = self.genai_client.models.generate_content(
            model=self.llm_model,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTION,
                temperature=0.2,
            ),
        )
        texto = respuesta.text or "No se recibió respuesta del modelo."
        return RespuestaRAG(
            reply=texto, rama_detectada=rama_detectada, fuentes=fragmentos
        )

    def consultar_stream(
        self,
        consulta: str,
        rama: str | None = None,
        historial: list[dict[str, str]] | None = None,
    ) -> tuple[Iterator[str], list[FragmentoRecuperado], str | None]:
        """Recupera contexto primero y devuelve un generador en streaming junto con las fuentes."""
        from google.genai import types

        fragmentos, rama_detectada = self.recuperar(consulta, rama=rama)
        prompt = self.construir_prompt(
            consulta, fragmentos, rama_detectada=rama_detectada
        )
        contents = self._construir_contents(prompt, historial=historial)

        def token_generator() -> Iterator[str]:
            stream = self.genai_client.models.generate_content_stream(
                model=self.llm_model,
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_INSTRUCTION,
                    temperature=0.2,
                ),
            )
            for chunk in stream:
                if chunk.text:
                    yield chunk.text

        return token_generator(), fragmentos, rama_detectada


_rag_instance: RAGScouts | None = None


def get_rag() -> RAGScouts:
    """Devuelve una única instancia inicializada de RAGScouts."""
    global _rag_instance
    if _rag_instance is None:
        _rag_instance = RAGScouts()
    return _rag_instance


# Pruebas de CLI


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prueba el motor RAG Scouts desde la terminal."
    )
    parser.add_argument(
        "consulta", help="Descripción de la acción, palabra clave o proyecto del Scout."
    )
    parser.add_argument(
        "--rama",
        choices=["Manada", "Tropa", "Comunidad", "Clan"],
        default=None,
        help="Filtra explícitamente por una sección Scout.",
    )
    parser.add_argument(
        "--solo-buscar",
        action="store_true",
        help="Muestra únicamente los fragmentos recuperados de ChromaDB sin llamar al LLM.",
    )
    args = parser.parse_args()

    rag = RAGScouts()
    if args.solo_buscar:
        fragmentos, rama_detectada = rag.recuperar(args.consulta, rama=args.rama)
        print(f"Rama detectada: {rama_detectada or 'Ninguna (búsqueda general)'}")
        print(f"Fragmentos recuperados: {len(fragmentos)}\n")
        for idx, frag in enumerate(fragmentos, start=1):
            print(
                f"{idx}. [{frag.tipo}] score={frag.score:.4f} | rama={frag.rama} | "
                f"área={frag.area} | especialidad={frag.especialidad}"
            )
            print(f"   fuente: {frag.fuente}")
            preview = frag.text.replace("\n", " ")[:180]
            print(f"   vista previa: {preview}...\n")
        return

    resultado = rag.consultar(args.consulta, rama=args.rama)
    print(f"Rama detectada: {resultado.rama_detectada or 'No especificada'}\n")
    print("=== RESPUESTA ===\n")
    print(resultado.reply)
    print("\n=== FUENTES CONSULTADAS ===")
    for idx, frag in enumerate(resultado.fuentes, start=1):
        print(
            f"  {idx}. [{frag.tipo}] ({frag.score:.2f}) {frag.rama} — {frag.area} "
            f"| {frag.especialidad} ({frag.fuente})"
        )


if __name__ == "__main__":
    main()
