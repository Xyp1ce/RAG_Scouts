"""
Lee las dos fuentes de conocimiento del proyecto:
1. Normativa oficial: docs/insignias.md (tipo="normativa")
2. Fichas de especialidades de referencia: data/normativa/<rama>/<area>/*.md
   (tipo="especialidad_ejemplo")

Limpia el ruido repetitivo de las plantillas, fragmenta únicamente cuando un
documento excede el tamaño objetivo, genera embeddings en lotes con Gemini
(gemini-embedding-001) y persiste los vectores + metadatos en ChromaDB.

Uso:
    python -m src.ingesta --dry-run   # Verifica lectura, limpieza y chunks sin llamar a la API
    python -m src.ingesta --reset     # Recrea la colección en chroma_db/ desde cero
    python -m src.ingesta             # Actualiza/inserta (upsert) los chunks en chroma_db/
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Rutas del proyecto (independientes del directorio de trabajo actual)
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "normativa"
NORMATIVA_MD = BASE_DIR / "docs" / "insignias.md"
CHROMA_DIR = BASE_DIR / "chroma_db"

# Configuración de ChromaDB y embeddings
COLLECTION_NAME = "rag_scouts"
EMBEDDING_MODEL = "gemini-embedding-001"
BATCH_SIZE = 25
BATCH_DELAY_SECONDS = 1.0
MAX_RETRIES = 4

# Parámetros de fragmentación
MAX_CHUNK_CHARS = 2600
CHUNK_OVERLAP_CHARS = 200

# Mapeos de secciones (ramas) y áreas de desempeño
RAMA_CANONICA: dict[str, str] = {
    "manada": "Manada",
    "tropa": "Tropa",
    "comunidad": "Comunidad",
    "clan": "Clan",
}

RAMA_DISPLAY: dict[str, str] = {
    "Manada": "Manada de Lobatos y Lobeznas",
    "Tropa": "Tropa de Scouts",
    "Comunidad": "Comunidad de Caminantes",
    "Clan": "Clan de Rovers",
    "Comunidad y Clan": "Comunidad de Caminantes y Clan de Rovers",
}

AREA_POR_CARPETA: dict[str, str] = {
    "EspecialidadesHabilidadesParaLaVida": "Habilidades para la vida",
    "EspecialidadesMedioAmbienteYSustentabilidad": "Medio Ambiente y Sustentabilidad",
    "EspecialidadesPazYParticipacion": "Paz y Acción Comunitaria",
    "EspecialidadesSaludYBienestar": "Salud y Bienestar",
}

# Patrones de texto repetitivo en las plantillas de fichas que no aportan valor semántico
BOILERPLATE_PATTERNS: list[str] = [
    r"(?m)^##\s*NOMBRE DE MI ESPECIALIDAD\s*$",
    r"(?m)^##\s*FICHA DE ESPECIALIDAD SCOUT\s*$",
    r"(?m)^OPCIONES\s*\|\s*MANADA.*$",
    r"(?m)^DESCUBRO\s+¿Qué me llama la atención\?.*$",
    r"(?m)^REFLEXIONO\s+¿Qué cosas nuevas aprendí\?.*$",
    r"(?m)^(?:#+\s*)?¿Qué más puedo aprender\?\s*$",
    r"(?m)^(?:#+\s*)?¿Qué más puedo hacer\?\s*$",
    r"(?m)^(?:#+\s*)?¿De qué otras formas puedo compartir\?\s*$",
    r"(?m)^01\s*\|\s*02\s*\|.*$",
    r"(?m)^HOJA\s*$",
    r"(?m)^0\d\s+DE\s+0\d\s*$",
    r"(?m)^DE\s*$",
]

ANEXO_SPLIT_RE = re.compile(
    r"(?m)^#\s*FICHA DE ESPECIALIDAD SCOUT ANEXO\s*$", re.IGNORECASE
)


@dataclass
class ChunkDocumento:
    """Representa un fragmento listo para generar embedding y guardar en ChromaDB."""

    chunk_id: str
    text: str
    metadata: dict[str, str | int]


def normalizar_area(nombre_crudo: str) -> str:
    """Normaliza variantes y typos de áreas hacia los 4 nombres oficiales canónicos."""
    texto = nombre_crudo.strip().lower()
    if "habilidad" in texto:
        return "Habilidades para la vida"
    if "ambiente" in texto or "sustent" in texto:
        return "Medio Ambiente y Sustentabilidad"
    if "paz" in texto or "comunitaria" in texto or "participa" in texto:
        return "Paz y Acción Comunitaria"
    if "salud" in texto or "bienestar" in texto:
        return "Salud y Bienestar"
    return nombre_crudo.strip()


def crear_id_estable(prefijo: str, clave: str, indice: int) -> str:
    """Genera un ID determinista y seguro para ChromaDB (idempotente en re-ingestas)."""
    digest = hashlib.sha1(clave.encode("utf-8")).hexdigest()[:12]
    return f"{prefijo}_{digest}_{indice}"


def dividir_por_parrafos(
    texto: str, max_chars: int = MAX_CHUNK_CHARS, overlap: int = CHUNK_OVERLAP_CHARS
) -> list[str]:
    """Divide texto largo respetando límites de párrafos con solapamiento configurable."""
    texto = texto.strip()
    if len(texto) <= max_chars:
        return [texto] if texto else []

    parrafos = [p.strip() for p in re.split(r"\n{2,}", texto) if p.strip()]
    chunks: list[str] = []
    actual: list[str] = []
    largo_actual = 0

    for parrafo in parrafos:
        # Si un solo párrafo supera max_chars, lo partimos por líneas o segmentos
        if len(parrafo) > max_chars:
            if actual:
                chunks.append("\n\n".join(actual))
                actual = []
                largo_actual = 0
            inicio = 0
            paso = max(max_chars - overlap, 1)
            while inicio < len(parrafo):
                chunks.append(parrafo[inicio : inicio + max_chars].strip())
                inicio += paso
            continue

        costo = len(parrafo) + (2 if actual else 0)
        if largo_actual + costo <= max_chars:
            actual.append(parrafo)
            largo_actual += costo
        else:
            bloque = "\n\n".join(actual)
            chunks.append(bloque)
            # Solapamiento con el cierre del bloque anterior
            cola = (
                bloque[-overlap:].strip()
                if overlap > 0 and len(bloque) > overlap
                else ""
            )
            if cola and len(cola) + 2 + len(parrafo) <= max_chars:
                actual = [cola, parrafo]
                largo_actual = len(cola) + 2 + len(parrafo)
            else:
                actual = [parrafo]
                largo_actual = len(parrafo)

    if actual:
        chunks.append("\n\n".join(actual))

    return chunks


# Normativa oficial (docs/insignias.md)


def parsear_normativa_insignias(ruta: Path) -> list[ChunkDocumento]:
    """Parsea docs/insignias.md en chunks semánticos por (rama, área)."""
    if not ruta.exists():
        raise FileNotFoundError(f"No se encontró el archivo de normativa: {ruta}")

    contenido = ruta.read_text(encoding="utf-8")
    fuente_rel = ruta.relative_to(BASE_DIR).as_posix()
    chunks: list[ChunkDocumento] = []

    # Dividir por encabezados de sección (## ...)
    secciones = re.split(r"(?m)^##\s+", contenido)
    for bloque in secciones[1:]:
        lineas = bloque.splitlines()
        titulo_seccion = lineas[0].strip()
        cuerpo = "\n".join(lineas[1:]).strip()
        titulo_lower = titulo_seccion.lower()

        if "manada" in titulo_lower:
            chunks.extend(_parsear_manada_tabla(cuerpo, fuente_rel))
        elif "tropa" in titulo_lower:
            chunks.extend(_parsear_subsecciones_area(cuerpo, "Tropa", fuente_rel))
        elif "comunidad" in titulo_lower or "clan" in titulo_lower:
            chunks.extend(
                _parsear_subsecciones_area(cuerpo, "Comunidad y Clan", fuente_rel)
            )

    return chunks


def _parsear_manada_tabla(cuerpo: str, fuente_rel: str) -> list[ChunkDocumento]:
    """Agrupa la tabla de Manada por área canónica para generar 1 chunk por área."""
    por_area: dict[str, list[str]] = {}
    for linea in cuerpo.splitlines():
        linea = linea.strip()
        if not linea.startswith("|"):
            continue
        celdas = [c.strip() for c in linea.strip("|").split("|")]
        if len(celdas) < 2:
            continue
        nombre, area_cruda = celdas[0], celdas[1]
        if nombre.lower() == "nombre" or set(nombre) <= {"-", " "}:
            continue
        area = normalizar_area(area_cruda)
        por_area.setdefault(area, []).append(nombre)

    chunks: list[ChunkDocumento] = []
    for area, nombres in por_area.items():
        lista = "\n".join(f"- {n}" for n in nombres)
        texto = (
            "[NORMATIVA OFICIAL ASMAC 2026]\n"
            f"Sección (Rama): {RAMA_DISPLAY['Manada']}\n"
            f"Área de desempeño: {area}\n\n"
            "Especialidades / categorías oficiales de esta área en Manada:\n"
            f"{lista}"
        )
        chunks.append(
            ChunkDocumento(
                chunk_id=crear_id_estable("norm", f"Manada::{area}", 0),
                text=texto,
                metadata={
                    "tipo": "normativa",
                    "rama": "Manada",
                    "area": area,
                    "especialidad": "Normativa general del área",
                    "tema_relacionado": ", ".join(nombres),
                    "fuente": fuente_rel,
                    "chunk_index": 0,
                },
            )
        )
    return chunks


def _formatear_tablas_markdown(texto: str) -> tuple[str, list[str]]:
    """Convierte tablas Markdown `| Nombre | Temas Relacionados |` en viñetas legibles."""
    salida: list[str] = []
    nombres_encontrados: list[str] = []
    en_tabla = False

    for linea in texto.splitlines():
        stripped = linea.strip()
        if stripped.startswith("|") and stripped.endswith("|"):
            celdas = [c.strip() for c in stripped.strip("|").split("|")]
            if not celdas:
                continue
            if celdas[0].lower() == "nombre" or set(celdas[0]) <= {"-", " "}:
                if not en_tabla:
                    salida.append("\nEspecialidades y temas relacionados:")
                    en_tabla = True
                continue
            if len(celdas) >= 2:
                nombres_encontrados.append(celdas[0])
                salida.append(f"- {celdas[0]}: {celdas[1]}")
            else:
                salida.append(f"- {celdas[0]}")
        else:
            en_tabla = False
            salida.append(linea)

    limpio = re.sub(r"\n{3,}", "\n\n", "\n".join(salida)).strip()
    return limpio, nombres_encontrados


def _parsear_subsecciones_area(
    cuerpo: str, rama: str, fuente_rel: str
) -> list[ChunkDocumento]:
    """Parsea bloques `### <Área>` de Tropa y Comunidad/Clan en chunks por área."""
    chunks: list[ChunkDocumento] = []
    subsecciones = re.split(r"(?m)^###\s+", cuerpo)

    for sub in subsecciones[1:]:
        lineas = sub.splitlines()
        area = normalizar_area(lineas[0].strip())
        contenido_area = "\n".join(lineas[1:]).strip()
        contenido_formateado, nombres = _formatear_tablas_markdown(contenido_area)

        partes = dividir_por_parrafos(contenido_formateado)
        for idx, parte in enumerate(partes):
            texto = (
                "[NORMATIVA OFICIAL ASMAC 2026]\n"
                f"Sección (Rama): {RAMA_DISPLAY[rama]}\n"
                f"Área de desempeño: {area}\n\n"
                f"{parte}"
            )
            chunks.append(
                ChunkDocumento(
                    chunk_id=crear_id_estable("norm", f"{rama}::{area}", idx),
                    text=texto,
                    metadata={
                        "tipo": "normativa",
                        "rama": rama,
                        "area": area,
                        "especialidad": "Normativa general del área",
                        "tema_relacionado": ", ".join(nombres),
                        "fuente": fuente_rel,
                        "chunk_index": idx,
                    },
                )
            )

    return chunks


# Fichas de especialidades de referencia (data/normativa/**/*.md)


def _extraer_nombre_especialidad(lineas_no_vacias: list[str], ruta: Path) -> str:
    """Extrae el nombre de la especialidad del contenido de la ficha."""
    for i, linea in enumerate(lineas_no_vacias[:10]):
        if "NOMBRE DE MI ESPECIALIDAD" in linea.upper():
            if i + 1 < len(lineas_no_vacias) and not lineas_no_vacias[i + 1].startswith(
                "#"
            ):
                return lineas_no_vacias[i + 1].strip(" .")

    for linea in lineas_no_vacias[:15]:
        if linea.startswith("## "):
            upper = linea.upper()
            if not any(
                k in upper for k in ("NOMBRE DE MI", "FICHA DE", "MEDIDAS DE", "ODS")
            ):
                return linea.lstrip("# ").strip(" .")

    # Respaldo usando el nombre del archivo
    return ruta.stem.replace("_", " ").strip()


def _extraer_tema_relacionado(lineas_no_vacias: list[str]) -> str:
    """Extrae el campo TEMA RELACIONADO de la ficha, si está presente."""
    for i, linea in enumerate(lineas_no_vacias):
        if linea.upper() == "TEMA RELACIONADO" and i + 1 < len(lineas_no_vacias):
            candidato = lineas_no_vacias[i + 1].strip()
            if not candidato.upper().startswith(("ODS", "##", "#")):
                return candidato.rstrip(".")
            break
    return ""


def _limpiar_texto_ficha(contenido: str) -> str:
    """Elimina boilerplate de plantilla y artefactos de escape de PDF-a-Markdown."""
    texto = contenido
    for patron in BOILERPLATE_PATTERNS:
        texto = re.sub(patron, "", texto, flags=re.IGNORECASE)

    # Eliminar etiquetas redundantes de cabecera (se agregan normalizadas en el encabezado)
    texto = re.sub(
        r"(?m)^SECCIÓN\s*\n+\s*(?:MANADA|TROPA|COMUNIDAD|CLAN)[^\n]*\n?",
        "",
        texto,
        flags=re.IGNORECASE,
    )
    texto = re.sub(
        r"(?m)^EJE TEMÁTICO DE LA ESPECIALIDAD\s*\n+(?:[^\n#]+\n)?",
        "",
        texto,
        flags=re.IGNORECASE,
    )

    # Limpiar escapes de Markdown generados al convertir PDF
    texto = texto.replace(r"\*", "*").replace(r"\-", "-")
    texto = re.sub(r"(?m)^-\s+\*\s+", "- ", texto)
    texto = re.sub(r"(?m)^\*\s+\*\s+", "- ", texto)
    texto = re.sub(r"(?m)^-\s+-\s+", "- ", texto)
    texto = re.sub(r"\n{3,}", "\n\n", texto)
    return texto.strip()


def parsear_fichas_especialidades(directorio: Path) -> list[ChunkDocumento]:
    """Recorre data/normativa/<rama>/<area>/*.md y genera los chunks con metadatos."""
    if not directorio.exists():
        raise FileNotFoundError(
            f"No se encontró el directorio de especialidades: {directorio}"
        )

    chunks: list[ChunkDocumento] = []
    archivos = sorted(directorio.rglob("*.md"))

    for archivo in archivos:
        rel_desde_data = archivo.relative_to(directorio)
        if len(rel_desde_data.parts) < 3:
            continue

        carpeta_rama = rel_desde_data.parts[0].lower()
        carpeta_area = rel_desde_data.parts[1]
        rama = RAMA_CANONICA.get(carpeta_rama, carpeta_rama.capitalize())
        area = AREA_POR_CARPETA.get(carpeta_area, normalizar_area(carpeta_area))

        raw = archivo.read_text(encoding="utf-8")
        lineas_no_vacias = [l.strip() for l in raw.splitlines() if l.strip()]
        especialidad = _extraer_nombre_especialidad(lineas_no_vacias, archivo)
        tema = _extraer_tema_relacionado(lineas_no_vacias)
        fuente_rel = archivo.relative_to(BASE_DIR).as_posix()

        # Separar ficha principal y anexo (si existe) para no mezclarlos ni truncar la ficha
        secciones_doc = ANEXO_SPLIT_RE.split(raw)
        fragmentos_cuerpo: list[str] = []
        for idx_sec, sec in enumerate(secciones_doc):
            limpio = _limpiar_texto_ficha(sec)
            if not limpio:
                continue
            if idx_sec > 0:
                limpio = f"ANEXO DE LA ESPECIALIDAD:\n{limpio}"
            fragmentos_cuerpo.extend(dividir_por_parrafos(limpio))

        encabezado_lineas = [
            "[ESPECIALIDAD DE REFERENCIA ASMAC 2026]",
            f"Sección (Rama): {RAMA_DISPLAY.get(rama, rama)}",
            f"Área de desempeño: {area}",
            f"Especialidad: {especialidad}",
        ]
        if tema:
            encabezado_lineas.append(f"Tema relacionado: {tema}")
        encabezado = "\n".join(encabezado_lineas)

        for idx, cuerpo in enumerate(fragmentos_cuerpo):
            texto_final = f"{encabezado}\n\n{cuerpo}"
            chunks.append(
                ChunkDocumento(
                    chunk_id=crear_id_estable("esp", fuente_rel, idx),
                    text=texto_final,
                    metadata={
                        "tipo": "especialidad_ejemplo",
                        "rama": rama,
                        "area": area,
                        "especialidad": especialidad,
                        "tema_relacionado": tema,
                        "fuente": fuente_rel,
                        "chunk_index": idx,
                    },
                )
            )

    return chunks


# Embeddings


def generar_embeddings_en_lotes(
    client: Any,
    textos: list[str],
    batch_size: int = BATCH_SIZE,
) -> list[list[float]]:
    """Genera embeddings en lotes con reintentos exponenciales ante límite 429."""
    from google.genai import types

    todos_los_vectores: list[list[float]] = []
    total_lotes = (len(textos) + batch_size - 1) // batch_size

    for num_lote, inicio in enumerate(range(0, len(textos), batch_size), start=1):
        lote = textos[inicio : inicio + batch_size]
        print(
            f"  -> Generando embeddings lote {num_lote}/{total_lotes} ({len(lote)} chunks)..."
        )

        for intento in range(1, MAX_RETRIES + 1):
            try:
                respuesta = client.models.embed_content(
                    model=EMBEDDING_MODEL,
                    contents=lote,
                    config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
                )
                if not respuesta.embeddings or len(respuesta.embeddings) != len(lote):
                    raise RuntimeError(
                        "Cantidad de embeddings devuelta no coincide con el lote enviado."
                    )
                todos_los_vectores.extend(emb.values for emb in respuesta.embeddings)
                break
            except Exception as exc:
                msg = str(exc)
                es_rate_limit = "429" in msg or "RESOURCE_EXHAUSTED" in msg
                if es_rate_limit and intento < MAX_RETRIES:
                    espera = 2**intento * 5
                    print(
                        f"     Límite de peticiones (429). Reintentando en {espera}s ({intento}/{MAX_RETRIES})..."
                    )
                    time.sleep(espera)
                else:
                    raise

        if num_lote < total_lotes:
            time.sleep(BATCH_DELAY_SECONDS)

    return todos_los_vectores


def guardar_en_chromadb(
    chunks: list[ChunkDocumento],
    embeddings: list[list[float]],
    reset: bool = False,
) -> int:
    """Guarda los chunks y sus vectores en ChromaDB con similitud coseno."""
    import chromadb

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    if reset:
        existentes = [c.name for c in chroma_client.list_collections()]
        if COLLECTION_NAME in existentes:
            chroma_client.delete_collection(name=COLLECTION_NAME)
            print(
                f"  -> Colección '{COLLECTION_NAME}' eliminada para re-creación limpia."
            )

    collection = chroma_client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    collection.upsert(
        ids=[c.chunk_id for c in chunks],
        documents=[c.text for c in chunks],
        metadatas=[c.metadata for c in chunks],
        embeddings=embeddings,
    )
    return collection.count()


def recolectar_chunks() -> list[ChunkDocumento]:
    """Carga y prepara todos los chunks de normativa y especialidades."""
    chunks_normativa = parsear_normativa_insignias(NORMATIVA_MD)
    chunks_especialidades = parsear_fichas_especialidades(DATA_DIR)
    return chunks_normativa + chunks_especialidades


def imprimir_resumen(chunks: list[ChunkDocumento]) -> None:
    """Muestra un resumen claro de los chunks generados por tipo, rama y área."""
    norm = [c for c in chunks if c.metadata["tipo"] == "normativa"]
    esp = [c for c in chunks if c.metadata["tipo"] == "especialidad_ejemplo"]
    largos = [len(c.text) for c in chunks]

    print("=== Resumen de documentos preparados ===")
    print(f"Total de chunks: {len(chunks)}")
    print(f"  - Normativa oficial ({NORMATIVA_MD.name}): {len(norm)} chunks")
    print(f"  - Especialidades de referencia: {len(esp)} chunks")
    if largos:
        print(
            f"  - Longitud (caracteres): min={min(largos)}, promedio={sum(largos) // len(largos)}, max={max(largos)}"
        )

    conteo_rama_area: dict[tuple[str, str], int] = {}
    for c in chunks:
        clave = (str(c.metadata["rama"]), str(c.metadata["area"]))
        conteo_rama_area[clave] = conteo_rama_area.get(clave, 0) + 1

    print("\nDistribución por (Rama, Área):")
    for (rama, area), total in sorted(conteo_rama_area.items()):
        print(f"  - {rama:16s} | {area:34s}: {total}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingesta de documentos ASMAC 2026 hacia ChromaDB."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Ejecuta lectura, limpieza y chunking sin llamar a Gemini ni escribir en ChromaDB.",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Elimina la colección existente en chroma_db/ antes de ingestar.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help=f"Tamaño de lote para embeddings (default: {BATCH_SIZE}).",
    )
    args = parser.parse_args()

    chunks = recolectar_chunks()
    imprimir_resumen(chunks)

    if args.dry_run:
        print("\n[Modo --dry-run] Ejemplo de chunk de normativa:\n")
        print(chunks[0].text[:600])
        print("\n[Modo --dry-run] Ejemplo de chunk de especialidad:\n")
        ejemplo_esp = next(
            c for c in chunks if c.metadata["tipo"] == "especialidad_ejemplo"
        )
        print(ejemplo_esp.text[:600])
        print("\nDry-run completado sin errores.")
        return

    from dotenv import load_dotenv
    from google import genai

    load_dotenv(BASE_DIR / ".env")
    api_key = os.getenv("API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "No se encontró API_KEY (ni GEMINI_API_KEY) en el archivo .env"
        )

    print(
        f"\nGenerando embeddings con '{EMBEDDING_MODEL}' (lotes de {args.batch_size})..."
    )
    client = genai.Client(api_key=api_key)
    embeddings = generar_embeddings_en_lotes(
        client=client,
        textos=[c.text for c in chunks],
        batch_size=args.batch_size,
    )

    print(f"\nGuardando en ChromaDB ({CHROMA_DIR})...")
    total_en_db = guardar_en_chromadb(chunks, embeddings, reset=args.reset)
    print(
        f"Ingesta completada con éxito. Total de vectores en '{COLLECTION_NAME}': {total_en_db}"
    )


if __name__ == "__main__":
    main()
