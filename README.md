# RAG Scouts — Asistente de Especialidades ASMAC 2026

Sistema de **Generación Aumentada por Recuperación (RAG)** conectado a una interfaz web tipo chatbot para orientar a dirigentes y jóvenes sobre el sistema nacional de especialidades de la **Asociación de Scouts de México (ASMAC)** implementado a partir de **2026**.

---

## Contexto y Problemática

La **ASMAC** es un movimiento educativo sin fines de lucro enfocado en el desarrollo continuo de jóvenes de **6 a 22 años**. Una de sus herramientas principales son las **especialidades** (insignias que el Scout porta en su uniforme al completar actividades de su interés).

En **2026** se implementó un cambio nacional en la logística y clasificación de las especialidades, ampliando considerablemente la variedad de áreas de desempeño para lograr mayor especificidad. Sin embargo:

* La capacitación a los adultos responsables suele concentrarse en pláticas cortas donde es difícil asimilar toda la normativa nueva.
* Las dudas sobre a qué área pertenece una actividad retrasan el inicio de la documentación necesaria para que los jóvenes obtengan su insignia.

---

## Propuesta

Desarrollar una aplicación web con **interfaz conversacional (GUI estilo Gemini / ChatGPT)** respaldada por un **sistema RAG** conectado a un **LLM (Gemini)**.

El usuario describe una palabra, acción o proyecto que el Scout desea realizar y el sistema:

1. **Identifica y clasifica** automáticamente a qué **área de desempeño** (y especialidad) corresponde.
2. **Justifica la respuesta** basándose únicamente en los manuales y lineamientos oficiales de ASMAC 2026 almacenados en la base vectorial, evitando alucinaciones.

### Estrategia de Conocimiento Dual (Normativa + Datos Sintéticos)

El sistema combina dos fuentes dentro de la base vectorial (**ChromaDB**), diferenciadas mediante **metadatos** (`tipo: "normativa"` vs. `tipo: "sintetico"`, `area`, `especialidad`, `rama`):

* **Normativa oficial:** Manuales, esquemas y lineamientos de ASMAC 2026 que aportan el sustento formal para las respuestas.
* **Base de datos sintética:** Conjunto amplio de descripciones cotidianas de acciones y proyectos típicos de Scouts mapeados a su área correspondiente. Esto sirve como puente semántico cuando el usuario utiliza términos coloquiales que no aparecen textualmente en el reglamento.

---

## Flujo del Sistema

### 1. Fase de Ingesta (Offline / Preparación)
```text
[ Manuales ASMAC 2026 ] + [ Datos Sintéticos ]
                   │
                   ▼ (Limpieza + Chunking + Metadatos)
         [ Embeddings de Gemini ] (en lotes / batches)
                   │
                   ▼
          [ ChromaDB en disco ]
```

### 2. Fase de Consulta (Tiempo Real)
```text
[ Página Web (Chatbot UI) ]
           │  POST /chat
           ▼
[ Backend API (FastAPI) ]
      │                   │
      ▼ (Top-K similitud) ▼ (Prompt + Contexto recuperado)
[ ChromaDB ]         [ Gemini LLM ]
                          │
                          ▼
             Respuesta clasificada y justificada
```

---

## Stack Tecnológico

* **Lenguaje:** Python 3.10+
* **LLM:** Google Gemini (`gemini-3.1-flash-lite` vía SDK oficial `google-genai`)
* **Embeddings:** Modelos de embedding de Gemini (`gemini-embedding-001`)
* **Base de Datos Vectorial:** ChromaDB (persistencia local en disco)
* **Backend / API:** FastAPI + Uvicorn
* **Frontend:** HTML5, CSS3 y JavaScript (`fetch` hacia la API)
* **Gestión de entorno:** `python-dotenv`

---

## Estructura del Proyecto

```text
RAG_Scouts/
├── data/
│   ├── normativa/       # PDFs, manuales o textos oficiales de ASMAC 2026
│   └── sinteticos/      # CSV/JSON con acciones de ejemplo y su área correspondiente
├── chroma_db/           # Base vectorial persistente en disco (generada en la ingesta)
├── src/
│   ├── ingest.py        # Lee data/, limpia, genera embeddings en lotes y guarda en ChromaDB
│   ├── rag.py           # Lógica RAG: consulta -> busca en ChromaDB -> genera respuesta con Gemini
│   └── api.py           # Servidor FastAPI con POST /chat (y CORS) para el frontend
├── frontend/
│   ├── index.html       # Interfaz tipo chat (estilo Gemini/ChatGPT)
│   ├── style.css        # Estilos de la interfaz conversacional
│   └── app.js           # Comunicación con FastAPI mediante fetch()
├── chat_basico.py       # Script inicial de prueba de conexión con Gemini
├── RAG_Scouts.md        # Documento base con contexto, problemática, propuesta y metodología
├── .env                 # Variables de entorno (API_KEY de Gemini — no subir a Git)
├── .gitignore           # Exclusión de .env, entornos virtuales y archivos temporales
├── requirements.txt     # Dependencias del proyecto
└── README.md            # Documentación principal del proyecto
```

> **Nota:** Las carpetas `data/`, `chroma_db/`, `src/` y `frontend/` se irán creando conforme avance la implementación del sistema.

---

## Configuración Inicial (Entorno Local)

1. **Crear y activar un entorno virtual:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate   # En Linux/macOS
   # .venv\Scripts\activate    # En Windows
   ```

2. **Instalar dependencias base:**
   ```bash
   pip install google-genai python-dotenv
   ```
   *(Más adelante se añadirán `chromadb`, `fastapi`, `uvicorn` y `pypdf` en `requirements.txt`.)*

3. **Configurar la API Key:**
   Asegúrate de tener tu clave de Google AI Studio en el archivo `.env`:
   ```env
   API_KEY=tu_api_key_aqui
   ```

4. **Probar conexión básica con Gemini:**
   ```bash
   python chat_basico.py
   ```

---

## Consideraciones del Nivel Gratuito (Free Tier)

* **Límites de peticiones (RPM / RPD):** Al ingestar la base sintética y los manuales, agrupa los fragmentos en **lotes (*batches*)** para evitar errores `429 RESOURCE_EXHAUSTED` y persiste los vectores en `chroma_db/` para no recalcularlos.
* **Privacidad:** En el nivel gratuito de Google AI Studio los datos enviados pueden utilizarse para mejorar el servicio; evita incluir datos personales reales o información confidencial durante las pruebas.

---

## Resultados Esperados

* Clasificación ágil y precisa de acciones o proyectos de Scouts hacia las áreas de desempeño de **ASMAC 2026**.
* Reducción significativa del tiempo que toma a dirigentes y jóvenes localizar la especialidad adecuada e iniciar su documentación.
* Respuestas claras, comprensibles y estrictamente fundamentadas en la documentación oficial.
