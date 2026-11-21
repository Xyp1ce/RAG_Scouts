# RAG Scouts — Asistente de Especialidades ASMAC 2026

Sistema de **Generación Aumentada por Recuperación (RAG)** conectado a una interfaz web tipo chatbot para orientar a dirigentes y jóvenes sobre el sistema nacional de especialidades de la **Asociación de Scouts de México (ASMAC)** implementado a partir de **2026**.

---

## Contexto y Problemática

La **ASMAC** es un movimiento educativo sin fines de lucro enfocado en el desarrollo continuo de jóvenes de **6 a 22 años**. Una de sus herramientas principales son las **especialidades** (insignias que el Scout porta en su uniforme al completar actividades de su interés).

En **2026** se implementó un cambio nacional en la logística y clasificación de las especialidades, reorganizándolas en **4 áreas de desempeño**:
* **Habilidades para la vida**
* **Medio Ambiente y Sustentabilidad**
* **Paz y Acción Comunitaria**
* **Salud y Bienestar**

Sin embargo:
* La capacitación a los adultos responsables suele concentrarse en pláticas cortas donde es difícil asimilar toda la normativa nueva.
* Las dudas sobre a qué área pertenece una actividad retrasan el inicio de la documentación necesaria para que los jóvenes obtengan su insignia.

---

## Propuesta

Una aplicación web con **interfaz conversacional (GUI estilo Gemini / ChatGPT)** respaldada por un **sistema RAG** conectado a **Google Gemini**.

El usuario describe una palabra clave, acción o proyecto que el Scout desea realizar y el sistema:

1. **Identifica y clasifica** automáticamente a qué **área de desempeño** (y categoría/tema relacionado) corresponde.
2. **Justifica la respuesta** basándose únicamente en la normativa y fichas oficiales de ASMAC 2026 almacenadas en la base vectorial, evitando alucinaciones.

### Estrategia de Conocimiento Dual (Normativa + Fichas de Referencia)

El sistema combina dos fuentes dentro de la base vectorial (**ChromaDB**), diferenciadas mediante **metadatos** (`tipo: "normativa"` vs. `tipo: "especialidad_ejemplo"`, `rama`, `area`, `especialidad`, `tema_relacionado`, `fuente`):

* **Normativa oficial (`docs/insignias.md`):** Lineamientos, objetivos y temas relacionados por sección y área de ASMAC 2026 que aportan el sustento formal de cada clasificación.
* **Fichas de especialidades de referencia (`data/normativa/`):** Conjunto de fichas reales de especialidades divididas en las **4 secciones** (*Manada*, *Tropa*, *Comunidad* y *Clan*) y mapeadas a sus **4 áreas de desempeño**. Funcionan como puente semántico cuando el usuario describe una actividad con términos coloquiales que no aparecen textualmente en la normativa general.

---

## Flujo del Sistema

### 1. Fase de Ingesta (Offline / Preparación)
```text
[ docs/insignias.md ] + [ data/normativa/<rama>/<area>/*.md ]
                       │
                       ▼  src/ingesta.py (Limpieza + Chunking + Metadatos)
             [ Embeddings de Gemini ] (gemini-embedding-001 en lotes)
                       │
                       ▼
          [ ChromaDB en disco (chroma_db/) ] (similitud coseno)
```

### 2. Fase de Consulta (Tiempo Real)
```text
[ Página Web (Chatbot UI) ]
           │  POST /chat/stream  (o POST /chat)
           ▼
[ Backend API (src/api.py) + Motor RAG (src/rag.py) ]
      │                                     │
      ▼ (Recuperación dual:                 ▼ (Prompt aumentado +
         Normativa + Fichas Top-K)             Contexto recuperado)
[ ChromaDB ]                          [ Gemini LLM ]
                                            │
                                            ▼
                         Respuesta clasificada y justificada
```

---

## Stack Tecnológico

* **Lenguaje:** Python 3.10+
* **LLM:** Google Gemini (`gemini-3.1-flash-lite` vía SDK oficial `google-genai`)
* **Embeddings:** `gemini-embedding-001` (`RETRIEVAL_DOCUMENT` en ingesta / `RETRIEVAL_QUERY` en consulta)
* **Base de Datos Vectorial:** ChromaDB (persistencia local en disco con distancia coseno)
* **Backend / API:** FastAPI + Uvicorn
* **Frontend:** HTML5, CSS3 y JavaScript (`fetch` + *streaming*)
* **Gestión de entorno:** `python-dotenv`

---

## Estructura del Proyecto

```text
RAG_Scouts/
├── data/
│   └── normativa/
│       ├── manada/       # Fichas .md organizadas en las 4 áreas de desempeño
│       ├── tropa/        # Fichas .md organizadas en las 4 áreas de desempeño
│       ├── comunidad/    # Fichas .md organizadas en las 4 áreas de desempeño
│       └── clan/         # Fichas .md organizadas en las 4 áreas de desempeño
├── docs/
│   ├── insignias.md                 # Normativa oficial consolidada de áreas y temas ASMAC 2026
│   ├── RAG_Scouts.md                # Documento base con contexto, problemática y propuesta
│   ├── Insiginias nuevo esquema.pdf # Material oficial de referencia
│   └── NUEVAS INSIGINIAS 2025.pdf   # Material oficial de referencia
├── chroma_db/            # Base vectorial persistente en disco (generada por src/ingesta.py)
├── src/
│   ├── ingesta.py        # Limpia, fragmenta, genera embeddings en lotes y guarda en ChromaDB
│   ├── rag.py            # Recuperación dual en ChromaDB + prompt fundamentado + generación con Gemini
│   └── api.py            # Servidor FastAPI (GET /health, POST /chat, POST /chat/stream)
├── frontend/
│   ├── index.html        # Interfaz tipo chat (estilo Gemini/ChatGPT)
│   ├── style.css         # Estilos de la interfaz conversacional
│   └── app.js            # Comunicación con FastAPI mediante fetch() con streaming
├── chat_basico.py        # Script inicial de prueba de conexión con Gemini
├── .env                  # Variables de entorno (API_KEY de Gemini — no subir a Git)
├── .gitignore            # Exclusión de .env, .venv/, chroma_db/ y archivos temporales
├── requirements.txt      # Dependencias del proyecto
└── README.md             # Documentación principal del proyecto
```

---

## Configuración Inicial (Entorno Local)

1. **Crear y activar un entorno virtual:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate   # En Linux/macOS
   # .venv\Scripts\activate    # En Windows
   ```

2. **Instalar dependencias:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configurar la API Key:**
   Crea un archivo `.env` en la raíz del proyecto con tu clave de Google AI Studio:
   ```env
   API_KEY=tu_api_key_aqui
   ```

4. **Generar la base vectorial (Ingesta):**
   ```bash
   # Opcional: verificar limpieza y conteo de chunks sin consumir cuota de API
   python -m src.ingesta --dry-run

   # Ejecutar la ingesta y poblar chroma_db/
   python -m src.ingesta --reset
   ```

5. **Probar el motor RAG desde la terminal (opcional):**
   ```bash
   # Solo inspección de fragmentos recuperados en ChromaDB
   python -m src.rag --solo-buscar "actividad de primeros auxilios en campamento"

   # Consulta completa con respuesta de Gemini
   python -m src.rag "Un Scout quiere diseñar una app móvil para su comunidad"

   # Filtrando explícitamente por una sección
   python -m src.rag --rama Tropa "jornada de reforestación y reciclaje"
   ```

---

## Levantar el Servidor y la Interfaz Web

1. **Iniciar la API (Backend con FastAPI + Uvicorn)** desde la raíz del proyecto (con el entorno virtual activado):
   ```bash
   uvicorn src.api:app --reload
   ```
   El servidor quedará disponible en `http://127.0.0.1:8000` con los endpoints:
   * `GET /health` — Verificación de estado del servidor y número de chunks indexados en ChromaDB.
   * `POST /chat` — Respuesta completa del asistente junto con la rama detectada y las fuentes consultadas (acepta `message` y opcionalmente `rama` e `history`).
   * `POST /chat/stream` — Respuesta en tiempo real (*streaming*), utilizada por el frontend.

2. **Abrir la interfaz del chatbot (Frontend):**
   * **Opción A (servidor local recomendado):** en otra terminal, ejecuta:
     ```bash
     python3 -m http.server 5500 --directory frontend
     ```
     y abre `http://localhost:5500` en tu navegador.
   * **Opción B:** abre directamente el archivo `frontend/index.html` en tu navegador.

---

## Consideraciones del Nivel Gratuito (Free Tier)

* **Límites de peticiones (RPM / RPD):** `src/ingesta.py` agrupa los fragmentos en **lotes (*batches*)** con pausas y reintentos automáticos ante errores `429 RESOURCE_EXHAUSTED`, y persiste los vectores en `chroma_db/` para no recalcularlos en cada arranque.
* **Privacidad:** En el nivel gratuito de Google AI Studio los datos enviados pueden utilizarse para mejorar el servicio; evita incluir datos personales reales o información confidencial durante las pruebas.

---

## Resultados Esperados

* Clasificación ágil y precisa de acciones o proyectos de Scouts hacia las **4 áreas de desempeño de ASMAC 2026**.
* Reducción significativa del tiempo que toma a dirigentes y jóvenes localizar la especialidad adecuada e iniciar su documentación.
* Respuestas claras, comprensibles y estrictamente fundamentadas en la documentación oficial.
