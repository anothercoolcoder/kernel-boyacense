# Informe de Análisis Técnico y Matriz de TODOs — CODEFEST AD ASTRA 2026 (Etapa 1)

> **Fecha:** 1 de agosto de 2026  
> **Proyecto:** `kernel-boyacense` (Base de Conocimiento Vectorial RAG)  
> **Referencia Térmica:** Especificación Técnica `context/CODEFEST_2026-1.pdf`  
> **Estado de la Evaluación de Metadata (§3.4):** INCOMPLETA / CON BUGS CRÍTICOS DETECTADOS

---

## 1. Contexto General del Reto (§1 - §11)

La Etapa 1 del **CODEFEST AD ASTRA 2026** requiere la construcción y validación de una **Base de Conocimiento Vectorial** de alto rendimiento a partir de insumos multimodales (PDF, HTML, MD, JSON, CSV, XLSX, Imágenes, PBF/OSM) distribuidos en tres fenómenos estratégicos:

1. **Fenómeno 1:** Inteligencia artificial e innovación en entornos militares.
2. **Fenómeno 2:** Seguridad espacial y órbita baja terrestre (LEO / _space debris_).
3. **Fenómeno 3:** Dinámicas territoriales en América Latina y el Caribe.

### Reglas de Negocio y Restricciones Operativas Clave:

- **Sin Modelos Generativos en Recuperación (§8.3):** Prohibición **absoluta** de usar decoders (GPT, LLaMA, Gemini, Claude) para _reranking_, reformulación de queries, filtrado o síntesis de fragmentos. La recuperación debe ser 100% determinista/vectorial basada en similitud coseno y metadata.
- **Completitud Lingüística (§3.3):** Ningún chunk puede cortar oraciones. Los cortes deben realizarse en límites oracionales completos.
- **Granularidad de Salida (§9):** Para cada una de las 50 consultas de evaluación (`q001` - `q050`), se deben retornar **exactamente 3 documentos** (`F1@3`) y **exactamente 10 fragmentos** de máximo 250 palabras (`NDCG@10`).
- **Formato Entregable (§9.3):** Archivo `resultados.jsonl` de exactamente 50 líneas JSON válidas siguiendo el esquema estricto de la evaluación.
- **Leaderboard Unificado (§11.2):** Puntuación combinada de NDCG@10 (fragmentos) y F1@3 (documentos) mediante el método de **Conteo de Borda**.

---

## 2. Auditoría Específica de Metadata (Punto 3.4 del PDF vs. Código Actual)

Revisión exhaustiva de la **Tabla 1 (§3.4)** frente a la implementación en `extraccion/extraccion.py`, `extraccion/fragmentacion.py` e `indexar/indexar.py`:

| Campo (§3.4) | Tipo     | Descripción Requerida                                        | Estado en Código Actual | Diagnóstico y Acción Requerida                                                                                                                                                                                                             |
| ------------ | -------- | ------------------------------------------------------------ | ----------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `doc_id`     | `cadena` | Identificador único del documento de origen.                 | **IMPLEMENTADO**        | Mapeado persistentemente en `doc_id_registry.json` con formato `DOC-XXXX`. Correcto.                                                                                                                                                       |
| `chunk_id`   | `cadena` | Identificador único del fragmento dentro del documento.      | **BUG CRÍTICO**         | Generado como `{doc_id}-chunk-{posicion:03d}` en `fragmentacion.py` (l. 297). Sin embargo, `posicion` se reinicia a 0 por cada `registro` (página de PDF o fila de CSV), produciendo **`chunk_id` duplicados** dentro del mismo documento. |
| `fuente`     | `cadena` | Nombre o URL del archivo original provisto por ADL.          | **INCOMPATIBLE**        | Actualmente almacena la ruta absoluta local en disco (`str(ruta.resolve())`), ej. `/home/jose/...`. El ground truth empareja por el nombre/ruta relativa original (`ruta.name` o relativo a ADL). Fallará la evaluación `F1@3`.            |
| `formato`    | `cadena` | Formato del archivo (`pdf`, `html`, `md`).                   | **DESALINEADO**         | Asigna `"markdown"` en lugar del estándar `"md"`, e incluye cadenas no normalizadas como `"imagen"`, `"pbf"`, `"json"`, etc. Debe normalizarse a los valores canónicos.                                                                    |
| `fenomeno`   | `entero` | Fenómeno temático (1, 2 o 3).                                | **INCOMPLETO**          | `main.py` (l. 44) lo infiere si la carpeta contenedora es `fenomeno_1`, `fenomeno_2` o `fenomeno_3`. En `corpus_adl/data/` actual queda como `None`, violando la obligatoriedad.                                                           |
| `posicion`   | `entero` | Índice ordinal del fragmento en el documento (empieza en 0). | **BUG CRÍTICO**         | Reiniciado a 0 en cada página/sección en lugar de ser un contador global continuo por documento.                                                                                                                                           |
| `num_tokens` | `entero` | Número de tokens del fragmento.                              | **IMPLEMENTADO**        | Calculado correctamente usando `AutoTokenizer` de HuggingFace (`intfloat/multilingual-e5-large-instruct`).                                                                                                                                 |
| `texto`      | `cadena` | Texto original del fragmento sin modificaciones.             | **IMPLEMENTADO**        | Extraído y preservado sin decoración. Correcto.                                                                                                                                                                                            |

---

## 3. Análisis de Componentes del Proyecto

### 3.1 Módulo `extraccion/`

- **Fortalezas:** Excelente cobertura multimodal (`PDF`, `HTML`, `MD`, `TXT`, `JSON`, `CSV`, `XLSX`, `Imágenes OCR` con RapidOCR/Tesseract, `PBF/MVT`). Implementación limpia con importación perezosa de dependencias.
- **Debilidades:**
  - Al extraer un PDF, emite múltiples objetos `Registro` (uno por página), lo que descoloca la secuencia continua de fragmentos si el fragmentador no mantiene un contador de documento.
  - No resuelve la asignación de `fenomeno` si los archivos no están previamente organizados en subcarpetas con nombres fijos.

### 3.2 Módulo `fragmentacion.py`

- **Fortalezas:** Detección de idioma multilingüe (español, inglés, portugués) para seleccionar el tokenizador Punkt de NLTK (§2.2). Respeto estricto del requisito de completitud lingüística (§3.3). Ventana deslizante con solape de oraciones.
- **Fallas:**
  - **Bug de conteo de `posicion`:** Como se procesa registro por registro, en documentos multipágina `posicion` vuelve a empezar en 0 para cada página, generando ids duplicados como `DOC-0001-chunk-000` en p.1 y `DOC-0001-chunk-000` en p.2.
  - **Discrepancia en límite de palabras (§9.2.1):** Fragmenta por `MAX_TOKENS = 500`. No obstante, la especificación de entrega en el punto 9.2.1 exige que cada fragmento devuelto en `resultados.jsonl` contenga **máximo 250 palabras**. 500 tokens de e5 pueden equivaler a ~350-400 palabras, lo que causaría descalificación automática en la evaluación.

### 3.3 Módulo `indexar/` (`indexar.py`) y `faiss_index/`

- **Fortalezas:** Indexación vectorial nativa con FAISS, normalización unitaria de embeddings (`normalize_embeddings=True`) para similitud coseno exacta (`IndexFlatIP`). Detección automática de aceleración hardware (CUDA vs CPU).
- **Debilidades / Limitaciones:**
  - **Vicio por datos de entrada:** Dado que el índice actual en `faiss_index/` fue construido antes de corregir los bugs de `chunk_id` y `fuente`, el índice en disco está viciado y debe reconstruirse.
  - **Sin soporte para búsqueda híbrida:** Actualmente solo soporta embeddings densos con un único modelo (`intfloat/multilingual-e5-large-instruct`). No cuenta con un índice disperso paralelo (BM25) ni fusión de rankings (§8.4).

### 3.4 Módulos Faltantes / Componentes Ausentes

1. **Módulo de Recuperación (`recuperar/`):** Ausente. No existe código para cargar el índice FAISS, calcular similitud coseno con prefijos de consulta e5, agrupar puntuaciones a nivel de documento, ni filtrar metadata.
2. **Generador del Entregable (`resultados.jsonl`):** Ausente. No existe script para iterar sobre las 50 consultas de evaluación (`q001` - `q050`) y formatear la salida JSON Lines según el punto 9.3.
3. **Métricas de Evaluación Local:** Ausente. No existe harness para medir `NDCG@10` ni `F1@3` localmente.

---

## 4. Matriz Completa de TODOs (Específicos y Accionables)

Para llevar el proyecto a un estado listo para producción y competencia, se definen las siguientes tareas técnicas detalladas:

### FASE 1: Corrección de Bugs en Ingesta y Metadatos Obligatorios (§3.4)

- [x] **TODO 1.1: Corregir el Contador de `posicion` y `chunk_id` en `fragmentacion.py`**
  - **Ubicación:** `extraccion/fragmentacion.py` -> `fragmentar_registros()`
  - **Detalle:** Agrupar los registros por `doc_id` antes de fragmentar, o mantener un diccionario/contador acumulativo de `posicion` por `doc_id`. Garantizar que para un `doc_id` dado, `posicion` incremente monotónicamente (`0, 1, 2, ... N`) independientemente del número de páginas del documento original.

- [x] **TODO 1.2: Normalizar y Sanear el Campo `fuente` (§3.4)**
  - **Ubicación:** `extraccion/extraccion.py` y `extraccion/fragmentacion.py`
  - **Detalle:** Reemplazar la ruta absoluta local (`str(ruta.resolve())`) por la ruta relativa estandarizada respecto a la raíz del corpus provisto por ADL (o `ruta.name`), garantizando coincidencia exacta con el ground truth en la evaluación `F1@3`.

- [x] **TODO 1.3: Normalizar el Campo `formato` (§3.4)**
  - **Ubicación:** `extraccion/extraccion.py`
  - **Detalle:** Mapear extensiones y tipos a los valores canónicos exigidos por la especificación: `.md`/`.markdown` -> `"md"`, `.html`/`.htm` -> `"html"`, `.pdf` -> `"pdf"`. Para otros formatos (CSV, XLSX, JSON, etc.), establecer una convención limpia y documentada (`csv`, `xlsx`, `json`, `txt`).

- [x] **TODO 1.4: Inferencia y Asignación Garantizada del Campo `fenomeno` (§3.4)**
  - **Ubicación:** `main.py` -> `_inferir_fenomeno()` y `extraccion/fragmentacion.py`
  - **Detalle:** Asegurar que ningún fragmento quede con `fenomeno = None`. Implementar fallback por palabras clave del contenido del documento o metadatos si el archivo no está en una carpeta `fenomeno_X`.

---

### FASE 2: Módulo de Recuperación y Agregación Semántica (`recuperar/`) (§8)

- [ ] **TODO 2.1: Crear la Arquitectura del Módulo `recuperar/recuperar.py`**
  - **Ubicación:** Nuevo directorio y archivo `recuperar/recuperar.py`
  - **Detalle:** Implementar la clase/función de búsqueda `BuscadorVectorial` que cargue el índice FAISS y el almacén de metadata persistido.

- [ ] **TODO 2.2: Formateo de Consultas para Modelo e5**
  - **Ubicación:** `recuperar/recuperar.py`
  - **Detalle:** Aplicar el prefijo de instrucción obligatorio de `multilingual-e5-large-instruct` a las consultas en lenguaje natural:
    `"Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: {pregunta}"`.

- [ ] **TODO 2.3: Implementar Agregación Documental para `F1@3` (§8.6)**
  - **Ubicación:** `recuperar/recuperar.py` -> `obtener_top_documentos()`
  - **Detalle:** Recuperar los top-$K$ chunks ($K \ge 50$), agrupar por `doc_id` y calcular la puntuación agregada por documento mediante _Max Pooling_ (score del mejor fragmento del doc) o _Sum Pooling_. Retornar la lista ordenada de los **3 mejores `doc_id`**.

- [ ] **TODO 2.4: Implementar Control y División de Chunks a 250 Palabras (§9.2.1)**
  - **Ubicación:** `recuperar/recuperar.py` -> `obtener_top_fragmentos()`
  - **Detalle:** Garantizar que ningún fragmento entregado supere las 250 palabras. Si un chunk del índice excede 250 palabras, recortarlo respetando oraciones completas (§3.3), asignando el mismo `chunk_id` a los sub-fragmentos según la norma §9.2.1. Retornar exactamente **10 fragmentos**.

---

### FASE 3: Generador de Entregable de Competencia (`generar_entregable.py`) (§9.3)

- [ ] **TODO 3.1: Implementar Script de Inferencia Masiva para `resultados.jsonl`**
  - **Ubicación:** Nuevo archivo `generar_entregable.py` en la raíz
  - **Detalle:** Leer el archivo de consultas `queries.jsonl` (o `queries.json`), ejecutar el pipeline de recuperación para cada `query_id` (`q001` a `q050`), y estructurar la respuesta JSON.

- [ ] **TODO 3.2: Validación Estricta del Esquema JSON Lines (§9.3.1 - Tabla 2)**
  - **Ubicación:** `generar_entregable.py` -> `validar_esquema_salida()`
  - **Detalle:** Agregar aserciones automáticas antes de guardar:
    - Archivo de exactamente 50 líneas JSONL.
    - Cada objeto con `query_id` (`q001`..`q050`).
    - Array `documents` de exactamente 3 objetos (`rank`: 1..3, `doc_id`).
    - Array `fragments` de exactamente 10 objetos (`rank`: 1..10, `chunk_id`, `doc_id`, `text`).
    - Ningún `text` > 250 palabras (`len(text.split()) <= 250`).

---

### FASE 4: Recuperación Híbrida Avanzada y Ensembles (§4.4, §8.4) [Mejoras de Rendimiento]

- [ ] **TODO 4.1: Implementar Índice Disperso BM25 Paralelo**
  - **Ubicación:** `indexar/indexar_bm25.py` o módulo dentro de `indexar/`
  - **Detalle:** Construir un índice BM25 (usando `rank_bm25` o `bm25s`) sobre los mismos fragmentos para capturar coincidencias exactas de términos técnicos, nombres de satélites o siglas militares.

- [ ] **TODO 4.2: Implementar Fusionador RRF (Reciprocal Rank Fusion) (§8.4)**
  - **Ubicación:** `recuperar/fusion.py`
  - **Detalle:** Combinar los rankings del índice vectorial FAISS y el índice disperso BM25 aplicando la fórmula RRF:
    $$s_{RRF}(c) = \sum_{j=1}^{m} \frac{1}{k_0 + r_j(c)}$$ con $k_0 = 60$. Operación 100% libre de modelos generativos.

---

### FASE 5: Limpieza y Mantenimiento del Repositorio

- [ ] **TODO 5.1: Depurar Archivos Obsoletos**
  - **Ubicación:** `extraccion/test_pipeline.py`
  - **Detalle:** Eliminar `extraccion/test_pipeline.py` (remanente desactualizado) para dejar `test_pipeline.py` en la raíz como la única suite canónica.
- [ ] **TODO 5.2: Re-indexar el Corpus Completo**
  - **Ubicación:** `faiss_index/`
  - **Detalle:** Una vez aplicados los TODOs 1.1 al 1.4, volver a ejecutar `python main.py` para regenerar el índice FAISS libre de vicios en metadata.

---

## 5. Resumen de Próximos Pasos Prioritarios

1. **Prioridad 1:** Aplicar correcciones a `extraccion/fragmentacion.py` (`posicion`, `chunk_id`, `fuente`, `formato`).
2. **Prioridad 2:** Construir `recuperar/recuperar.py` con agregación de score a nivel de documento (`F1@3`) y recorte a 250 palabras (`NDCG@10`).
3. **Prioridad 3:** Crear `generar_entregable.py` para producir y validar el archivo `resultados.jsonl` de 50 consultas.
4. **Prioridad 4:** Implementar hibridación BM25 + FAISS con Reciprocal Rank Fusion (RRF) para maximizar métricas en la tabla clasificatoria.
