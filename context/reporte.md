# Reporte de Estado y Diagnóstico Técnico — kernel-boyacense

Para ver el **Informe de Análisis Técnico Completo y Matriz Detallada de TODOs** para la Etapa 1 del CODEFEST AD ASTRA 2026, consulte el documento:
 **[`context/informe_analisis_todos.md`](file:///home/jose/repositories/kernel-boyacense/context/informe_analisis_todos.md)**

---

## Síntesis Ejecutiva del Estado Actual

### 1. Auditoría del Punto 3.4 del PDF (`CODEFEST_2026-1.pdf`)
Todos los problemas de metadatos identificados en la ingesta han sido **completamente resueltos y verificados**:

*  **`doc_id`:** Mapeado de forma inmutable y persistente en `doc_id_registry.json` (`DOC-XXXX`).
*  **`chunk_id` / `posicion`:** Corregidos en `fragmentacion.py`. La posición se gestiona de forma continua a nivel de documento (`posiciones_por_doc`), evitando reinicios por página y garantizando IDs únicos (`DOC-XXXX-chunk-YYY`).
*  **`fuente`:** Normalizado a rutas relativas Posix estandarizadas respecto al corpus (`fenomeno_X/archivo.ext`), asegurando coincidencia con la evaluación `F1@3`.
*  **`formato`:** Normalizado a los valores canónicos requeridos (`md`, `html`, `pdf`, `txt`, `json`, `csv`, `xlsx`, `pbf`, `gpkg`).
*  **`fenomeno`:** Inferencia garantizada (1, 2 o 3) por estructura de directorios o búsqueda de palabras clave (`inferir_fenomeno`); nunca retorna `None`.
*  **`num_tokens`:** Calculado utilizando el tokenizador `intfloat/multilingual-e5-large-instruct`.
*  **`texto`:** Extraído de manera limpia y sin contaminación de metadatos.

---

### 2. Módulos y Herramientas Desarrolladas

* **`extraccion/`**: Extracción multimodal completa (PDF, HTML, MD, TXT, JSON, CSV, XLSX, Imagen OCR con RapidOCR/Tesseract, PBF/OSM).
* **`indexar/`**: Conversión a vectorstore FAISS con embeddings normalizados L2 (`intfloat/multilingual-e5-large-instruct`).
* **`exportar_faiss.py`**: Herramienta CLI para parsear, inspeccionar y exportar el índice FAISS a formatos JSON, JSONL, CSV y Markdown (`salida/faiss_resumen.md`), con testbench semántico por CLI.
* **`visualizador_faiss.py`**: Servidor web interactivo en Python (Puerto 8501) con Dashboard de métricas (Chart.js), explorador de chunks con filtrado/paginación, visor de metadatos JSON y banco de pruebas de búsqueda vectorial en tiempo real.
* **`pipeline/`**: Omitido deliberadamente por ser un módulo desactualizado e inactivo.

---

## Próximos Pasos (ROADMAP Prioritario)

1. **Desarrollo del Módulo de Recuperación (`recuperar/`):** Implementar búsqueda vectorial con prefijo de instrucción e5, agregación a nivel de documento (`F1@3`) y fragmentos ≤ 250 palabras (`NDCG@10`).
2. **Generador del Entregable (`generar_entregable.py`):** Procesador masivo de 50 consultas de evaluación (`q001` - `q050`) produciendo `resultados.jsonl` en formato JSON Lines válido (§9.3).
3. **Búsqueda Híbrida (BM25 + FAISS + RRF):** Fusión de rankings (Reciprocal Rank Fusion) sin decoders LLM generativos (§8.4) para maximizar la puntuación en el Conteo de Borda (§11.2).



