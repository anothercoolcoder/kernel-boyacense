# Bitácora de Cambios RAG

Este archivo es append-only. No borrar ni sobrescribir entradas anteriores.
Cada fase debe agregar nueva entrada con fecha, archivos, cambios y pruebas.

## 2026-08-08 - Fase 1: contrato P0

### Contexto

- Corpus `corpus_adl` fue ampliado y reindexado por el usuario.
- `preguntas.txt` y `respuestas.txt` contienen preguntas y respuestas para corpus ampliado.
- No se revisó contenido documental; se tomó como contexto entregado.
- RRF permanece vigente en esta fase. CombSUM queda para fase posterior.

### Implementado

- `recuperar/recuperar.py`
  - Recorte por oraciones antes de aplicar límite de 250 palabras.
  - Fallback de límite estricto para oración individual sobredimensionada.
  - Padding de documentos con IDs únicos y score `0.0`.
  - Padding de fragmentos con IDs únicos desde documentos seleccionados.
  - Fragmentos restringidos a `doc_id` presentes en `documents`.
  - Diagnóstico de documentos y fragmentos rellenados.
  - Función única `contar_palabras()`.
- `generar_entregable.py`
  - Validación de IDs únicos.
  - Validación de pertenencia fragmento-documento.
  - Conteo compartido mediante `contar_palabras()`.

### No implementado todavía

- CombSUM.
- Benchmark NDCG@10/F1@3.
- Normalización BM25.
- Manifiesto de reindexación.

### Verificación pendiente

- `python -m py_compile recuperar/recuperar.py generar_entregable.py`: OK.
- Prueba directa de recorte oracional: OK.
- `python -m unittest test.test_recuperar_contrato -v`: 2 pruebas OK.
- Smoke test con índice reindexado: pendiente; requiere cargar modelo.
- Confirmar cardinalidad exacta en todas las preguntas: pendiente.

## 2026-08-08 - Fase 2: benchmark local

### Implementado

- `benchmark_rag.py`
  - Carga preguntas desde `preguntas.txt` con IDs estables.
  - Carga ground truth opcional desde JSONL.
  - Calcula `NDCG@10` con relevancia graduada.
  - Calcula `F1@3` documental.
  - Valida cardinalidad, IDs únicos, pertenencia documental y límite de palabras.
  - Reporta consultas anotadas, no anotadas e incompletas.
  - No usa LLM, decoder ni generación de respuestas.
- `test/test_benchmark_rag.py`
  - Pruebas unitarias de NDCG, F1 y ejecución con ground truth.

### Limitación explícita

`preguntas.txt` y `respuestas.txt` no contienen todavía `relevant_doc_ids`,
`relevant_chunk_ids` ni `graded_relevance`. Runner no inventa ground truth;
sin anotaciones, métricas principales quedan `null`.

### Verificación pendiente

- `python -m unittest test.test_benchmark_rag test.test_recuperar_contrato -v`: 5 pruebas OK.
- `python -m py_compile benchmark_rag.py test/test_benchmark_rag.py`: OK.
- `python benchmark_rag.py --output resultados_benchmark_baseline.json`: OK.
- Baseline procesó 123 consultas activas.
- `incomplete_rate`: `0.0`.
- `annotated_queries`: `0`; `NDCG@10` y `F1@3` quedan `null` hasta anotar ground truth.

## 2026-08-08 - Fase 3: pivote CombSUM

### Implementado

- `recuperar/recuperar.py`
  - CombSUM queda como fusión predeterminada.
  - Scores BM25 y FAISS se normalizan mediante min-max a `[0, 1]`.
  - RRF permanece disponible con `fusion_method="rrf"` para comparación baseline.
  - Desempate determinista por índice.
  - Campo de salida renombrado a `score_fusion`.
  - Diagnóstico registra método de fusión.
- `benchmark_rag.py`
  - Permite ejecutar `--fusion-method combsum` o `--fusion-method rrf`.
- `test/test_recuperar_contrato.py`
  - Prueba de normalización CombSUM.

### Limitación

Comparación NDCG@10/F1@3 aún no posible: ground truth sigue sin anotarse.

### Verificación

- `python -m unittest test.test_recuperar_contrato test.test_benchmark_rag -v`: 6 pruebas OK.
- `python -m py_compile recuperar/recuperar.py benchmark_rag.py`: OK.
- Corrida CombSUM: 123 consultas, `incomplete_rate=0.0`.
- Corrida RRF: 123 consultas, `incomplete_rate=0.0`.
- Ambas corridas: `annotated_queries=0`; métricas de calidad `null`.

## 2026-08-08 - Fase 4: ground truth candidato

### Implementado

- `preparar_ground_truth.py`
  - Alinea preguntas y respuestas por orden.
  - Asocia grupos documentales con `doc_id` del registro actual.
  - Busca candidatos por coincidencia léxica normalizada en metadata.
  - Conserva página, sección y score candidato.
  - Genera `context/benchmark_rag.jsonl`.
- `benchmark_rag.py`
  - Solo cuenta registros con `annotation_status="approved"`.
- `test/test_ground_truth.py`
  - Impide que borradores contaminen métricas.

### Regla de anotación

`candidate_chunks` son sugerencias automáticas, no ground truth.
Revisor debe confirmar `relevant_doc_ids`, `relevant_chunk_ids` y
`graded_relevance`; después cambiar `annotation_status` a `approved`.
No borrar historial ni convertir automáticamente candidatos en verdad.

### Estado del borrador

- Registros: `123`.
- Registros con candidatos automáticos: `115`.
- Registros sin candidatos automáticos: `8`.
- Registros aprobados: `0`.
