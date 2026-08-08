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
