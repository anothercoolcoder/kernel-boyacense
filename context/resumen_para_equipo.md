# Resumen del Trabajo RAG

Documento guía para equipo. Explica qué se hizo, cómo probarlo y cuál es
estado actual sin entrar en detalles innecesarios.

## Objetivo

Sistema debe recibir preguntas y devolver:

- 3 documentos.
- 10 fragmentos.
- Fragmentos sin IDs duplicados.
- Fragmentos pertenecientes a documentos devueltos.
- Máximo 250 palabras por fragmento.

Además, sistema debe funcionar con consultas en español, inglés y portugués.

## Cambios Principales

### 1. Contrato de salida

Se corrigió recuperación para cumplir formato esperado.

- Recorte respeta oraciones.
- Padding usa documentos y fragmentos únicos.
- No se duplican IDs.
- Se registran motivos de padding.
- Validaciones detectan fragmentos fuera de documento y textos largos.

### 2. Benchmark

Se creó medición automática de calidad.

- `NDCG@10`: mide si evidencia correcta aparece arriba.
- `F1@3`: mide si documentos necesarios aparecen entre primeros tres.
- Validación de cardinalidad y límite de palabras.
- Sin LLM ni generación automática de respuestas.

### 3. CombSUM

Sistema conserva dos señales:

- BM25: busca coincidencias de palabras.
- FAISS: busca similitud semántica con embeddings.

CombSUM combina ambas señales después de normalizarlas. RRF permanece como
baseline para comparar.

BM25 no fue eliminado.

### 4. Ground truth

Se anotaron preguntas con evidencia esperada.

- `relevant_doc_ids`: documentos necesarios.
- `relevant_chunk_ids`: fragmentos con evidencia.
- Relevancia `2`: respuesta directa.
- Relevancia `1`: contexto o evidencia parcial.
- Relevancia `0`: irrelevante.

Dataset final candidato tiene 50 intenciones, cada una en ES, EN y PT:

- 50 consultas ES.
- 50 consultas EN.
- 50 consultas PT.
- 150 consultas total.

Ground truth corregido mantiene `150/150` registros aprobados.

### 5. Evaluación de idioma

Se compararon CombSUM y RRF con dataset balanceado.

También se revisaron consultas débiles en inglés y portugués para separar:

- Problemas de BM25.
- Problemas de búsqueda semántica.
- Problemas de combinación.
- Problemas de corpus o fragmentación.
- Errores de ground truth.

Correcciones de ground truth eliminaron falsos errores de recuperación.

### 6. Normalización y alpha

Se probaron distintas formas de poner BM25 y FAISS en escala comparable.

Min-max quedó como opción estable. Otras variantes no mostraron mejora clara.

También se probó cuánto peso dar a BM25:

```text
score = alpha * BM25 + (1 - alpha) * FAISS
```

Configuración recomendada por evaluación:

```text
CombSUM
normalización min-max
alpha=0.1
```

Esto da poco peso a BM25 porque BM25 funciona débilmente entre idiomas.

No cambiar configuración de producción sin nueva aprobación del equipo.

## Archivos Importantes

- `recuperar/recuperar.py`: recuperación, padding, CombSUM y scores.
- `benchmark_rag.py`: métricas y runner.
- `context/benchmark_rag.jsonl`: ground truth general.
- `context/benchmark_multilingue.jsonl`: benchmark ES/EN/PT.
- `resultados_multilingue_combsum_gt_corregido.json`: resultado CombSUM corregido.
- `resultados_multilingue_rrf_gt_corregido.json`: resultado RRF corregido.
- `resultados_pilot_split_variance.json`: prueba de estabilidad de normalizaciones.
- `resultados_barrido_alpha.json`: comparación de pesos BM25/FAISS en dev.
- `resultados_test_alpha.json`: evaluación final sobre test reservado.
- `context/bitacora_cambios_rag.md`: historial append-only de cambios.

## Ejecutar Suite

Desde raíz del proyecto:

```bash
python -m unittest discover -s test -v
```

Este comando ejecuta todas pruebas estándar del proyecto.

### Qué valida cada grupo

- `test_recuperar_contrato.py`: recorte, IDs únicos, padding y cardinalidad.
- `test_benchmark_rag.py`: cálculo de NDCG, F1 y carga multilingüe.
- `test_ground_truth.py`: borradores no contaminan métricas.
- `test_revisar_ground_truth.py`: interfaz de anotación y regla de aprobación.
- `test_pilot_split_variance.py`: regla de selección de normalización.
- `test_barrido_alpha.py`: barrido completo de pesos de `0.0` a `1.0`.
- `test_evaluar_test.py`: test usa configuración elegida en dev.
- `test_analisis_senales.py`: clasificación básica de fallo lexical.

### Pruebas adicionales

Verificar sintaxis:

```bash
python -m py_compile recuperar/recuperar.py benchmark_rag.py
```

Ejecutar benchmark CombSUM:

```bash
python benchmark_rag.py \
  --dataset context/benchmark_multilingue.jsonl \
  --fusion-method combsum \
  --normalization-method minmax \
  --alpha 0.1 \
  --output resultados_combsum_equipo.json
```

Ejecutar benchmark RRF para comparación:

```bash
python benchmark_rag.py \
  --dataset context/benchmark_multilingue.jsonl \
  --fusion-method rrf \
  --output resultados_rrf_equipo.json
```

## Estado Actual

- Recuperación cumple cardinalidad en smoke benchmark.
- Ground truth corregido y aprobado.
- Benchmark balanceado entre tres idiomas.
- CombSUM supera RRF en evaluación corregida.
- BM25 sigue siendo útil como señal secundaria, no principal.
- Brecha entre español e inglés todavía existe.
- No usar resultados antiguos generados antes de corregir ground truth.
- Revisar siempre `context/bitacora_cambios_rag.md` antes de cambiar código.

## Siguiente Paso

Equipo debe reproducir suite y benchmark. Después, revisar consultas EN/PT
con peor NDCG y probar mejoras una por una. Cada cambio debe medirse contra
test reservado y registrarse en bitácora sin borrar entradas anteriores.
