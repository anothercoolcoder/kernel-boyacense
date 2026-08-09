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

## 2026-08-08 - Interfaz de revisión ground truth

### Implementado

- `revisar_ground_truth.py`
  - Servidor local sin LLM en `127.0.0.1:8765`.
  - Usa `resultados_benchmark_combsum.json` como ranking base.
  - Enriquece texto desde `metadata.jsonl`.
  - Muestra hasta 15 candidatos únicos por consulta.
  - Permite asignar relevancia `0`, `1` o `2`.
  - Guarda `draft` incrementalmente.
  - Aprueba solo con al menos un chunk `2`.
  - Actualiza `relevant_doc_ids` desde chunks marcados `1` o `2`.
  - Crea backup `.bak` antes de cada escritura.
- `test/test_revisar_ground_truth.py`
  - Valida candidatos únicos y límite 15.
  - Valida requisito de aprobación.

### Uso

```bash
python revisar_ground_truth.py
```

Abrir `http://127.0.0.1:8765`. Interfaz local; no exponer fuera de máquina.

### Verificación

- `python -m unittest test.test_revisar_ground_truth test.test_benchmark_rag test.test_ground_truth test.test_recuperar_contrato -v`: 9 pruebas OK.
- `python -m py_compile revisar_ground_truth.py test/test_revisar_ground_truth.py`: OK.
- `git diff --check`: OK.

## 2026-08-08 - Evaluación con anotaciones humanas

### Resultado parcial

- Registros totales: `123`.
- Registros `approved`: `10` (`q001`-`q010`).
- Registros `draft`: `113`.
- CombSUM `NDCG@10`: `0.9043306788914519`.
- CombSUM `F1@3`: `0.5`.
- `incomplete_rate`: `0.0`.
- Artefacto: `resultados_benchmark_combsum_anotado.json`.

### Bloqueos de calidad

- Aprobaciones no tienen `annotator`.
- `q002`, `q003` y `q004` tienen `relevant_chunk_ids` inconsistentes con
  `graded_relevance`.
- Métricas son provisionales: muestra aprobada pequeña y tres registros
  inconsistentes.
- No comparar contra baseline ni declarar mejora todavía.

### Acción siguiente

- Corregir anotaciones `q001`-`q010`.
- Completar aprobación de consultas restantes.
- Repetir CombSUM y luego ejecutar RRF sobre mismo ground truth congelado.

## 2026-08-08 - Benchmark completo con ground truth

### Verificación

- Ground truth: `123/123` registros `approved`.
- `annotator`: `JoseSKL` en todos.
- IDs y grados: consistentes.
- CombSUM: `NDCG@10=0.9517803039022844`, `F1@3=0.702439024390244`.
- RRF: `NDCG@10=0.8584312273859401`, `F1@3=0.6883468834688348`.
- Delta CombSUM-RRF: `NDCG@10=+0.09334907651634428`, `F1@3=+0.014092140921409202`.
- Ambos: `incomplete_rate=0.0`.
- Artefactos: `resultados_benchmark_combsum_final.json`,
  `resultados_benchmark_rrf_final.json`.

### Cobertura lingüística

- ES: `115` consultas.
- EN: `8` consultas.
- PT: `0` consultas.
- Métricas no representan dataset oficial equilibrado ES/EN/PT.

### Siguiente fase

- Crear consultas paralelas ES/EN/PT por `intent_id`.
- Anotar traducciones y mantenerlas en mismo split.
- Repetir comparación CombSUM/RRF por idioma.

## 2026-08-08 - Benchmark paralelo multilingüe

### Implementado

- `crear_benchmark_multilingue.py`
  - Selecciona 20 intenciones de documentos y tipos variados.
  - Crea variantes ES, EN y PT con mismo `intent_id`.
  - Hereda evidencia aprobada como referencia inicial.
  - Marca traducciones `draft`; no aprueba automáticamente evidencia
    cross-lingual.
  - Genera `context/benchmark_multilingue.jsonl` con 60 consultas.
- `benchmark_rag.py`
  - Acepta dataset JSONL mediante `--dataset`.
- `test/test_benchmark_rag.py`
  - Valida 60 registros y tres idiomas.

### Uso posterior a revisión

```bash
python benchmark_rag.py --dataset context/benchmark_multilingue.jsonl --fusion-method combsum --output resultados_multilingue_combsum.json
python benchmark_rag.py --dataset context/benchmark_multilingue.jsonl --fusion-method rrf --output resultados_multilingue_rrf.json
```

No ejecutar comparación final mientras registros permanezcan `draft`.

## 2026-08-08 - Ampliación a conjunto oficial candidato

### Implementado

- `crear_benchmark_multilingue.py`
  - Ampliado de 20 a 50 intenciones.
  - Genera 150 consultas: 50 ES, 50 EN, 50 PT.
  - Mantiene mismo `intent_id` para cada trío paralelo.
  - Mantiene traducciones y anotaciones en `draft` para nueva revisión.
- `test/test_benchmark_rag.py`
  - Valida 150 registros y 50 intenciones.

### Estado

Dataset oficial candidato listo para revisión lingüística y de evidencia.
No calcular métricas finales hasta aprobar 150 registros.

## 2026-08-08 - Benchmark multilingüe final candidato

### Verificación

- `translation_status=approved`: `150/150`.
- `annotation_status=approved`: `150/150`.
- `annotator=JoseSKL`: `150/150`.
- Distribución: ES `50`, EN `50`, PT `50`.
- IDs y grados: consistentes.
- Consultas sin grado `2`: `9`; mantener advertencia metodológica.

### Resultados globales

- CombSUM: `NDCG@10=0.6326166578134127`, `F1@3=0.49644444444444447`.
- RRF: `NDCG@10=0.5971450947728327`, `F1@3=0.4904444444444444`.
- Delta CombSUM-RRF: NDCG `+0.035472`, F1 `+0.006`.
- Ambos: `incomplete_rate=0.0`.

### Resultados por idioma

- CombSUM: ES `0.9006/0.5520`, EN `0.3979/0.4553`, PT `0.5993/0.4820`.
- RRF: ES `0.8302/0.5520`, EN `0.3658/0.4373`, PT `0.5955/0.4820`.
- Formato: `NDCG@10/F1@3`.

### Siguiente decisión

CombSUM gana globalmente y en cada NDCG por idioma, pero brecha ES-EN
 permanece severa. Antes de tocar pesos, revisar queries EN con NDCG bajo,
 confirmar chunks cross-lingual y congelar test set.

## 2026-08-08 - Instrumentación y pilot split-variance

### Implementado

- `recuperar/recuperar.py`: normalizaciones `minmax`, `clipped_minmax`,
  `percentile`, `zsigmoid`; parámetro `alpha`; diagnóstico experimental.
- `benchmark_rag.py`: CLI acepta `--normalization-method` y `--alpha`.
- `pilot_split_variance.py`: 30 intenciones `dev`, 20 `test`, 20 seeds,
  5 grouped folds, regla lexicográfica y bootstrap CI pareado.
- `test/test_pilot_split_variance.py`: valida regla de selección.

### Estado

Pilot pendiente de ejecución. Producción sigue CombSUM/min-max/alpha 0.5.

### Resultado pilot

- Split variance: NDCG mean `0.6141199888880591`, std `0.005354952588768398`.
- Split variance: F1 mean `0.4833333333333333`, std `0.0`.
- Selección por folds: min-max `50`, z-score+sigmoid `50`.
- Clipped min-max vs min-max CI: `[-0.06939, -0.01212]`.
- Percentile vs min-max CI: `[-0.07229, -0.01677]`.
- Z-score+sigmoid vs min-max CI: `[-0.04297, 0.00966]`.
- Tie rate: `0.3333`.

### Lectura

Split variance menor que diferencias negativas observadas para clipped y
percentile. Z-score+sigmoid empata estadísticamente con min-max; no hay base
para reemplazar baseline. Min-max queda default hasta barrido alpha.

## 2026-08-08 - Barrido alpha preparado

### Implementado

- `barrido_alpha.py`
  - Congela normalización `minmax`.
  - Evalúa alpha completo de `0.0` a `1.0` en pasos `0.1`.
  - Conserva 30 intenciones `dev` y 20 `test` reservado.
  - Usa regla lexicográfica predefinida.
  - Escribe `resultados_barrido_alpha.json`.
- `test/test_barrido_alpha.py`
  - Verifica barrido completo sin sesgo hacia `0.5`.

### Estado

Ejecución pendiente. No modificar alpha producción hasta revisar métricas y
confirmar comportamiento por idioma.

### Resultado barrido alpha

- `alpha=0.1` seleccionado por regla lexicográfica.
- Dev NDCG: `0.6375071610870084`.
- Dev F1: `0.49444444444444446`.
- Peor idioma NDCG: `0.44267826914670977`.
- Brecha lingüística NDCG: `0.33863524790659555`.
- `alpha=0.5` baseline: NDCG `0.6250`, F1 `0.4833`.
- `alpha=0.0` dense: NDCG `0.6258`, F1 `0.5000`.
- `alpha=1.0` BM25: NDCG `0.2984`, F1 `0.3000`.
- Artefacto: `resultados_barrido_alpha.json`.

## 2026-08-08 - Evaluación test preparada

### Implementado

- `evaluar_test.py`
  - Lee test reservado desde reporte alpha.
  - Evalúa `alpha=0.0`, alpha elegido en dev y `alpha=0.5` baseline.
  - No selecciona configuración con métricas test.
  - Escribe `resultados_test_alpha.json`.
- `test/test_evaluar_test.py`
  - Verifica existencia de reporte dev.

### Estado

Ejecución pendiente. Test permanece ciego para selección.

### Resultado test

- Alpha elegido en dev: `0.1`.
- Test `alpha=0.0`: NDCG `0.6319575033519266`, F1 `0.5466666666666666`,
  peor idioma `0.5846636005548919`, brecha `0.13779857400425233`.
- Test `alpha=0.1`: NDCG `0.6419948662595757`, F1 `0.5383333333333333`,
  peor idioma `0.5867439311678663`, brecha `0.16398287926157973`.
- Test `alpha=0.5`: NDCG `0.644043237684194`, F1 `0.5161111111111112`,
  peor idioma `0.5328241330958562`, brecha `0.30735662439254485`.
- Artefacto: `resultados_test_alpha.json`.

### Lectura

Alpha `0.1` cumple regla de peor idioma, pero no maximiza NDCG global ni F1.
Alpha `0.5` tiene NDCG global apenas mayor, con peor paridad. No reajustar
alpha usando test; configuración seleccionada sigue `minmax + alpha=0.1`.

## 2026-08-08 - Error analysis por señal

### Implementado

- `recuperar/recuperar.py`: scores BM25/FAISS raw, normalizados y ranks
  individuales expuestos por fragmento.
- `analisis_senales.py`: compara BM25, FAISS, CombSUM alpha `0.1` y RRF en
  30 consultas EN/PT débiles; clasifica fallo lexical, semantic, fusion,
  corpus/chunking o mixto.
- `test/test_analisis_senales.py`: valida clasificación lexical.

### Estado

Ejecución pendiente. No cambiar pesos ni producción antes de leer reporte.

### Resultado análisis

- Consultas analizadas: `30` bottom EN/PT.
- Clasificación inicial: corpus/chunking `26`, lexical BM25 `1`, mixto/anotación `3`.
- NDCG medio en muestra débil:
  - BM25: `0.0286`.
  - FAISS: `0.1021`.
  - CombSUM alpha `0.1`: `0.1035`.
  - RRF: `0.0585`.
- F1 medio en muestra débil:
  - BM25: `0.1444`.
  - FAISS: `0.4167`.
  - CombSUM alpha `0.1`: `0.3833`.
  - RRF: `0.3056`.

### Lectura

BM25 falla fuerte en muestra EN/PT. FAISS supera BM25; CombSUM alpha `0.1`
apenas supera FAISS en NDCG, pero pierde F1. Clasificación corpus/chunking es
heurística inicial, no diagnóstico final; revisar texto, idioma y ground truth
de cada query antes de atribuir causa.

### Lectura

BM25 solo colapsa cross-lingual. Dense domina calidad mínima por idioma.
Alpha `0.1` mejora peor idioma y NDCG contra `0.5`, pero F1 queda menor que
dense puro. Alpha aún no se aplica a test ni producción.

## 2026-08-08 - Benchmark multilingüe verificado

### Estado

- `translation_status=approved`: `60/60`.
- `annotation_status=approved`: `60/60`.
- Distribución: ES `20`, EN `20`, PT `20`.
- Tres registros por `intent_id`.

### Resultados globales

- CombSUM: `NDCG@10=0.6350289949632824`, `F1@3=0.44333333333333336`.
- RRF: `NDCG@10=0.6111166480723021`, `F1@3=0.435`.
- Delta CombSUM-RRF: NDCG `+0.023912`, F1 `+0.008333`.
- Ambos: `incomplete_rate=0.0`.

### Resultados por idioma

- CombSUM: ES `0.8383/0.5350`, EN `0.5139/0.4183`, PT `0.5529/0.3767`.
- RRF: ES `0.7603/0.5350`, EN `0.5083/0.3933`, PT `0.5647/0.3767`.
- Formato por idioma: `NDCG@10/F1@3`.

### Lectura

CombSUM gana globalmente. RRF supera CombSUM en NDCG portugués por `0.0118`.
Brecha lingüística continúa alta; benchmark tiene 20 intenciones, no conjunto
oficial final de 50.

## 2026-08-08 - Revisión de 30 queries débiles EN/PT

### Alcance

- Revisadas las 30 consultas seleccionadas por `resultados_analisis_senales.json`.
- Inicio confirmado: `ml-008`, `ml-032`, `ml-033`, `ml-038`, `ml-039`.
- Se compararon variantes BM25, FAISS, CombSUM y RRF, tríos por `intent_id`,
  texto de `metadata.jsonl`, traducción y anotación aprobada.

### Diagnóstico

- `21/30`: fallo primario de idioma/señal cross-lingual; evidencia correcta
  existe, pero consulta EN/PT pierde recuperación frente a corpus ES/EN/PT.
- `9/30`: ground truth incorrecto o contaminado; métricas no confiables.
- `0/30`: defecto de traducción semántica material confirmado.
- `0/30`: pérdida de evidencia por chunk confirmada.
- Índice revisado: `0` `chunk_id` duplicados; chunks de evidencia existen.

### Ground truth incorrecto

- `ml-032`, `ml-033` (`q078`): evidencia real en
  `DOC-0011-chunk-035`; anotación apunta a chunks irrelevantes de `DOC-0007`.
- `ml-038`, `ml-039` (`q091`): evidencia real en `DOC-0009-chunk-002` y
  `DOC-0009-chunk-004`; anotación apunta a chunks irrelevantes de `DOC-0003`.
- `ml-041`, `ml-042` (`q096`): evidencia real en `DOC-0010-chunk-002`;
  anotación apunta a chunks irrelevantes de `DOC-0007` y `DOC-0003`.
- `ml-051` (`q111`): `DOC-0002-chunk-013` es correcto, pero anotación mezcla
  chunks irrelevantes de `DOC-0007` y `DOC-0008`.
- `ml-107` (`q020`): evidencia real en `DOC-0003-chunk-031`; anotación lo
  marca grado `0` y usa chunks de perfiles de Guatemala, Argentina y Panamá.
- `ml-137` (`q032`): valor `-0.1` está en `DOC-0004-chunk-004`; anotación
  mezcla `DOC-0005` con valor distinto `-0.5` y chunk irrelevante de `DOC-0007`.

### Lectura

La brecha observada no permite atribuirse a chunking ni traducción. Ground
truth corrupto explica nueve consultas; idioma y ranking cross-lingual explican
la mayoría restante. Congelar métricas después de corregir anotaciones y
repetir evaluación por idioma.

## 2026-08-08 - Corrección GT y revaluación multilingüe

### Correcciones

- `context/benchmark_rag.jsonl`: corregidas anotaciones `q020`, `q032`,
  `q078`, `q091`, `q096` y `q111`.
- `context/benchmark_multilingue.jsonl`: propagadas correcciones a `18`
  variantes paralelas.
- Eliminados documentos y chunks irrelevantes; grados reducidos a evidencia
  verificable en metadata.

### Verificación

- Dataset multilingüe: `150` registros, `50` ES, `50` EN, `50` PT,
  `50` intents, `150/150` aprobados.
- CombSUM min-max alpha `0.5`: `NDCG@10=0.6252119968957256`,
  `F1@3=0.49`, `incomplete_rate=0.0`.
- RRF: `NDCG@10=0.5851477097468414`, `F1@3=0.484`,
  `incomplete_rate=0.0`.

### CombSUM por idioma

- ES: `0.8149210632219086 / 0.496`.
- EN: `0.44027922302423406 / 0.472`.
- PT: `0.6204357044410342 / 0.502`.
- Formato: `NDCG@10/F1@3`.

### Lectura

Corrección GT elimina falsos positivos de métricas anteriores, pero no elimina
brecha lingüística. No cambiar alpha ni chunking. Mantener GT corregido como
base congelada para siguiente experimento de recuperación cross-lingual.

## 2026-08-08 - Revaluación ejecutada tras corrección GT

### Verificación local

- Dataset: `150/150` aprobadas.
- Idiomas: ES `50`, EN `50`, PT `50`.
- `annotator=JoseSKL`: `150/150`.
- IDs y grados consistentes: `0` errores.
- CombSUM/min-max/alpha `0.5`: NDCG `0.6252119968957256`, F1 `0.49`.
- RRF: NDCG `0.5851477097468414`, F1 `0.484`.
- Ambos: `incomplete_rate=0.0`.
- Artefactos: `resultados_multilingue_combsum_gt_corregido.json`,
  `resultados_multilingue_rrf_gt_corregido.json`.

### Test suite

- `13` pruebas OK.
- `1` error: `test/test_revisar_ground_truth.py` importa
  `revisar_ground_truth.py`, archivo ausente en worktree.
- Restaurar interfaz o retirar test obsoleto antes de compartir.

## 2026-08-08 - Última revaluación GT corregido

### Ejecutado

- Error analysis actualizado contra GT corregido.
- Pilot split variance repetido.
- Alpha sweep repetido con min-max.
- Test reservado repetido sin selección post-test.

### Error analysis

- Clasificación: corpus/chunking `23`, lexical BM25 `3`, mixto/anotación `4`.
- Muestra bottom EN/PT NDCG:
  - BM25: `0.0470 / 0.1833`.
  - FAISS: `0.1710 / 0.4633`.
  - CombSUM alpha `0.1`: `0.1713 / 0.4300`.
  - RRF: `0.0943 / 0.3800`.
- Formato: `NDCG@10/F1@3`.

### Pilot

- Split NDCG mean `0.6011315170930323`, std `0.004586864302431589`.
- Clipped CI vs min-max: `[-0.07160, -0.01779]`.
- Percentile CI vs min-max: `[-0.07558, -0.02473]`.
- Zsigmoid CI vs min-max: `[-0.04777, 0.00020]`.
- Tie rate `0.3333`; min-max permanece.

### Alpha y test

- Dev eligió alpha `0.1`.
- Test alpha `0.0`: NDCG `0.6496017152033953`, F1 `0.5527777777777777`,
  peor idioma `0.6096224598383809`, brecha `0.0639952954092603`.
- Test alpha `0.1`: NDCG `0.6578192714870922`, F1 `0.5444444444444444`,
  peor idioma `0.6291634591410735`, brecha `0.049303728821476045`.
- Test alpha `0.5`: NDCG `0.6422887276021586`, F1 `0.5`,
  peor idioma `0.612168150340191`, brecha `0.06374323870031218`.
- Configuración recomendada: CombSUM + min-max + alpha `0.1`.
- Artefactos: `resultados_analisis_senales.json`,
  `resultados_pilot_split_variance.json`, `resultados_barrido_alpha.json`,
  `resultados_test_alpha.json`.

## 2026-08-08 - Resumen para equipo

- Creado `context/resumen_para_equipo.md`.
- Incluye cambios, estado actual, comandos de suite y propósito de cada grupo
  de pruebas.
- Bitácora permanece append-only.

### Verificación final

- `python -m unittest discover -s test -v`: `15` pruebas OK.

## 2026-08-08 - Reanálisis final con GT corregido

### Alcance

- `analisis_senales.py` actualizado para usar
  `resultados_multilingue_combsum_gt_corregido.json`.
- Pilot, alpha sweep y test serán recalculados desde GT corregido.
- Resultados anteriores no reutilizados.
