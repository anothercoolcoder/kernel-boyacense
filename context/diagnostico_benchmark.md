# Diagnóstico de problemas

## Resumen

La validación muestra que el índice y la metadata están consistentes, pero la recuperación no está alineada con el ground truth del benchmark.

## Problemas detectados

1. **Desalineación de IDs**
   - El dataset usa `DOC-0001` a `DOC-0011` como `relevant_doc_ids`.
   - El índice usa `doc_id` con prefijos como `F1-ILIA-*`, `F2-INPE-*` y `F3-SIPRI-*`.
   - Resultado: el benchmark original da `ndcg_at_10 = 0.0` y `f1_at_3 = 0.0`.

2. **Recuperación poco precisa**
   - Con el mapping automático aplicado, solo **7 de 123** consultas recuperan al menos un documento relevante en el top-3.
   - **116 de 123** siguen sin documento relevante en el top-3.

3. **Sesgo hacia documentos cercanos**
   - La recuperación repite con frecuencia documentos del bloque ILIA, aunque no correspondan a la consulta.
   - Ejemplo de patrones repetidos:
     - `F1-ILIA-005`, `F1-ILIA-002`, `F1-ILIA-008`
     - `F1-ILIA-006`, `F1-ILIA-008`, `F1-ILIA-009`

4. **Ground truth no coincide con el índice**
   - El benchmark referencia documentos que no existen con esos IDs en `metadata.jsonl`.
   - El problema no es de cardinalidad del índice, sino de correspondencia semántica entre anotaciones e índice.

## Estado de validación

- Índice FAISS: **OK**
- Metadata: **OK**
- Suite de pruebas: **OK** (21 tests)
- Exportación de chunks: **OK**
- Benchmark: **fallando por desalineación documental**

## Tests añadidos

- Verifican que los `relevant_doc_ids` del benchmark mapeado existan en `metadata.jsonl`.
- Verifican que el reporte del benchmark mapeado recupere al menos algunos documentos relevantes.
- La suite completa quedó en verde después de agregarlos.

## Conclusión

El sistema no está fallando por ausencia de datos, sino por una mezcla de:

- IDs incompatibles entre anotaciones e índice.
- Recuperación que no prioriza el documento correcto.
- Posible necesidad de rehacer el mapeo entre ground truth y corpus.
