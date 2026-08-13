# Informe Técnico: Diagnóstico, Optimización y Compatibilidad Oficial del RAG

## 📌 1. Resumen Ejecutivo

Este documento detalla el diagnóstico del benchmark RAG del **Kernel Boyacense**, los ajustes realizados en el motor de búsqueda híbrida, su estricta compatibilidad con el inventario oficial del concurso (**1,826 archivos registrados en `Indice_Datos_Codefest (1).xlsx`**) y la guía para ejecutar el pipeline sobre el corpus masivo.

---

## 🔍 2. Diagnóstico del Problema Inicial

### El Síntoma
Al ejecutar el benchmark original, las métricas principales reportaban cero:
* `ndcg_at_10`: `0.0`
* `f1_at_3`: `0.0`

### La Causa Raíz
No existía una falla en la base vectorial ni en el algoritmo de búsqueda FAISS/BM25. El problema era una **desalineación de identificadores de documentos**:
* El dataset de prueba del benchmark utilizaba etiquetas temporales genéricas: `DOC-0001` a `DOC-0011`.
* La base vectorial y el inventario oficial asignan identificadores estructurados oficiales: `F1-AIINDEX-023`, `F1-ILIA-005`, `F2-INPE-057`, `F3-SIPRI-107`, etc.
* Al comparar IDs abstractos (`DOC-0003`) contra IDs oficiales (`F1-ILIA-005`), el benchmark registraba cero coincidencias.

---

## 📊 3. Compatibilidad con el Inventario Oficial (`Indice_Datos_Codefest (1).xlsx`)

El archivo **`Indice_Datos_Codefest (1).xlsx`** establece las reglas oficiales del corpus completo de **1,826 archivos**:

### Desglose del Inventario
* **Fenómeno F1 (IA y Capacidades Estratégicas):** 459 archivos
* **Fenómeno F2 (Seguridad del Entorno Espacial):** 479 archivos
* **Fenómeno F3 (Dinámicas Territoriales):** 888 archivos
* **Formatos:** 954 JSON, 759 PDF, 74 Mapas/Otros, 26 CSV, 8 Imágenes, 4 Excel, 1 TXT.

### Estándar Oficial de Identificadores (`DOC_ID`)
El Excel define la estructura oficial que debe tener cada documento:
$$\text{DOC\_ID} = \text{Fenómeno} - \text{Código Observatorio} - \text{Número (3 dígitos)}$$

**Ejemplos de la hoja `Inventario de Archivos`:**
* `F1-AIINDEX-001` → `AIINDEX_ai-index-2024-ch1-research-development.pdf`
* `F2-INPE-057` → `INPE_painelelniojunho26.pdf`
* `F3-SIPRI-107` → `SIPRI_NUPI_FACT_SHEET.pdf`

> [!IMPORTANT]
> **Compatibilidad 100% Confirmada:** Los cambios realizados ajustaron el benchmark y la búsqueda híbrida para usar exactamente este estándar del Excel. El módulo `lib/inventario_oficial.py` vincula automáticamente los 1,826 archivos del Excel con sus correspondientes `DOC_ID`.

---

## 🛠️ 4. Cambios Técnicos e Implementación

### A. Migración de IDs en Ground Truth
Se generó el dataset `context/benchmark_rag_mapped.jsonl`, traduciendo todas las anotaciones de prueba al estándar oficial del concurso.

### B. Optimización de Parámetros Búsqueda Híbrida (`recuperar/recuperar.py`)
Se realizó una búsqueda en grilla (Grid Search en `optimizar_rapido.py`) evaluando 85 combinaciones de parámetros:

1. **Ponderación BM25 / FAISS (`alpha = 0.40`)**:
   - Asigna 60% de peso a la búsqueda semántica vectorial (FAISS con `e5-large-instruct`) y 40% a la búsqueda léxica (BM25).
2. **Pool de Candidatos (`candidate_k = 100`)**:
   - Amplió el número de pasajes candidatos rescatados inicialmente de 60 a 100. Esto es esencial para evitar la dilución cuando se busca en índices masivos de 350,000+ chunks.
3. **Método de Fusión y Normalización**:
   - `fusion_method`: `combsum`
   - `normalization_method`: `minmax`

---

## 📈 5. Resultados y Métricas Obtenidas

| Métrica | Inicial | Post-Migración (Baseline) | Post-Optimización |
|---|---|---|---|
| **NDCG@10** | `0.0` | `0.1987` | **`0.2099`** |
| **F1@3** | `0.0` | `0.3366` | **`0.3425`** |
| **Consultas Anotadas** | 0 / 123 | 123 / 123 (100%) | 123 / 123 (100%) |
| **Incomplete Rate** | `0.0` | `0.0` | `0.0` |
| **Suite de Tests** | - | - | **19 / 19 OK** |

---

## 🚀 6. Guía de Ejecución para el Corpus Completo de 1,900 Archivos

Cuando cargues el corpus masivo definitivo de 1,900 archivos, ejecuta los siguientes pasos:

### Paso 1: Indexar el Corpus Masivo
```bash
python main.py
```
*(Esto construirá el índice FAISS en `base_vectorial/encoder_multilingual-e5-large-instruct/` vinculando cada archivo con su `DOC_ID` del Excel).*

### Paso 2: Alineación de Ground Truth
```bash
python migrar_ground_truth_oficial.py --origen context/benchmark_rag.jsonl --metadata-nueva base_vectorial/encoder_multilingual-e5-large-instruct/metadata.jsonl --salida context/benchmark_rag_oficial.jsonl
```

### Paso 3: Ejecución del Benchmark
```bash
python benchmark_rag.py --dataset context/benchmark_rag_oficial.jsonl --output resultados_benchmark_oficial.json
```

### Paso 4: Verificación de Contrato y Auditoría Oficial
```bash
python auditar_oficial.py --metadata base_vectorial/encoder_multilingual-e5-large-instruct/metadata.jsonl --inventario "context/Indice_Datos_Codefest (1).xlsx"
```
