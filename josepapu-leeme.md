# Walkthrough — Implementación del Módulo Grafo de Conocimiento

Se ha implementado de forma **100% desacoplada y libre de conflictos** el **Componente Bonus: Grafo de Conocimiento** para el proyecto **Kernel Boyacense**, cumpliendo estrictamente con todas las reglas de negocio del retador **CODEFEST AD ASTRA 2026**.

---

## 🎯 Requisitos Cumplidos

| Requisito | Estado | Implementación / Evidencia |
| :--- | :---: | :--- |
| **Prohibición de LLMs / Decoders** |  Cumplido | Uso exclusivo de **GLiNER** (`urchade/gliner_multi-v2.1`), modelo *encoder* zero-shot tipo BERT/DeBERTa + fallback heurístico. |
| **Co-ocurrencia por Fragmento** |  Cumplido | `ConstructorGrafo` mapea entidades mencionadas dentro del mismo `doc_id` y `chunk_id`. |
| **Evidencia Textual en Aristas** |  Cumplido | Cada arista en el grafo incluye las propiedades obligatorias `doc_id` y `chunk_id`. |
| **Formato Entregable GraphML** |  Cumplido | Exportación nativa a `entrega/grafo/grafo.graphml` con `NetworkX`. |
| **Recuperación Topológica (Paso 4)** |  Cumplido | `GrafoRecuperador` extrae entidades de la query, consulta subgrafos 1-hop y asigna scores de coincidencia. |
| **Fusión RRF** |  Cumplido | `fusionar_rrf` combina los fragmentos del Grafo con FAISS y BM25 usando $k_0 = 60$. |
| **Cero Conflictos con Indexación/RRF** |  Cumplido | Todo el código se aisló en la subcarpeta `grafo/`. No se modificó `recuperar.py` ni `indexar.py`. |

---

## 🛠️ Estructura del Módulo `grafo/`

- **[grafo/extractor.py](file:///home/niichan/Documents/university/kernel-boyacense/grafo/extractor.py):** Extractor NER zero-shot multilingüe con GLiNER y fallback seguro.
- **[grafo/builder.py](file:///home/niichan/Documents/university/kernel-boyacense/grafo/builder.py):** Constructor del grafo NetworkX y exportador a `GraphML`.
- **[grafo/retriever.py](file:///home/niichan/Documents/university/kernel-boyacense/grafo/retriever.py):** Motor de búsqueda topológica y exploración 1-hop.
- **[grafo/fusion.py](file:///home/niichan/Documents/university/kernel-boyacense/grafo/fusion.py):** Función de fusión RRF para FAISS + BM25 + Grafo.
- **[construir_grafo.py](file:///home/niichan/Documents/university/kernel-boyacense/construir_grafo.py):** Script principal ejecutable CLI.
- **[test/test_grafo.py](file:///home/niichan/Documents/university/kernel-boyacense/test/test_grafo.py):** Suite de pruebas unitarias e integración end-to-end.

---

## 🧪 Verificación y Resultados de Pruebas

### 1. Ejecución de Pruebas Unitarias
El script `test/test_grafo.py` pasó el 100% de las aserciones:
- **NER con GLiNER:** Detectó exitosamente entidades como `EE.UU.`, `OTAN`, `NASA`, `DARPA`, `sistemas autónomos`, `Colombia`.
- **Construcción y GraphML:** Generó nodos y aristas validando que las propiedades `doc_id` y `chunk_id` estén presentes en cada relación.
- **Recuperación Topológica y RRF:** Devolvió ranking ordenado por score RRF fusionando los canales.

### 2. Generación del Archivo Entregable (`grafo.graphml`)
Se ejecutó el pipeline de construcción generando el entregable oficial:
- **Archivo:** [entrega/grafo/grafo.graphml](file:///home/niichan/Documents/university/kernel-boyacense/entrega/grafo/grafo.graphml)
- **Tamaño:** `265 KB`
- **Contenido:** 90 Nodos y 940 Aristas de co-ocurrencia extraídas.

---

## 🚀 Cómo Usar / Conectar cuando tu compañero suba su código

Cuando tu compañero suba sus cambios en `recuperar.py`:

```python
from grafo import GrafoRecuperador, fusionar_rrf

# 1. Cargar recuperador de grafo
recuperador_grafo = GrafoRecuperador("entrega/grafo/grafo.graphml")

# 2. En la función de búsqueda:
res_faiss = ... # tus resultados de FAISS
res_bm25  = ... # tus resultados de BM25
res_grafo = recuperador_grafo.recuperar_fragmentos(query, top_k=50)

# 3. Fusión RRF de 3 vías:
res_finales = fusionar_rrf({
    "faiss": res_faiss,
    "bm25": res_bm25,
    "grafo": res_grafo
}, k0=60)
```

Y para actualizar el grafo final cuando esté listo el nuevo `metadata.jsonl`:
```bash
python construir_grafo.py --metadata base_vectorial/RUTA_NUEVA/metadata.jsonl
```
