"""Script de Evaluación Comparativa: Base Vectorial (FAISS + BM25) vs. Híbrido + Grafo de Conocimiento.

Mapea fragmentos recuperados a IDs del benchmark oficial utilizando la página del documento 
proveniente de metadatos, garantizando mediciones exactas de NDCG@10 y F1@3.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple
import numpy as np

RAIZ = Path(__file__).parent
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from benchmark_rag import cargar_dataset_jsonl, f1_at_3, ndcg_at_k
from recuperar.recuperar import BuscadorHibrido
from grafo import GrafoRecuperador, fusionar_rrf

DATASET_PATH = RAIZ / "context" / "benchmark_multilingue.jsonl"
RAG_DATASET_PATH = RAIZ / "context" / "benchmark_rag.jsonl"
GRAFO_PATH = RAIZ / "entrega" / "grafo" / "grafo.graphml"

DOC_MAP = {
    "DOC-0001": "F1-AIINDEX-023",
    "DOC-0002": "F2-ESA-013",
    "DOC-0003": "F1-ILIA-005",
    "DOC-0004": "F2-INPE-057",
    "DOC-0005": "F2-INPE-058",
    "DOC-0006": "F2-INPE-059",
    "DOC-0007": "F3-MAPPOEA-014",
    "DOC-0008": "F3-MAPPOEA-017",
    "DOC-0009": "F3-SIPRI-107",
    "DOC-0010": "F3-SIPRI-122",
    "DOC-0011": "F2-SWF-128",
}

REVERSE_DOC_MAP = {v: k for k, v in DOC_MAP.items()}


def construir_mapa_paginas() -> Dict[Tuple[str, int], str]:
    """Mapea (doc_oficial, numero_pagina) -> chunk_id_oficial del benchmark."""
    page_map: Dict[Tuple[str, int], str] = {}
    if RAG_DATASET_PATH.exists():
        with open(RAG_DATASET_PATH, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                data = json.loads(line.strip())
                for candidate in data.get("candidate_chunks", []):
                    doc_id = candidate.get("doc_id")
                    pagina = candidate.get("pagina")
                    chunk_id = candidate.get("chunk_id")
                    if doc_id and pagina is not None and chunk_id:
                        page_map[(doc_id, int(pagina))] = chunk_id
    return page_map


def mapear_doc_id_oficial(doc_id: str) -> str:
    """Convierte doc_id interno a oficial."""
    return REVERSE_DOC_MAP.get(doc_id, doc_id)


def mapear_chunk_con_metadatos(
    doc_id: str,
    chunk_id: str,
    metadatos_list: List[Dict[str, Any]],
    page_map: Dict[Tuple[str, int], str]
) -> str:
    """Busca los metadatos reales del fragmento (para extraer página) y lo mapea al ID oficial del benchmark."""
    doc_oficial = mapear_doc_id_oficial(doc_id)

    # Buscar página en metadatos por doc_id y chunk_id
    pagina = None
    c_str = str(chunk_id).split(":")[-1]
    for meta in metadatos_list:
        if str(meta.get("doc_id")) == str(doc_id) and str(meta.get("chunk_id")) == c_str:
            pagina = meta.get("pagina")
            break

    if pagina is not None and (doc_oficial, int(pagina)) in page_map:
        return page_map[(doc_oficial, int(pagina))]

    try:
        num_c = int(c_str)
        return f"{doc_oficial}-chunk-{num_c:03d}"
    except ValueError:
        return f"{doc_oficial}-chunk-{chunk_id}"


def evaluar_sistema_con_y_sin_grafo(limit: int | None = None) -> Dict[str, Any]:
    print("================================================================================")
    print("EVALUACIÓN COMPARATIVA: BASE VECTORIAL VS. BASE VECTORIAL + GRAFO DE CONOCIMIENTO")
    print("================================================================================")

    if not DATASET_PATH.exists():
        print(f"[!] Error: No se encontró el dataset benchmark en {DATASET_PATH}")
        sys.exit(1)

    base_vec_dir = RAIZ / "base_vectorial"
    subdirs = [d for d in base_vec_dir.iterdir() if d.is_dir()]
    faiss_dir = subdirs[0] if subdirs else base_vec_dir / "encoder_multilingual-e5-large-instruct-prueba"

    print(f"[*] Inicializando Buscador Híbrido usando índice en: {faiss_dir}")
    buscador = BuscadorHibrido(faiss_dir=faiss_dir)

    if not GRAFO_PATH.exists():
        print(f"[*] Grafo no encontrado en {GRAFO_PATH}. Construyendo grafo sobre índice actual...")
        from construir_grafo import main as construir_main
        sys.argv = ["construir_grafo.py", "--metadata", str(faiss_dir / "metadata.jsonl"), "--salida", str(GRAFO_PATH)]
        construir_main()

    recuperador_grafo = GrafoRecuperador(GRAFO_PATH)
    page_map = construir_mapa_paginas()

    print(f"[*] Cargando dataset benchmark desde: {DATASET_PATH}")
    registros = cargar_dataset_jsonl(DATASET_PATH)
    if limit:
        registros = registros[:limit]

    print(f"[*] Total de consultas a evaluar: {len(registros)}")

    f1_baseline, ndcg_baseline = [], []
    f1_grafo, ndcg_grafo = [], []

    metadata_map = {idx: meta for idx, meta in enumerate(buscador.metadatos)}

    for registro in registros:
        query = registro["query"]
        rel_doc_ids = set(registro.get("relevant_doc_ids", []))
        graded_rel = registro.get("graded_relevance", {})

        # -------------------------------------------------------------
        # 1. BASELINE (FAISS + BM25)
        # -------------------------------------------------------------
        res_baseline = buscador.buscar(query, top_k_docs=3, top_k_chunks=10, fusion_method="rrf")
        base_docs = [mapear_doc_id_oficial(d["doc_id"]) for d in res_baseline.get("documents", [])]
        base_chunks = [
            mapear_chunk_con_metadatos(f["doc_id"], f["chunk_id"], buscador.metadatos, page_map)
            for f in res_baseline.get("fragments", [])
        ]

        f1_base = f1_at_3(base_docs, rel_doc_ids)
        ndcg_base = ndcg_at_k(base_chunks, graded_rel, k=10)

        f1_baseline.append(f1_base)
        ndcg_baseline.append(ndcg_base)

        # -------------------------------------------------------------
        # 2. HÍBRIDO + GRAFO DE CONOCIMIENTO
        # -------------------------------------------------------------
        res_grafo_raw = recuperador_grafo.recuperar_fragmentos(query, top_k=50)

        # Búsqueda FAISS densa real
        instruccion = "Given a question, retrieve passages from documents that contain the exact factual information needed to answer the question"
        query_fmt = f"Instruct: {instruccion}\nQuery: {query}"
        q_vec = np.array([buscador.embeddings_model.embed_query(query_fmt)], dtype=np.float32)
        scores_f, indices_f = buscador.faiss_index.search(q_vec, 50)

        faiss_raw = [
            {"doc_id": metadata_map[idx]["doc_id"], "chunk_id": str(metadata_map[idx]["chunk_id"]), "score": float(sc)}
            for sc, idx in zip(scores_f[0], indices_f[0]) if idx >= 0
        ]

        bm25_raw = [
            {"doc_id": metadata_map[i]["doc_id"], "chunk_id": str(metadata_map[i]["chunk_id"]), "score": sc}
            for i, sc in buscador.bm25.search(query, top_k=50)
        ]

        fusion_3way = fusionar_rrf({
            "faiss": faiss_raw,
            "bm25": bm25_raw,
            "grafo": res_grafo_raw
        }, k0=60, top_k=10)

        grafo_chunks = [
            mapear_chunk_con_metadatos(item["doc_id"], item["chunk_id"], buscador.metadatos, page_map)
            for item in fusion_3way
        ]

        # Max pooling documental
        grafo_docs_scores: Dict[str, float] = {}
        for item in fusion_3way:
            d_id = mapear_doc_id_oficial(item["doc_id"])
            sc = item["score"]
            if d_id not in grafo_docs_scores or sc > grafo_docs_scores[d_id]:
                grafo_docs_scores[d_id] = sc
        grafo_docs = [d_id for d_id, _ in sorted(grafo_docs_scores.items(), key=lambda x: x[1], reverse=True)[:3]]

        f1_g = f1_at_3(grafo_docs, rel_doc_ids)
        ndcg_g = ndcg_at_k(grafo_chunks, graded_rel, k=10)

        f1_grafo.append(f1_g)
        ndcg_grafo.append(ndcg_g)

    mean_f1_base = sum(f1_baseline) / len(f1_baseline) if f1_baseline else 0.0
    mean_ndcg_base = sum(ndcg_baseline) / len(ndcg_baseline) if ndcg_baseline else 0.0

    mean_f1_grafo = sum(f1_grafo) / len(f1_grafo) if f1_grafo else 0.0
    mean_ndcg_grafo = sum(ndcg_grafo) / len(ndcg_grafo) if ndcg_grafo else 0.0

    diff_f1 = mean_f1_grafo - mean_f1_base
    diff_ndcg = mean_ndcg_grafo - mean_ndcg_base

    pct_f1 = (diff_f1 / mean_f1_base * 100) if mean_f1_base else 0.0
    pct_ndcg = (diff_ndcg / mean_ndcg_base * 100) if mean_ndcg_base else 0.0

    reporte = {
        "total_consultas_evaluadas": len(registros),
        "baseline_vectorial": {
            "F1_at_3": round(mean_f1_base, 4),
            "NDCG_at_10": round(mean_ndcg_base, 4),
        },
        "hibrido_con_grafo": {
            "F1_at_3": round(mean_f1_grafo, 4),
            "NDCG_at_10": round(mean_ndcg_grafo, 4),
        },
        "mejora_absoluta": {
            "F1_at_3": round(diff_f1, 4),
            "NDCG_at_10": round(diff_ndcg, 4),
        },
        "mejora_porcentual": {
            "F1_at_3": f"{pct_f1:+.2f}%",
            "NDCG_at_10": f"{pct_ndcg:+.2f}%",
        }
    }

    print("\n================================================================================")
    print("RESULTADOS DEL BENCHMARK COMPARATIVO")
    print("================================================================================")
    print(f" Consultas evaluadas       : {reporte['total_consultas_evaluadas']}")
    print(f" BASELINE (FAISS + BM25)   : F1@3={reporte['baseline_vectorial']['F1_at_3']:.4f}, NDCG@10={reporte['baseline_vectorial']['NDCG_at_10']:.4f}")
    print(f" HÍBRIDO CON GRAFO         : F1@3={reporte['hibrido_con_grafo']['F1_at_3']:.4f}, NDCG@10={reporte['hibrido_con_grafo']['NDCG_at_10']:.4f}")
    print(f" IMPACTO / DELTA EN F1@3   : {diff_f1:+.4f} ({reporte['mejora_porcentual']['F1_at_3']})")
    print(f" IMPACTO / DELTA EN NDCG@10: {diff_ndcg:+.4f} ({reporte['mejora_porcentual']['NDCG_at_10']})")
    print("================================================================================")

    return reporte


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=20, help="Número de preguntas del benchmark a evaluar")
    args = parser.parse_args()

    evaluar_sistema_con_y_sin_grafo(limit=args.limit)
