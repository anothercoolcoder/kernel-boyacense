"""Módulo de Fusión RRF (Reciprocal Rank Fusion) para combinar FAISS, BM25 y Grafo.

Cumple estrictamente con las especificaciones del proyecto:
- k0 = 60 (constante RRF recomendada).
- Trata al grafo de conocimiento como un canal de recuperación equivalente a un índice más.
- 100% determinista.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Sequence, Tuple


def normalizar_resultado(item: Any) -> Tuple[str, str, float]:
    """Convierte dict o tuple a (doc_id, chunk_id, score)."""
    if isinstance(item, dict):
        return (str(item.get("doc_id", "")), str(item.get("chunk_id", "")), float(item.get("score", 0.0)))
    elif isinstance(item, (list, tuple)) and len(item) >= 2:
        doc_id = str(item[0])
        chunk_id = str(item[1])
        score = float(item[2]) if len(item) >= 3 else 0.0
        return (doc_id, chunk_id, score)
    return ("", "", 0.0)


def fusionar_rrf(
    canales: Dict[str, Sequence[Any]],
    k0: int = 60,
    pesos: Dict[str, float] | None = None,
    top_k: int = 50
) -> List[Dict[str, Any]]:
    """Combina listas de resultados provenientes de múltiples canales (FAISS, BM25, Grafo) usando RRF.

    Score RRF = sum_c ( peso_c / (k0 + rank_c) )
    """
    scores_rrf: Dict[Tuple[str, str], float] = defaultdict(float)
    canales_presentes: Dict[Tuple[str, str], List[str]] = defaultdict(list)
    pesos_dict = pesos or {}

    for nombre_canal, resultados in canales.items():
        peso_canal = pesos_dict.get(nombre_canal, 1.0)
        for rank, item in enumerate(resultados, start=1):
            doc_id, chunk_id, _ = normalizar_resultado(item)
            if not doc_id or not chunk_id:
                continue

            key = (doc_id, chunk_id)
            rrf_score = peso_canal / (k0 + rank)
            scores_rrf[key] += rrf_score
            canales_presentes[key].append(nombre_canal)

    lista_final = []
    for (d_id, c_id), score in scores_rrf.items():
        lista_final.append({
            "doc_id": d_id,
            "chunk_id": c_id,
            "score": float(score),
            "canales": canales_presentes[(d_id, c_id)]
        })

    lista_final.sort(key=lambda x: x["score"], reverse=True)
    return lista_final[:top_k]
