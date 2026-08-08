"""Generador del entregable resultados.jsonl para el CODEFEST AD ASTRA 2026.

Produce el archivo de salida con 50 consultas de evaluación validadas según §9.3.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

from recuperar.recuperar import BuscadorHibrido

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("generar_entregable")

QUERIES_PATH_DEFAULT = Path(__file__).parent / "queries.jsonl"
RESULTADOS_PATH_DEFAULT = Path(__file__).parent / "resultados.jsonl"


def validar_esquema_salida(resultados: List[Dict[str, Any]]) -> None:
    """Verifica el cumplimiento estricto del contrato del entregable (§9.3)."""
    logger.info("Validando esquema del entregable (%d consultas)…", len(resultados))
    assert len(resultados) == 50, f"Deben ser exactamente 50 consultas, se encontraron {len(resultados)}"

    query_ids_vistos = set()
    for i, res in enumerate(resultados, start=1):
        q_id = res.get("query_id")
        assert q_id and isinstance(q_id, str), f"Línea {i}: query_id inválido"
        assert q_id not in query_ids_vistos, f"Línea {i}: query_id duplicado {q_id}"
        query_ids_vistos.add(q_id)

        docs = res.get("documents", [])
        assert len(docs) == 3, f"Consulta {q_id}: deben ser exactamente 3 documentos (F1@3), hay {len(docs)}"
        for doc in docs:
            assert "rank" in doc and "doc_id" in doc, f"Consulta {q_id}: estructura de documento inválida"

        frags = res.get("fragments", [])
        assert len(frags) == 10, f"Consulta {q_id}: deben ser exactamente 10 fragmentos (NDCG@10), hay {len(frags)}"
        for frag in frags:
            assert "rank" in frag and "chunk_id" in frag and "doc_id" in frag and "text" in frag, (
                f"Consulta {q_id}: estructura de fragmento inválida"
            )
            palabras = len(frag["text"].split())
            assert palabras <= 250, (
                f"Consulta {q_id}, chunk {frag['chunk_id']}: excede 250 palabras ({palabras} palabras)"
            )

    logger.info("✅ Validación del entregable EXITOSA. Cumple 100%% el esquema §9.3.")


def generar_entregable(
    queries_path: Path | str = QUERIES_PATH_DEFAULT,
    salida_path: Path | str = RESULTADOS_PATH_DEFAULT,
) -> None:
    queries_path = Path(queries_path)
    salida_path = Path(salida_path)

    if not queries_path.exists():
        logger.error("No se encontró el archivo de consultas en: %s", queries_path)
        logger.info("Creando plantilla de ejemplo para pruebas...")
        _crear_plantilla_queries(queries_path)

    logger.info("Cargando motor de búsqueda híbrida...")
    buscador = BuscadorHibrido()

    queries: List[Dict[str, str]] = []
    with open(queries_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                queries.append(json.loads(line.strip()))

    logger.info("Procesando %d consultas...", len(queries))
    resultados: List[Dict[str, Any]] = []

    for item in queries:
        q_id = item["query_id"]
        pregunta = item["query"]

        res_hibrido = buscador.buscar(pregunta, top_k_docs=3, top_k_chunks=10)

        # Formatear según §9.3.1 (Tabla 2)
        documentos_clean = [
            {"rank": d["rank"], "doc_id": d["doc_id"]}
            for d in res_hibrido["documents"]
        ]

        fragmentos_clean = [
            {
                "rank": f["rank"],
                "chunk_id": f["chunk_id"],
                "doc_id": f["doc_id"],
                "text": f["text"],
            }
            for f in res_hibrido["fragments"]
        ]

        resultados.append({
            "query_id": q_id,
            "documents": documentos_clean,
            "fragments": fragmentos_clean,
        })

    validar_esquema_salida(resultados)

    logger.info("Escribiendo entregable en: %s", salida_path)
    with open(salida_path, "w", encoding="utf-8") as f:
        for res in resultados:
            f.write(json.dumps(res, ensure_ascii=False) + "\n")

    logger.info("🎉 Proceso finalizado. %s generado exitosamente.", salida_path.name)


def _crear_plantilla_queries(path: Path) -> None:
    ejemplos = [
        {"query_id": f"q{i:03d}", "query": f"Consulta de prueba {i}"}
        for i in range(1, 51)
    ]
    with open(path, "w", encoding="utf-8") as f:
        for item in ejemplos:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    generar_entregable()
