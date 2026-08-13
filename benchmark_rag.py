"""Benchmark local de recuperación, sin modelos generativos."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Callable, Iterable


def cargar_preguntas(path: Path | str) -> list[dict[str, str]]:
    """Carga preguntas de texto y asigna IDs estables según orden de archivo."""
    preguntas: list[dict[str, str]] = []
    with open(path, encoding="utf-8") as archivo:
        for linea in archivo:
            query = linea.strip()
            if not query or query.startswith("#"):
                continue
            preguntas.append({"query_id": f"q{len(preguntas) + 1:03d}", "query": query})
    return preguntas


def cargar_anotaciones(path: Path | str) -> dict[str, dict[str, Any]]:
    """Carga ground truth JSONL indexado por ``query_id``."""
    anotaciones: dict[str, dict[str, Any]] = {}
    with open(path, encoding="utf-8") as archivo:
        for numero, linea in enumerate(archivo, 1):
            if not linea.strip():
                continue
            item = json.loads(linea)
            query_id = item.get("query_id")
            if not query_id or query_id in anotaciones:
                raise ValueError(f"Anotación inválida o duplicada en línea {numero}")
            anotaciones[query_id] = item
    return anotaciones


def cargar_dataset_jsonl(path: Path | str) -> list[dict[str, Any]]:
    """Carga dataset benchmark completo, incluyendo idioma e intent_id."""
    registros: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as archivo:
        for numero, linea in enumerate(archivo, 1):
            if not linea.strip():
                continue
            registro = json.loads(linea)
            if not registro.get("query_id") or not registro.get("query"):
                raise ValueError(f"Registro inválido en línea {numero}")
            registros.append(registro)
    return registros


def ndcg_at_k(retrieved_chunk_ids: Iterable[str], graded_relevance: dict[str, int], k: int = 10) -> float:
    """Calcula NDCG@k con grados 0, 1 y 2."""
    retrieved = list(retrieved_chunk_ids)[:k]
    dcg = sum(
        (2 ** graded_relevance.get(chunk_id, 0) - 1) / math.log2(rank + 2)
        for rank, chunk_id in enumerate(retrieved)
    )
    ideal = sorted((max(0, int(value)) for value in graded_relevance.values()), reverse=True)[:k]
    idcg = sum((2**value - 1) / math.log2(rank + 2) for rank, value in enumerate(ideal))
    return dcg / idcg if idcg else 0.0


def f1_at_3(retrieved_doc_ids: Iterable[str], relevant_doc_ids: Iterable[str]) -> float:
    """Calcula F1 documental usando hasta tres documentos únicos."""
    retrieved = set(list(retrieved_doc_ids)[:3])
    relevant = set(relevant_doc_ids)
    if not retrieved or not relevant:
        return 0.0
    overlap = len(retrieved & relevant)
    precision = overlap / len(retrieved)
    recall = overlap / len(relevant)
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def validar_resultado(resultado: dict[str, Any], top_k_docs: int = 3, top_k_chunks: int = 10) -> None:
    """Falla si recuperación rompe contrato cardinal o coherencia documental."""
    documents = resultado.get("documents", [])
    fragments = resultado.get("fragments", [])
    if len(documents) != top_k_docs or len(fragments) != top_k_chunks:
        raise AssertionError("Resultado incompleto: cardinalidad fuera de contrato")

    doc_ids = [item["doc_id"] for item in documents]
    chunk_ids = [item["chunk_id"] for item in fragments]
    if len(set(doc_ids)) != len(doc_ids) or len(set(chunk_ids)) != len(chunk_ids):
        raise AssertionError("Resultado contiene IDs duplicados")
    if any(item["doc_id"] not in doc_ids for item in fragments):
        raise AssertionError("Fragmento fuera de documents")
    if any(len(item.get("text", "").split()) > 250 for item in fragments):
        raise AssertionError("Fragmento excede 250 palabras")


def ejecutar_benchmark(
    registros: list[dict[str, Any]],
    buscar: Callable[[str], dict[str, Any]],
    validar: bool = True,
) -> dict[str, Any]:
    """Ejecuta consultas y reporta métricas disponibles y diagnósticos."""
    por_consulta: list[dict[str, Any]] = []
    ndcgs: list[float] = []
    f1s: list[float] = []
    incompletas = 0

    for registro in registros:
        resultado = buscar(registro["query"])
        if validar:
            try:
                validar_resultado(resultado)
            except AssertionError:
                incompletas += 1

        retrieved_docs = [item.get("doc_id") for item in resultado.get("documents", [])]
        retrieved_chunks = [item.get("chunk_id") for item in resultado.get("fragments", [])]
        anotado = (
            registro.get("annotation_status", "approved") == "approved"
            and bool(registro.get("relevant_doc_ids"))
            and bool(registro.get("graded_relevance"))
        )
        detalle: dict[str, Any] = {
            "query_id": registro["query_id"],
            "retrieved_doc_ids": retrieved_docs,
            "retrieved_chunk_ids": retrieved_chunks,
            "annotated": anotado,
        }
        if anotado:
            ndcg = ndcg_at_k(retrieved_chunks, registro["graded_relevance"])
            f1 = f1_at_3(retrieved_docs, registro["relevant_doc_ids"])
            ndcgs.append(ndcg)
            f1s.append(f1)
            detalle.update({"ndcg_at_10": ndcg, "f1_at_3": f1})
        por_consulta.append(detalle)

    total = len(registros)
    anotadas = len(ndcgs)
    return {
        "total_queries": total,
        "annotated_queries": anotadas,
        "unannotated_queries": total - anotadas,
        "ndcg_at_10": sum(ndcgs) / anotadas if anotadas else None,
        "f1_at_3": sum(f1s) / anotadas if anotadas else None,
        "incomplete_rate": incompletas / total if total else 0.0,
        "per_query": por_consulta,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, default=Path("preguntas.txt"))
    parser.add_argument("--dataset", type=Path, help="Dataset JSONL completo; reemplaza --questions")
    parser.add_argument("--annotations", type=Path)
    parser.add_argument("--output", type=Path, default=Path("resultados_benchmark.json"))
    parser.add_argument("--fusion-method", choices=("combsum", "rrf"), default="combsum")
    parser.add_argument(
        "--normalization-method",
        choices=("minmax", "clipped_minmax", "percentile", "zsigmoid"),
        default="minmax",
    )
    parser.add_argument("--alpha", type=float, default=0.4)
    args = parser.parse_args()

    from recuperar.recuperar import BuscadorHibrido

    registros = cargar_dataset_jsonl(args.dataset) if args.dataset else cargar_preguntas(args.questions)
    if args.annotations:
        anotaciones = cargar_anotaciones(args.annotations)
        for registro in registros:
            registro.update(anotaciones.get(registro["query_id"], {}))

    buscador = BuscadorHibrido()
    reporte = ejecutar_benchmark(
        registros,
        lambda query: buscador.buscar(
            query,
            top_k_docs=3,
            top_k_chunks=10,
            fusion_method=args.fusion_method,
            normalization_method=args.normalization_method,
            alpha=args.alpha,
        ),
    )
    reporte["fusion_method"] = args.fusion_method
    reporte["normalization_method"] = args.normalization_method
    reporte["alpha"] = args.alpha
    with open(args.output, "w", encoding="utf-8") as archivo:
        json.dump(reporte, archivo, ensure_ascii=False, indent=2)
    print(json.dumps({key: value for key, value in reporte.items() if key != "per_query"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
