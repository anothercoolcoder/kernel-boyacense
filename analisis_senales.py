"""Analiza fallos por señal en consultas débiles EN/PT."""

from __future__ import annotations

import json
from pathlib import Path

from benchmark_rag import cargar_dataset_jsonl, f1_at_3, ndcg_at_k
from recuperar.recuperar import BuscadorHibrido

ROOT = Path(__file__).parent
DATASET = ROOT / "context/benchmark_multilingue.jsonl"
BASELINE = ROOT / "resultados_multilingue_combsum_gt_corregido.json"
OUTPUT = ROOT / "resultados_analisis_senales.json"
VARIANTS = {
    "bm25": {"fusion_method": "combsum", "alpha": 1.0},
    "dense": {"fusion_method": "combsum", "alpha": 0.0},
    "combsum_alpha_01": {"fusion_method": "combsum", "alpha": 0.1},
    "rrf": {"fusion_method": "rrf", "alpha": 0.5},
}


def clasificar(metricas: dict[str, float]) -> str:
    bm25 = metricas["bm25"]["ndcg"]
    dense = metricas["dense"]["ndcg"]
    combsum = metricas["combsum_alpha_01"]["ndcg"]
    rrf = metricas["rrf"]["ndcg"]
    if bm25 < 0.3 and dense >= 0.6:
        return "lexical_bm25"
    if dense < 0.3 and bm25 >= 0.6:
        return "semantic_faiss"
    if max(bm25, dense) >= 0.6 and max(combsum, rrf) < 0.4:
        return "fusion"
    if max(bm25, dense, combsum, rrf) < 0.4:
        return "corpus_or_chunking"
    return "mixed_or_annotation"


def main() -> None:
    records = cargar_dataset_jsonl(DATASET)
    by_id = {record["query_id"]: record for record in records}
    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    ranked = [item for item in baseline["per_query"] if by_id[item["query_id"]]["language"] in {"en", "pt"}]
    ranked.sort(key=lambda item: item["ndcg_at_10"])
    selected_ids = [item["query_id"] for item in ranked[:30]]
    buscador = BuscadorHibrido()
    report = []

    for query_id in selected_ids:
        record = by_id[query_id]
        variant_report = {}
        for name, config in VARIANTS.items():
            result = buscador.buscar(
                record["query"], top_k_docs=3, top_k_chunks=10,
                normalization_method="minmax", **config,
            )
            variant_report[name] = {
                "ndcg": ndcg_at_k([item["chunk_id"] for item in result["fragments"]], record["graded_relevance"]),
                "f1": f1_at_3([item["doc_id"] for item in result["documents"]], record["relevant_doc_ids"]),
                "fragments": result["fragments"],
            }
        report.append({
            "query_id": query_id, "language": record["language"],
            "query": record["query"], "answer": record.get("answer", ""),
            "classification": clasificar(variant_report), "variants": variant_report,
        })

    output = {
        "source": str(BASELINE), "query_count": len(report), "selected_queries": selected_ids,
        "variants": VARIANTS,
        "classification_counts": {
            label: sum(item["classification"] == label for item in report)
            for label in sorted({item["classification"] for item in report})
        },
        "queries": report,
    }
    OUTPUT.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"query_count": output["query_count"], "classification_counts": output["classification_counts"], "output": str(OUTPUT)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
