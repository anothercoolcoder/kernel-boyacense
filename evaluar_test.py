"""Evalúa configuraciones congeladas sobre test reservado."""

from __future__ import annotations

import json
from pathlib import Path

from benchmark_rag import cargar_dataset_jsonl, f1_at_3, ndcg_at_k
from pilot_split_variance import metricas
from recuperar.recuperar import BuscadorHibrido

ROOT = Path(__file__).parent
DATASET = ROOT / "context/benchmark_multilingue.jsonl"
ALPHA_REPORT = ROOT / "resultados_barrido_alpha.json"
OUTPUT = ROOT / "resultados_test_alpha.json"


def main() -> None:
    records = cargar_dataset_jsonl(DATASET)
    if any(record.get("annotation_status") != "approved" for record in records):
        raise ValueError("Dataset contiene anotaciones no aprobadas")
    alpha_report = json.loads(ALPHA_REPORT.read_text(encoding="utf-8"))
    test_intents = set(alpha_report["test_intents_reserved"])
    test_records = [record for record in records if record["intent_id"] in test_intents]
    alphas = (0.0, float(alpha_report["selected_alpha"]), 0.5)
    buscador = BuscadorHibrido()
    resultados = {}

    for alpha in dict.fromkeys(alphas):
        scores = {}
        for record in test_records:
            result = buscador.buscar(
                record["query"], top_k_docs=3, top_k_chunks=10,
                fusion_method="combsum", normalization_method="minmax", alpha=alpha,
            )
            scores[record["query_id"]] = {
                "ndcg": ndcg_at_k([item["chunk_id"] for item in result["fragments"]], record["graded_relevance"]),
                "f1": f1_at_3([item["doc_id"] for item in result["documents"]], record["relevant_doc_ids"]),
            }
        resultados[str(alpha)] = metricas(test_records, scores)

    reporte = {
        "dataset": str(DATASET), "split": "test", "test_intents": sorted(test_intents),
        "normalization_method": "minmax", "alphas": list(dict.fromkeys(alphas)),
        "selected_alpha_from_dev": alpha_report["selected_alpha"], "metrics_by_alpha": resultados,
    }
    OUTPUT.write_text(json.dumps(reporte, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(reporte, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
