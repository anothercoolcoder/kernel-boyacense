"""Barrido de alpha con normalización min-max congelada."""

from __future__ import annotations

import json
from pathlib import Path

from benchmark_rag import cargar_dataset_jsonl, f1_at_3, ndcg_at_k
from pilot_split_variance import clave_seleccion, metricas
from recuperar.recuperar import BuscadorHibrido

ROOT = Path(__file__).parent
DATASET = ROOT / "context/benchmark_multilingue.jsonl"
OUTPUT = ROOT / "resultados_barrido_alpha.json"
ALPHAS = tuple(index / 10 for index in range(11))


def main() -> None:
    records = cargar_dataset_jsonl(DATASET)
    if any(record.get("annotation_status") != "approved" for record in records):
        raise ValueError("Dataset contiene anotaciones no aprobadas")

    intents = sorted({record["intent_id"] for record in records})
    dev_intents, test_intents = intents[:30], intents[30:]
    dev_records = [record for record in records if record["intent_id"] in dev_intents]
    buscador = BuscadorHibrido()
    resultados = {}

    for alpha in ALPHAS:
        scores = {}
        for record in dev_records:
            result = buscador.buscar(
                record["query"], top_k_docs=3, top_k_chunks=10,
                fusion_method="combsum", normalization_method="minmax", alpha=alpha,
            )
            scores[record["query_id"]] = {
                "ndcg": ndcg_at_k(
                    [item["chunk_id"] for item in result["fragments"]],
                    record["graded_relevance"],
                ),
                "f1": f1_at_3(
                    [item["doc_id"] for item in result["documents"]],
                    record["relevant_doc_ids"],
                ),
            }
        resultados[str(alpha)] = metricas(dev_records, scores)

    alpha_elegido = max(ALPHAS, key=lambda alpha: clave_seleccion(resultados[str(alpha)]))
    reporte = {
        "dataset": str(DATASET), "normalization_method": "minmax",
        "dev_intents": dev_intents, "test_intents_reserved": test_intents,
        "alphas": list(ALPHAS),
        "selection_rule": "worst_language_ndcg, language_gap_ndcg, global_ndcg, global_f1",
        "selected_alpha": alpha_elegido, "metrics_by_alpha": resultados,
    }
    OUTPUT.write_text(json.dumps(reporte, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "selected_alpha": alpha_elegido,
        "selected_metrics": resultados[str(alpha_elegido)],
        "output": str(OUTPUT),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
