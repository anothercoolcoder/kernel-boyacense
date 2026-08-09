"""Pilot de varianza por split para normalizaciones CombSUM."""

from __future__ import annotations

import json
import random
import statistics
from collections import Counter
from pathlib import Path

from benchmark_rag import cargar_dataset_jsonl, f1_at_3, ndcg_at_k
from recuperar.recuperar import BuscadorHibrido

ROOT = Path(__file__).parent
DATASET = ROOT / "context/benchmark_multilingue.jsonl"
OUTPUT = ROOT / "resultados_pilot_split_variance.json"
METHODS = ("minmax", "clipped_minmax", "percentile", "zsigmoid")


def metricas(records: list[dict], scores: dict[str, dict[str, float]]) -> dict[str, float]:
    por_idioma: dict[str, list[float]] = {}
    ndcgs: list[float] = []
    f1s: list[float] = []
    for record in records:
        valor = scores[record["query_id"]]
        ndcgs.append(valor["ndcg"])
        f1s.append(valor["f1"])
        por_idioma.setdefault(record["language"], []).append(valor["ndcg"])
    medias = {language: sum(values) / len(values) for language, values in por_idioma.items()}
    return {
        "ndcg": sum(ndcgs) / len(ndcgs),
        "f1": sum(f1s) / len(f1s),
        "worst_language_ndcg": min(medias.values()),
        "language_gap_ndcg": max(medias.values()) - min(medias.values()),
    }


def clave_seleccion(metric: dict[str, float]) -> tuple[float, float, float, float]:
    return (
        metric["worst_language_ndcg"],
        -metric["language_gap_ndcg"],
        metric["ndcg"],
        metric["f1"],
    )


def bootstrap_ci(values: list[float], seed: int, iterations: int = 1000) -> tuple[float, float]:
    rng = random.Random(seed)
    means = []
    for _ in range(iterations):
        sample = [values[rng.randrange(len(values))] for _ in values]
        means.append(sum(sample) / len(sample))
    means.sort()
    return means[int(iterations * 0.025)], means[int(iterations * 0.975)]


def main() -> None:
    records = cargar_dataset_jsonl(DATASET)
    if any(record.get("annotation_status") != "approved" for record in records):
        raise ValueError("Dataset contiene anotaciones no aprobadas")

    intents = sorted({record["intent_id"] for record in records})
    dev_intents, test_intents = intents[:30], intents[30:]
    dev_records = [record for record in records if record["intent_id"] in dev_intents]
    buscador = BuscadorHibrido()
    scores: dict[str, dict[str, dict[str, float]]] = {method: {} for method in METHODS}

    for method in METHODS:
        for record in dev_records:
            result = buscador.buscar(
                record["query"], top_k_docs=3, top_k_chunks=10,
                fusion_method="combsum", normalization_method=method, alpha=0.5,
            )
            scores[method][record["query_id"]] = {
                "ndcg": ndcg_at_k(
                    [item["chunk_id"] for item in result["fragments"]],
                    record["graded_relevance"],
                ),
                "f1": f1_at_3(
                    [item["doc_id"] for item in result["documents"]],
                    record["relevant_doc_ids"],
                ),
            }

    seed_results = []
    selected_methods = Counter()
    for seed in range(20):
        shuffled = list(dev_intents)
        random.Random(seed).shuffle(shuffled)
        folds = [shuffled[index::5] for index in range(5)]
        heldout_metrics = []
        heldout_methods = []
        for fold in folds:
            heldout = set(fold)
            train = [r for r in dev_records if r["intent_id"] not in heldout]
            validation = [r for r in dev_records if r["intent_id"] in heldout]
            candidates = {method: metricas(train, scores[method]) for method in METHODS}
            selected = max(METHODS, key=lambda method: clave_seleccion(candidates[method]))
            selected_methods[selected] += 1
            heldout_methods.append(selected)
            heldout_metrics.append(metricas(validation, scores[selected]))
        seed_results.append({
            "seed": seed,
            "selected_methods": heldout_methods,
            "ndcg": sum(item["ndcg"] for item in heldout_metrics) / len(heldout_metrics),
            "f1": sum(item["f1"] for item in heldout_metrics) / len(heldout_metrics),
        })

    intent_scores = {
        method: {
            intent: sum(
                scores[method][r["query_id"]]["ndcg"]
                for r in dev_records if r["intent_id"] == intent
            ) / 3
            for intent in dev_intents
        }
        for method in METHODS
    }
    baseline = intent_scores["minmax"]
    comparison = {}
    ties = 0
    for index, method in enumerate(METHODS[1:]):
        differences = [intent_scores[method][i] - baseline[i] for i in dev_intents]
        low, high = bootstrap_ci(differences, seed=100 + index)
        comparison[method] = {"difference_ci_95": [low, high]}
        ties += int(low <= 0.0 <= high)

    split_ndcg = [item["ndcg"] for item in seed_results]
    split_f1 = [item["f1"] for item in seed_results]
    report = {
        "dataset": str(DATASET), "dev_intents": dev_intents,
        "test_intents_reserved": test_intents, "methods": list(METHODS),
        "alpha_control": 0.5, "seeds": 20, "folds": 5,
        "selection_rule": "worst_language_ndcg, language_gap_ndcg, global_ndcg, global_f1",
        "split_variance": {
            "ndcg_mean": sum(split_ndcg) / len(split_ndcg),
            "ndcg_std": statistics.stdev(split_ndcg),
            "f1_mean": sum(split_f1) / len(split_f1),
            "f1_std": statistics.stdev(split_f1),
        },
        "selected_method_counts": dict(selected_methods),
        "normalization_difference_bootstrap": comparison,
        "normalization_tie_rate": ties / (len(METHODS) - 1),
        "per_seed": seed_results,
    }
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "per_seed"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
