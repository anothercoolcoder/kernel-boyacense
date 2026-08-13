import json
import unittest
from functools import lru_cache
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
METADATA_PATH = ROOT / "base_vectorial" / "encoder_multilingual-e5-large-instruct" / "metadata.jsonl"
MAPPED_DATASET_PATH = ROOT / "context" / "benchmark_rag_mapped.jsonl"
MAPPED_REPORT_PATH = ROOT / "salida" / "benchmark_mapped.json"


@lru_cache(maxsize=1)
def _metadata_doc_ids() -> set[str]:
    doc_ids: set[str] = set()
    with METADATA_PATH.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            doc_id = item.get("doc_id")
            if doc_id:
                doc_ids.add(doc_id)
    return doc_ids


def _load_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


class TestBenchmarkAlignment(unittest.TestCase):
    def test_mapped_relevant_docs_exist_in_metadata(self):
        if not MAPPED_DATASET_PATH.exists():
            self.skipTest("benchmark_rag_mapped.jsonl no existe")

        registros = _load_jsonl(MAPPED_DATASET_PATH)
        relevant_doc_ids = {
            doc_id
            for registro in registros
            for doc_id in registro.get("relevant_doc_ids", [])
        }
        metadata_doc_ids = _metadata_doc_ids()

        faltantes = sorted(relevant_doc_ids - metadata_doc_ids)
        self.assertEqual(faltantes, [])

    def test_mapped_report_has_some_relevant_hits(self):
        if not MAPPED_DATASET_PATH.exists() or not MAPPED_REPORT_PATH.exists():
            self.skipTest("artefactos de benchmark_mapped no existen")

        registros = _load_jsonl(MAPPED_DATASET_PATH)
        por_id = {registro["query_id"]: registro for registro in registros}
        reporte = json.loads(MAPPED_REPORT_PATH.read_text(encoding="utf-8"))

        hits = 0
        for item in reporte.get("per_query", []):
            registro = por_id.get(item.get("query_id"))
            if not registro:
                continue
            relevantes = set(registro.get("relevant_doc_ids", []))
            recuperados = set(item.get("retrieved_doc_ids", []))
            if relevantes & recuperados:
                hits += 1

        self.assertGreater(hits, 0)
        self.assertGreater(reporte.get("f1_at_3", 0.0), 0.0)


if __name__ == "__main__":
    unittest.main()
