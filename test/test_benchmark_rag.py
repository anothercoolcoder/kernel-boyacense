import unittest

from benchmark_rag import cargar_dataset_jsonl, f1_at_3, ndcg_at_k, ejecutar_benchmark


class TestBenchmarkRag(unittest.TestCase):
    def test_dataset_multilingue_conserva_idioma(self):
        registros = cargar_dataset_jsonl("context/benchmark_multilingue.jsonl")
        self.assertEqual(len(registros), 150)
        self.assertEqual({registro["language"] for registro in registros}, {"es", "en", "pt"})
        self.assertEqual(len({registro["intent_id"] for registro in registros}), 50)

    def test_ndcg_prioriza_chunk_relevante(self):
        relevancia = {"c1": 2, "c2": 1}
        self.assertEqual(ndcg_at_k(["c1", "c2"], relevancia), 1.0)
        self.assertLess(ndcg_at_k(["c2", "c1"], relevancia), 1.0)

    def test_f1_documental(self):
        self.assertAlmostEqual(f1_at_3(["d1", "d2", "d3"], ["d1"]), 0.5)
        self.assertEqual(f1_at_3(["d1", "d2", "d3"], ["d4"]), 0.0)

    def test_metricas_solo_con_ground_truth(self):
        registros = [{
            "query_id": "q001",
            "query": "consulta",
            "relevant_doc_ids": ["d1"],
            "graded_relevance": {"c1": 2},
        }]
        resultado = ejecutar_benchmark(
            registros,
            lambda _query: {
                "documents": [{"doc_id": "d1"}, {"doc_id": "d2"}, {"doc_id": "d3"}],
                "fragments": [
                    {"doc_id": doc, "chunk_id": chunk, "text": "texto"}
                    for doc, chunk in [("d1", "c1"), ("d2", "c2"), ("d3", "c3")]
                ] + [
                    {"doc_id": "d1", "chunk_id": f"c{index}", "text": "texto"}
                    for index in range(4, 11)
                ],
            },
        )
        self.assertEqual(resultado["annotated_queries"], 1)
        self.assertEqual(resultado["ndcg_at_10"], 1.0)
        self.assertAlmostEqual(resultado["f1_at_3"], 0.5)


if __name__ == "__main__":
    unittest.main()
