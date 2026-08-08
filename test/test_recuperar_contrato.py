import unittest
from types import SimpleNamespace

import numpy as np

from recuperar.recuperar import BM25Nativo, BuscadorHibrido, contar_palabras


class IndiceFalso:
    def search(self, _query_vector, _candidate_k):
        return np.array([[0.9]], dtype=np.float32), np.array([[0]], dtype=np.int64)


class ModeloFalso:
    def embed_query(self, _query):
        return [1.0]


class TestContratoRecuperacion(unittest.TestCase):
    def test_normalizacion_combsum(self):
        scores = BuscadorHibrido._normalizar_scores({1: 2.0, 2: 4.0})
        self.assertEqual(scores, {1: 0.0, 2: 1.0})

    def test_recorte_conserva_oraciones_y_limite(self):
        texto = "Primera oración completa. Segunda oración demasiado larga."
        resultado = BuscadorHibrido._recortar_a_250_palabras(texto, max_words=3)
        self.assertEqual(resultado, "Primera oración completa.")
        self.assertLessEqual(contar_palabras(resultado), 250)

    def test_padding_usa_ids_unicos_y_fragmentos_coherentes(self):
        buscador = object.__new__(BuscadorHibrido)
        buscador.metadatos = [
            {"doc_id": "DOC-A", "chunk_id": "A-1", "texto": "Texto A uno."},
            {"doc_id": "DOC-B", "chunk_id": "B-1", "texto": "Texto B uno."},
            {"doc_id": "DOC-C", "chunk_id": "C-1", "texto": "Texto C uno."},
            {"doc_id": "DOC-C", "chunk_id": "C-2", "texto": "Texto C dos."},
        ]
        buscador.bm25 = SimpleNamespace(search=lambda _query, top_k: [(0, 1.0)])
        buscador.faiss_index = IndiceFalso()
        buscador.embeddings_model = ModeloFalso()

        resultado = buscador.buscar("consulta", top_k_docs=3, top_k_chunks=4)
        docs = [item["doc_id"] for item in resultado["documents"]]
        chunks = [item["chunk_id"] for item in resultado["fragments"]]

        self.assertEqual(docs, ["DOC-A", "DOC-B", "DOC-C"])
        self.assertEqual(len(set(docs)), 3)
        self.assertEqual(len(chunks), 4)
        self.assertEqual(len(set(chunks)), 4)
        self.assertTrue(all(item["doc_id"] in docs for item in resultado["fragments"]))
        self.assertEqual(resultado["diagnostics"]["padded_documents"], ["DOC-B", "DOC-C"])


if __name__ == "__main__":
    unittest.main()
