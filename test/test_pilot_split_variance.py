import unittest

from pilot_split_variance import clave_seleccion


class TestPilotSplitVariance(unittest.TestCase):
    def test_regla_prioriza_peor_idioma(self):
        mejor_peor = {"worst_language_ndcg": 0.5, "language_gap_ndcg": 0.3, "ndcg": 0.8, "f1": 0.8}
        peor_global = {"worst_language_ndcg": 0.4, "language_gap_ndcg": 0.01, "ndcg": 0.9, "f1": 0.9}
        self.assertGreater(clave_seleccion(mejor_peor), clave_seleccion(peor_global))


if __name__ == "__main__":
    unittest.main()
