import unittest

from analisis_senales import clasificar


class TestAnalisisSenales(unittest.TestCase):
    def test_detecta_fallo_lexical(self):
        metricas = {
            "bm25": {"ndcg": 0.1}, "dense": {"ndcg": 0.8},
            "combsum_alpha_01": {"ndcg": 0.7}, "rrf": {"ndcg": 0.7},
        }
        self.assertEqual(clasificar(metricas), "lexical_bm25")


if __name__ == "__main__":
    unittest.main()
