import unittest

from evaluar_test import ALPHA_REPORT


class TestEvaluarTest(unittest.TestCase):
    def test_reporte_alpha_dev_existe(self):
        self.assertTrue(ALPHA_REPORT.exists())


if __name__ == "__main__":
    unittest.main()
