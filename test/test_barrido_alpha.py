import unittest

from barrido_alpha import ALPHAS


class TestBarridoAlpha(unittest.TestCase):
    def test_barrido_completo(self):
        self.assertEqual(ALPHAS, tuple(index / 10 for index in range(11)))


if __name__ == "__main__":
    unittest.main()
