import unittest
from src.lob_modeling.models.kyle import KyleModel
from src.lob_modeling.models.almgren_chriss import AlmgrenChriss2000

class TestModels(unittest.TestCase):
    def test_kyle_init(self):
        model = KyleModel()
        self.assertIsNotNone(model)

    def test_almgren_init(self):
        model = AlmgrenChriss2000()
        self.assertIsNotNone(model)

if __name__ == '__main__':
    unittest.main()
