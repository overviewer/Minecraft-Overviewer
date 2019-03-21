import unittest
import tempfile

from contrib.cyrillic_convert import convert


class TestCyrillicConvert(unittest.TestCase):
    def test_convert(self):
        gibberish = '{chunk: [-2, 0],y: 65,msg: "ðåëèãèè",x: -20,z: 4,type: "sign"}'
        cyrillic = '{chunk: [-2, 0],y: 65,msg: "религии",x: -20,z: 4,type: "sign"}'
        self.assertEqual(convert(gibberish), cyrillic)
