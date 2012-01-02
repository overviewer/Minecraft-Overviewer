import unittest

import os

from overviewer_core import world

class ExampleWorldTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Make sure that test/data/worlds/example exists
        # if it doesn't, then give a little 
        if not os.path.exists("test/data/worlds/exmaple"):
            raise Exception("test data doesn't exist.  Maybe you need to init/update your submodule?")

    def test_basic(self):
        "Basic test of the world constructor and regionset constructor"
        w = world.World("test/data/worlds/exmaple")

        regionsets = w.get_regionsets()
        self.assertEquals(len(regionsets), 3)
        
         

if __name__ == "__main__":
    unittest.main()
