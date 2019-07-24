import unittest

import os

from overviewer_core import world

class ExampleWorldTest(unittest.TestCase):
    @unittest.skip("Broken old garbage, find a newer world")
    def test_basic(self):
        "Basic test of the world constructor and regionset constructor"
        if not os.path.exists("test/data/worlds/exmaple"):
            raise unittest.SkipTest("test data doesn't exist.  Maybe you need to init/update your submodule?")
        w = world.World("test/data/worlds/exmaple")

        regionsets = w.get_regionsets()
        self.assertEqual(len(regionsets), 3)

        regionset = regionsets[0]
        self.assertEqual(regionset.get_region_path(0,0), 'test/data/worlds/exmaple/DIM-1/region/r.0.0.mcr')
        self.assertEqual(regionset.get_region_path(-1,0), 'test/data/worlds/exmaple/DIM-1/region/r.-1.0.mcr')
        self.assertEqual(regionset.get_region_path(1,1), 'test/data/worlds/exmaple/DIM-1/region/r.0.0.mcr')
        self.assertEqual(regionset.get_region_path(35,35), None)

        # a few random chunks.  reference timestamps fetched with libredstone
        self.assertEqual(regionset.get_chunk_mtime(0,0), 1316728885)
        self.assertEqual(regionset.get_chunk_mtime(-1,-1), 1316728886)
        self.assertEqual(regionset.get_chunk_mtime(5,0), 1316728905)
        self.assertEqual(regionset.get_chunk_mtime(-22,16), 1316786786)

        
         

if __name__ == "__main__":
    unittest.main()
