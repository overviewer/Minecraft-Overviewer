#!/usr/bin/env python
import unittest

# For convenience
import sys,os
sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), os.pardir))

# Import unit test cases or suites here
from test_tileobj import TileTest
from test_rendertileset import RendertileSetTest
from test_settings import SettingsTest
from test_tileset import TilesetTest

if __name__ == "__main__":
    unittest.main()
