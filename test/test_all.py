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
from test_cache import TestLRU

# DISABLE THIS BLOCK TO GET LOG OUTPUT FROM TILESET FOR DEBUGGING
if 0:
    import logging
    root = logging.getLogger()
    class NullHandler(logging.Handler):
        def handle(self, record):
            pass
        def emit(self, record):
            pass
        def createLock(self):
            self.lock = None
    root.addHandler(NullHandler())
else:
    import overviewer
    import logging
    overviewer.configure_logger(logging.DEBUG, True)


if __name__ == "__main__":
    unittest.main()
