#!/usr/bin/env python3
from test_contributors import TestContributors
from test_cache import TestLRU
from test_tileset import TilesetTest
from test_settings import SettingsTest
from test_rendertileset import RendertileSetTest
from test_tileobj import TileTest
import unittest

# For convenience
import sys
import os
import logging
sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), os.pardir))

Import unit test cases or suites here
DISABLE THIS BLOCK TO GET LOG OUTPUT FROM TILESET FOR DEBUGGING
if 0:
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
    from overviewer_core import logger
    logger.configure(logging.DEBUG, True)


if __name__ == "__main__":
    unittest.main()
