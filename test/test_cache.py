import unittest

from overviewer_core import cache

class TestLRU(unittest.TestCase):

    def setUp(self):
        self.lru = cache.LRUCache(size=5)

    def test_single_insert(self):
        self.lru[1] = 2
        self.assertEquals(self.lru[1], 2)

    def test_multiple_insert(self):
        self.lru[1] = 2
        self.lru[3] = 4
        self.lru[5] = 6
        self.assertEquals(self.lru[1], 2)
        self.assertEquals(self.lru[3], 4)
        self.assertEquals(self.lru[5], 6)

    def test_full(self):
        self.lru[1] = 'asdf'
        self.lru[2] = 'asdf'
        self.lru[3] = 'asdf'
        self.lru[4] = 'asdf'
        self.lru[5] = 'asdf'
        self.lru[6] = 'asdf'
        self.assertRaises(KeyError, self.lru.__getitem__, 1)
        self.assertEquals(self.lru[2], 'asdf')
        self.assertEquals(self.lru[3], 'asdf')
        self.assertEquals(self.lru[4], 'asdf')
        self.assertEquals(self.lru[5], 'asdf')
        self.assertEquals(self.lru[6], 'asdf')

    def test_lru(self):
        self.lru[1] = 'asdf'
        self.lru[2] = 'asdf'
        self.lru[3] = 'asdf'
        self.lru[4] = 'asdf'
        self.lru[5] = 'asdf'

        self.assertEquals(self.lru[1], 'asdf')
        self.assertEquals(self.lru[2], 'asdf')
        self.assertEquals(self.lru[4], 'asdf')
        self.assertEquals(self.lru[5], 'asdf')

        # 3 should be evicted now
        self.lru[6] = 'asdf'

        self.assertRaises(KeyError, self.lru.__getitem__, 3)
        self.assertEquals(self.lru[1], 'asdf')
        self.assertEquals(self.lru[2], 'asdf')
        self.assertEquals(self.lru[4], 'asdf')
        self.assertEquals(self.lru[5], 'asdf')
        self.assertEquals(self.lru[6], 'asdf')
