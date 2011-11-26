import unittest

from overviewer_core.quadtree import DirtyTiles, iterate_base4

class DirtyTilesTest(unittest.TestCase):
    dirty_paths = frozenset([
            # Entire subtree 0/0 is dirty, nothing else under 0
            (0,0,0),
            (0,0,1),
            (0,0,2),
            (0,0,3),
            # A few tiles under quadrant 1
            (1,0,3),
            (1,1,3),
            (1,2,0),
            # Entire subtree under quadrant 2 is dirty
            (2,0,0),
            (2,0,1),
            (2,0,2),
            (2,0,3),
            (2,1,0),
            (2,1,1),
            (2,1,2),
            (2,1,3),
            (2,2,0),
            (2,2,1),
            (2,2,2),
            (2,2,3),
            (2,3,0),
            (2,3,1),
            (2,3,2),
            (2,3,3),
            # Nothing under quadrant 3
            ])

    def setUp(self):
        self.tree = DirtyTiles(3)
        for t in self.dirty_paths:
            self.tree.set_dirty(t)

    def test_query(self):
        """Make sure the correct tiles are marked as dirty"""
        for path in iterate_base4(3):
            if path in self.dirty_paths:
                self.assertTrue( self.tree.query_path(path) )
            else:
                self.assertFalse( self.tree.query_path(path) )

    def test_iterate(self):
        """Make sure iterating over the tree returns each dirty tile exactly once"""
        dirty = set(self.dirty_paths)
        for p in self.tree.iterate_dirty():
            # Can't use assertIn, was only added in 2.7
            self.assertTrue(p in dirty)

            # Should not see this one again
            dirty.remove(p)

        # Make sure they were all returned
        self.assertEqual(len(dirty), 0)

    def test_iterate_levelmax(self):
        """Same as test_iterate, but specifies the level explicitly"""
        dirty = set(self.dirty_paths)
        for p in self.tree.iterate_dirty(3):
            # Can't use assertIn, was only added in 2.7
            self.assertTrue(p in dirty)

            # Should not see this one again
            dirty.remove(p)

        # Make sure they were all returned
        self.assertEqual(len(dirty), 0)

    def test_iterate_fail(self):
        """Meta-test: Make sure test_iterate() would actually fail"""
        # if an extra item were returned"""
        self.tree.set_dirty((1,1,1))
        self.assertRaises(AssertionError, self.test_iterate)

        # If something was supposed to be returned but wasn't
        tree = DirtyTiles(3)
        c = len(self.dirty_paths) // 2
        for t in self.dirty_paths:
            tree.set_dirty(t)
            c -= 1
            if c <= 0:
                break
        self.tree = tree
        self.assertRaises(AssertionError, self.test_iterate)

    def test_count(self):
        self.assertEquals(self.tree.count(), len(self.dirty_paths))

    def test_bool(self):
        "Tests the boolean status of a node"
        self.assertTrue(self.tree)
        t = DirtyTiles(3)
        self.assertFalse(t)
        t.set_dirty((0,0,0))
        self.assertTrue(t)

    def test_query_level(self):
        "Tests querying at a level other than max"
        # level 2
        l2 = set()
        for p in self.dirty_paths:
            l2.add(p[0:2])
        for path in iterate_base4(2):
            if path in l2:
                self.assertTrue( self.tree.query_path(path) )
            else:
                self.assertFalse( self.tree.query_path(path) )

        # level 1:
        self.assertTrue( self.tree.query_path((0,)))
        self.assertTrue( self.tree.query_path((1,)))
        self.assertTrue( self.tree.query_path((2,)))
        self.assertFalse( self.tree.query_path((3,)))

    def test_iterate_level(self):
        """Test iterating at a level other than max"""
        # level 2
        l2 = set()
        for p in self.dirty_paths:
            l2.add(p[0:2])
        for p in self.tree.iterate_dirty(2):
            self.assertTrue(p in l2, "%s was not supposed to be returned!" % (p,))
            l2.remove(p)
        self.assertEqual(len(l2), 0, "Never iterated over these items: %s" % l2)

        # level 1
        l1 = set()
        for p in self.dirty_paths:
            l1.add(p[0:1])
        for p in self.tree.iterate_dirty(1):
            self.assertTrue(p in l1, "%s was not supposed to be returned!" % (p,))
            l1.remove(p)
        self.assertEqual(len(l1), 0, "Never iterated over these items: %s" % l1)

if __name__ == "__main__":
    unittest.main()
