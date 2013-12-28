import unittest

from itertools import chain, izip

from overviewer_core.tileset import iterate_base4, RendertileSet
from overviewer_core.util import roundrobin

class RendertileSetTest(unittest.TestCase):
    # If you change this definition, you must also change the hard-coded
    # results list in test_posttraverse()
    tile_paths = frozenset([
            # Entire subtree 0/0 is in the set, nothing else under 0
            (0,0,0),
            (0,0,1),
            (0,0,2),
            (0,0,3),
            # A few tiles under quadrant 1
            (1,0,3),
            (1,1,3),
            (1,2,0),
            # Entire subtree under quadrant 2 is in the set
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
    # The paths as yielded by posttraversal, in an expanding-from-the-center
    # order.
    tile_paths_posttraversal_lists = [
        [
            (0,0,3),
            (0,0,1),
            (0,0,2),
            (0,0,0),
            (0,0),
            (0,),
        ],
        [
            (1,2,0),
            (1,2),

            (1,0,3),
            (1,0),

            (1,1,3),
            (1,1),
            (1,),
        ],
        [
            (2,1,1),
            (2,1,0),
            (2,1,3),
            (2,1,2),
            (2,1),

            (2,0,1),
            (2,0,3),
            (2,0,0),
            (2,0,2),
            (2,0),

            (2,3,1),
            (2,3,0),
            (2,3,3),
            (2,3,2),
            (2,3),

            (2,2,1),
            (2,2,0),
            (2,2,3),
            (2,2,2),
            (2,2),
            (2,),
        ],
    ]
    # Non-round robin post-traversal: finish the first top-level quadrant
    # before moving to the second etc.
    tile_paths_posttraversal       = list(chain(*tile_paths_posttraversal_lists))     + [()]
    # Round-robin post-traversal: start rendering to all directions from the
    # center.
    tile_paths_posttraversal_robin = list(roundrobin(tile_paths_posttraversal_lists)) + [()]

    def setUp(self):
        self.tree = RendertileSet(3)
        for t in self.tile_paths:
            self.tree.add(t)

    def test_query(self):
        """Make sure the correct tiles in the set"""
        for path in iterate_base4(3):
            if path in self.tile_paths:
                self.assertTrue( self.tree.query_path(path) )
            else:
                self.assertFalse( self.tree.query_path(path) )

    def test_iterate(self):
        """Make sure iterating over the tree returns each tile exactly once"""
        dirty = set(self.tile_paths)
        for p in self.tree:
            # Can't use assertIn, was only added in 2.7
            self.assertTrue(p in dirty)

            # Should not see this one again
            dirty.remove(p)

        # Make sure they were all returned
        self.assertEqual(len(dirty), 0)

    def test_iterate_levelmax(self):
        """Same as test_iterate, but specifies the level explicitly"""
        dirty = set(self.tile_paths)
        for p in self.tree.iterate(3):
            # Can't use assertIn, was only added in 2.7
            self.assertTrue(p in dirty)

            # Should not see this one again
            dirty.remove(p)

        # Make sure they were all returned
        self.assertEqual(len(dirty), 0)

    def test_iterate_fail(self):
        """Meta-test: Make sure test_iterate() would actually fail"""
        # if an extra item were returned"""
        self.tree.add((1,1,1))
        self.assertRaises(AssertionError, self.test_iterate)

        # If something was supposed to be returned but wasn't
        tree = RendertileSet(3)
        c = len(self.tile_paths) // 2
        for t in self.tile_paths:
            tree.add(t)
            c -= 1
            if c <= 0:
                break
        self.tree = tree
        self.assertRaises(AssertionError, self.test_iterate)

    def test_count(self):
        self.assertEquals(self.tree.count(), len(self.tile_paths))

    def test_bool(self):
        "Tests the boolean status of a node"
        self.assertTrue(self.tree)
        t = RendertileSet(3)
        self.assertFalse(t)
        t.add((0,0,0))
        self.assertTrue(t)

    def test_query_level(self):
        "Tests querying at a level other than max"
        # level 2
        l2 = set()
        for p in self.tile_paths:
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
        for p in self.tile_paths:
            l2.add(p[0:2])
        for p in self.tree.iterate(2):
            self.assertTrue(p in l2, "%s was not supposed to be returned!" % (p,))
            l2.remove(p)
        self.assertEqual(len(l2), 0, "Never iterated over these items: %s" % l2)

        # level 1
        l1 = set()
        for p in self.tile_paths:
            l1.add(p[0:1])
        for p in self.tree.iterate(1):
            self.assertTrue(p in l1, "%s was not supposed to be returned!" % (p,))
            l1.remove(p)
        self.assertEqual(len(l1), 0, "Never iterated over these items: %s" % l1)

    def test_posttraverse(self):
        """Test a post-traversal of the tree's dirty tiles"""
        # Expect the results in this proper order.
        iterator = iter(self.tree.posttraversal())
        for expected, actual in izip(self.tile_paths_posttraversal, iterator):
            self.assertEqual(actual, expected)

        self.assertRaises(StopIteration, next, iterator)

    def test_posttraverse_roundrobin(self):
        """Test a round-robin post-traversal of the tree's dirty tiles"""
        # Expect the results in this proper order.
        iterator = iter(self.tree.posttraversal(robin=True))
        for expected, actual in izip(self.tile_paths_posttraversal_robin, iterator):
            self.assertEqual(actual, expected)

        self.assertRaises(StopIteration, next, iterator)

    def test_count_all(self):
        """Tests getting a count of all tiles (render tiles plus upper tiles)

        """
        c = self.tree.count_all()
        self.assertEqual(c, 35)

if __name__ == "__main__":
    unittest.main()
