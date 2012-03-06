import unittest

from overviewer_core.tileset import iterate_base4, RenderTile

items = [
        ((-4,-8), (0,0)),
        ((-2,-8), (0,1)),
        ((0,-8), (1,0)),
        ((2,-8), (1,1)),
        ((-4,-4), (0,2)),
        ((-2,-4), (0,3)),
        ((0,-4), (1,2)),
        ((2,-4), (1,3)),
        ((-4,0), (2,0)),
        ((-2,0), (2,1)),
        ((0,0), (3,0)),
        ((2,0), (3,1)),
        ((-4,4), (2,2)),
        ((-2,4), (2,3)),
        ((0,4), (3,2)),
        ((2,4), (3,3)),
        ]

class TileTest(unittest.TestCase):
    def test_compute_path(self):
        """Tests that the correct path is computed when a col,row,depth is
        given to compute_path

        """
        for path in iterate_base4(7):
            t1 = RenderTile.from_path(path)
            col = t1.col
            row = t1.row
            depth = len(path)

            t2 = RenderTile.compute_path(col, row, depth)
            self.assertEqual(t1, t2)

    def test_equality(self):
        t1 = RenderTile(-6, -20, (0,1,2,3))
        
        self.assertEqual(t1, RenderTile(-6, -20, (0,1,2,3)))
        self.assertNotEqual(t1, RenderTile(-4, -20, (0,1,2,3)))
        self.assertNotEqual(t1, RenderTile(-6, -24, (0,1,2,3)))
        self.assertNotEqual(t1, RenderTile(-6, -20, (0,1,2,0)))

    def test_depth2_from_path(self):
        """Test frompath on all 16 tiles of a depth 2 tree"""
        for (col, row), path in items:
            t = RenderTile.from_path(path)
            self.assertEqual(t.col, col)
            self.assertEqual(t.row, row)

    def test_depth2_compute_path(self):
        """Test comptue_path on all 16 tiles of a depth 2 tree"""
        for (col, row), path in items:
            t = RenderTile.compute_path(col, row, 2)
            self.assertEqual(t.path, path)


if __name__ == "__main__":
    unittest.main()
