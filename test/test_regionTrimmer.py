import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import networkx

import contrib.regionTrimmer as region_trimmer

class TestRegionTrimmer(unittest.TestCase):
    def test_get_nodes(self):
        coords = [(0, 0), (0, -1), (-1, 0), (-1, -1)]
        with TemporaryDirectory() as tmpdirname:
            region_file = Path(tmpdirname)
            for x, z in coords:
                region_fname = "r.{x}.{z}.mca".format(x=x, z=z)
                (region_file / region_fname).touch()

            nodes = region_trimmer.get_nodes(region_file)
            self.assertListEqual(sorted(nodes), sorted(coords))

    def test_get_nodes_returns_empty_list_when_no_region_files(self):
        with TemporaryDirectory() as tmpdirname:
            region_file = Path(tmpdirname)
            (region_file / "not_region_file.txt").touch()
            nodes = region_trimmer.get_nodes(region_file)
            self.assertListEqual(nodes, [])

    def test_get_region_file_from_node(self):
        node = (0, 0)
        regionset_path = Path('/path/to/regions')

        self.assertEqual(region_trimmer.get_region_file_from_node(
            regionset_path, node), Path('/path/to/regions/r.0.0.mca'))

    def test_get_graph_bounds(self):
        """ Should return (max_x, min_x, max_z, min_z) of all nodes
        """
        graph = networkx.Graph()
        graph.add_nodes_from([(0, 0), (0, -1), (-1, 0), (-1, -1)])

        self.assertEqual(region_trimmer.get_graph_bounds(graph), (0, -1, 0, -1))

    def test_get_graph_center_by_bounds(self):
        self.assertEqual(region_trimmer.get_graph_center_by_bounds((0, -1, 0, -1)), (-1, -1))

    def test_generate_edges(self):
        graph = networkx.Graph()
        graph.add_nodes_from(
            [(0, 0), (0, -1), (-1, 0), (-1, -1)]
        )
        graph = region_trimmer.generate_edges(graph)
        self.assertEqual(
            graph.adj,
            {
                (0, -1): {(0, 0): {}, (-1, -1): {}},
                (0, 0): {
                    (0, -1): {},
                    (-1, 0): {},
                    (-1, -1): {},
                },
                (-1, 0): {(0, 0): {}, (-1, -1): {}},
                (-1, -1): {
                    (0, -1): {},
                    (0, 0): {},
                    (-1, 0): {},
                },
            },
        )
