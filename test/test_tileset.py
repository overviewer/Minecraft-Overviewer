import unittest
import tempfile
import shutil
from collections import defaultdict
import os
import os.path
import random

from overviewer_core import tileset

# Supporing data
# chunks list: chunkx, chunkz mapping to chunkmtime
# In comments: col, row
chunks = {
        (0, 0): 5, # 0, 0
        (0, 1): 5, # 1, 1
        (0, 2): 5, # 2, 2
        (0, 3): 5, # 3, 3
        (0, 4): 5, # 4, 4
        (1, 0): 5, # 1, -1
        (1, 1): 5, # 2, 0
        (1, 2): 5, # 3, 1
        (1, 3): 5, # 4, 2
        (1, 4): 5, # 5, 3
        (2, 0): 5, # 2, -2
        (2, 1): 5, # 3, -1
        (2, 2): 5, # 4, 0
        (2, 3): 5, # 5, 1
        (2, 4): 5, # 6, 2
        (3, 0): 5, # 3, -3
        (3, 1): 5, # 4, -2
        (3, 2): 5, # 5, -1
        (3, 3): 5, # 6, 0
        (3, 4): 5, # 7, 1
        (4, 0): 5, # 4, -4
        (4, 1): 5, # 5, -3
        (4, 2): 5, # 6, -2
        (4, 3): 5, # 7, -1
        (4, 4): 5, # 8, 0
        }

# Supporting resources
######################

class FakeRegionset(object):
    def __init__(self, chunks):
        self.chunks = dict(chunks)

    def get_chunk(self, x,z):
        return NotImplementedError()

    def iterate_chunks(self):
        for (x,z),mtime in self.chunks.iteritems():
            yield x,z,mtime

    def iterate_newer_chunks(self, filemtime):
        for (x,z),mtime in self.chunks.iteritems():
            yield x,z,mtime

    def get_chunk_mtime(self, x, z):
        try:
            return self.chunks[x,z]
        except KeyError:
            return None

class FakeAssetmanager(object):
    def __init__(self, lastrendertime):
        self.lrm = lastrendertime

    def get_tileset_config(self, _):
        return {'lastrendertime': self.lrm}

def get_tile_set(chunks):
    """Given the dictionary mapping chunk coordinates their mtimes, returns a
    dict mapping the tiles that are to be rendered to their mtimes that are
    expected. Useful for passing into the create_fakedir() function. Used by
    the compare_iterate_to_expected() method.
    """
    tile_set = defaultdict(int)
    for (chunkx, chunkz), chunkmtime in chunks.iteritems():

        col, row = tileset.convert_coords(chunkx, chunkz)

        for tilec, tiler in tileset.get_tiles_by_chunk(col, row):
            tile = tileset.RenderTile.compute_path(tilec, tiler, 5)
            tile_set[tile.path] = max(tile_set[tile.path], chunkmtime)

    # At this point, tile_set holds all the render-tiles
    for tile, tile_mtime in tile_set.copy().iteritems():
        # All render-tiles are length 5. Hard-code its upper tiles
        for i in reversed(xrange(5)):
            tile_set[tile[:i]] = max(tile_set[tile[:i]], tile_mtime)
    return dict(tile_set)

def create_fakedir(outputdir, tiles):
    """Takes a base output directory and a tiles dict mapping tile paths to
    tile mtimes as returned by get_tile_set(), creates the "tiles" (empty
    files) and sets mtimes appropriately

    """
    for tilepath, tilemtime in tiles.iteritems():
        dirpath = os.path.join(outputdir, *(str(x) for x in tilepath[:-1]))
        if len(tilepath) == 0:
            imgname = "base.png"
        else:
            imgname = str(tilepath[-1]) + ".png"

        if not os.path.exists(dirpath):
            os.makedirs(dirpath)
        finalpath = os.path.join(dirpath, imgname)
        open(finalpath, 'w').close()
        os.utime(finalpath, (tilemtime, tilemtime))

# The test cases
################
class TilesetTest(unittest.TestCase):
    def setUp(self):
        # Set up the region set
        self.rs = FakeRegionset(chunks)

        self.tempdirs = []

        # Consistent random numbers
        self.r = random.Random(1)

    def tearDown(self):
        for d in self.tempdirs:
            shutil.rmtree(d)

    def get_outputdir(self):
        d = tempfile.mkdtemp(prefix="OVTEST")
        self.tempdirs.append(d)
        return d

    def get_tileset(self, options, outputdir, preprocess=None):
        """Returns a newly created TileSet object and return it.
        A set of default options are provided. Any options passed in will
        override the defaults. The output directory is passed in and it is
        recommended to use a directory from self.get_outputdir()

        preprocess, if given, is a function that takes the tileset object. It
        is called before do_preprocessing()
        """
        defoptions = {
                'name': 'world name',
                'bgcolor': '#000000',
                'imgformat': 'png',
                'optimizeimg': 0,
                'rendermode': 'normal',
                'rerenderprob': 0
                }
        defoptions.update(options)
        ts = tileset.TileSet(None, self.rs, FakeAssetmanager(0), None, defoptions, outputdir)
        if preprocess:
            preprocess(ts)
        ts.do_preprocessing()
        return ts

    def compare_iterate_to_expected(self, ts, chunks):
        """Runs iterate_work_items on the tileset object and compares its
        output to what we'd expect if it was run with the given chunks

        chunks is a dictionary whose keys are chunkx,chunkz. This method
        calculates from that set of chunks the tiles they touch and their
        parent tiles, and compares that to the output of ts.iterate_work_items().

        """
        paths = set(x[0] for x in ts.iterate_work_items(0))

        # Get what tiles we expect to be returned
        expected = get_tile_set(chunks)

        # Check that all paths returned are in the expected list
        for tilepath in paths:
            self.assertTrue(tilepath in expected, "%s was not expected to be returned. Expected %s" % (tilepath, expected))

        # Now check that all expected tiles were indeed returned
        for tilepath in expected.iterkeys():
            self.assertTrue(tilepath in paths, "%s was expected to be returned but wasn't: %s" % (tilepath, paths))

    def test_get_phase_length(self):
        ts = self.get_tileset({'renderchecks': 2}, self.get_outputdir())
        self.assertEqual(ts.get_num_phases(), 1)
        self.assertEqual(ts.get_phase_length(0), len(get_tile_set(chunks)))

    def test_forcerender_iterate(self):
        """Tests that a rendercheck mode 2 iteration returns every render-tile
        and upper-tile
        """
        ts = self.get_tileset({'renderchecks': 2}, self.get_outputdir())
        self.compare_iterate_to_expected(ts, self.rs.chunks)


    def test_update_chunk(self):
        """Tests that an update in one chunk properly updates just the
        necessary tiles for rendercheck mode 0, normal operation. This
        shouldn't touch the filesystem at all.

        """

        # Update one chunk with a newer mtime
        updated_chunks = {
                (0,0): 6
                }
        self.rs.chunks.update(updated_chunks)

        # Create the tileset and set its last render time to 5
        ts = self.get_tileset({'renderchecks': 0}, self.get_outputdir(),
                lambda ts: setattr(ts, 'last_rendertime', 5))

        # Now see if the return is what we expect
        self.compare_iterate_to_expected(ts, updated_chunks)

    def test_update_chunk2(self):
        """Same as above but with a different set of chunks
        """
        # Pick 3 random chunks to update
        chunks = self.rs.chunks.keys()
        self.r.shuffle(chunks)
        updated_chunks = {}
        for key in chunks[:3]:
            updated_chunks[key] = 6
        self.rs.chunks.update(updated_chunks)
        ts = self.get_tileset({'renderchecks': 0}, self.get_outputdir(),
                lambda ts: setattr(ts, 'last_rendertime', 5))
        self.compare_iterate_to_expected(ts, updated_chunks)

    def test_rendercheckmode_1(self):
        """Tests that an interrupted render will correctly pick up tiles that
        need rendering

        """
        # For this we actually need to set the tile mtimes on disk and have the
        # TileSet object figure out from that what it needs to render.
        # Strategy: set some tiles on disk to mtime 3, and TileSet needs to
        # find them and update them to mtime 5 as reported by the RegionSet
        # object.
        # Chosen at random:
        outdated_tiles = [
                (0,3,3,3,3),
                (1,2,2,2,1),
                (2,1,1),
                (3,)
                ]
        # These are the tiles that we also expect it to return, even though
        # they were not outdated, since they depend on the outdated tiles
        additional = [
                (0,3,3,3),
                (0,3,3),
                (0,3),
                (0,),
                (1,2,2,2),
                (1,2,2),
                (1,2),
                (1,),
                (2,1),
                (2,),
                (),
                ]

        outputdir = self.get_outputdir()
        # Fill the output dir with tiles
        all_tiles = get_tile_set(self.rs.chunks)
        all_tiles.update(dict((x,3) for x in outdated_tiles))
        create_fakedir(outputdir, all_tiles)

        # Create the tileset and do the scan
        ts = self.get_tileset({'renderchecks': 1}, outputdir)

        # Now see if it's right
        paths = set(x[0] for x in ts.iterate_work_items(0))
        expected = set(outdated_tiles) | set(additional)
        for tilepath in paths:
            self.assertTrue(tilepath in expected, "%s was not expected to be returned. Expected %s" % (tilepath, expected))

        for tilepath in expected:
            self.assertTrue(tilepath in paths, "%s was expected to be returned but wasn't: %s" % (tilepath, paths))
