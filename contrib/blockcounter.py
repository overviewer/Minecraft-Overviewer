"""Produces block counts

"""

import world, chunk

import sys
from numpy import *

def block_breakdown(worlddir):
    """Analyzes the given world dir and tallys up total block counts for each
    type. Returns two arrays.
    The first is a mapping from (blocktype, z-height) to count
    The second is a mapping from blocktype to total count
    """
    # Maps (blocktype, z-height) to count
    heights = zeros((256,128), dtype=int)

    # Maps (blocktype) to total
    totals = zeros((256,), dtype=int)

    all_chunks = world.find_chunkfiles(worlddir)
    for i, (chunkx, chunky, chunkfile) in enumerate(all_chunks):
        print "{0} / {1}".format(i, len(all_chunks))
        sys.stdout.flush()
        blockarr = chunk.get_blockarray_fromfile(chunkfile)

        for coord, blocktype in ndenumerate(blockarr):
            totals[blocktype] += 1
            heights[blocktype, coord[2]] += 1

    return heights, totals

# Some data from my world on iron
iron = array([   0,  329, 1978, 4454, 6068, 7057, 7116, 7070, 7232, 7441, 7198,
       7206, 7163, 6846, 6965, 7145, 7251, 7136, 6878, 7142, 7421, 7206,
       7163, 7264, 7311, 7355, 7145, 7117, 7181, 7424, 7304, 7560, 7591,
       7321, 7528, 7487, 7355, 7198, 7334, 7566, 7518, 7146, 7510, 7577,
       7532, 7681, 7612, 7376, 7319, 7216, 7195, 6863, 6399, 6198, 5983,
       5599, 5320, 4861, 4604, 4250, 3531, 3129, 3045, 2782, 2433, 1966,
        891,  117,    0,    0,    0,    0,    0,    0,    0,    0,    0,
          0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
          0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
          0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
          0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
          0,    0,    0,    0,    0,    0,    0])
totaliron = 416159
