"""Simple Benchmarking script.

Usage and example:

$ python contrib/benchmark.py World4/
Rendering 50 chunks...
Took 20.290062 seconds or 0.405801 seconds per chunk, or 2.464261 chunks per second
"""


import chunk
import world
import tempfile
import glob
import time
import cProfile
import os
import sys
import shutil


# create a new, empty, cache dir
cachedir = tempfile.mkdtemp(prefix="benchmark_cache", dir=".")
if os.path.exists("benchmark.prof"): os.unlink("benchmark.prof")

w = world.WorldRenderer("World4", cachedir)

numchunks = 50
chunklist = w._find_chunkfiles()[:numchunks]

print "Rendering %d chunks..." % (numchunks)
def go():
    for f in chunklist:
        chunk.render_and_save(f[2], w.cachedir, w, (None,None), None)
start = time.time()
if "-profile" in sys.argv:
    cProfile.run("go()", 'benchmark.prof')
else:
    go()
stop = time.time()

delta = stop - start

print "Took %f seconds or %f seconds per chunk, or %f chunks per second" % (delta, delta/numchunks, numchunks/delta)

if "-profile" in sys.argv:
    print "Profile is below:\n----\n"

    import pstats
    p = pstats.Stats('benchmark.prof')

    p.strip_dirs().sort_stats("cumulative").print_stats(20)


shutil.rmtree(cachedir)
