import math
import os
import os.path
import subprocess
import tempfile
import argparse

from overviewer.oil import Matrix, Image, FORMAT_PNG
from overviewer import assetpack
from overviewer import chunkrenderer

"""This script takes a block ID and a data value and renders a single image
rendering of that block.

This file serves two purposes: to let users view and test their block models,
and to serve as a test of the block rendering routines.

"""

# FIXME this does not work no more man!
def main():
    parser = argparse.ArgumentParser(description="Renders an animated GIF of a single block. Used to test the renderer and block models")
    parser.add_argument("blockid", type=int, help="The block ID to render")
    parser.add_argument("data", type=int, help="The data value of the block to render, if applicable", default=0, nargs='?')
    parser.add_argument("--texturepath", type=str, help="specify a resource pack to use", default=None)

    args = parser.parse_args()
    blockid = args.blockid
    data = args.data
    if args.texturepath:
        other = assetpack.ZipAssetPack(args.texturepath)
    else:
        other = None
    render(blockid, data, otherap=other)

def render(block, out="output.gif"):
    FRAMES = 60
    SIZE = 512
    
    scale = 2 / math.sqrt(3)
    filenames = []
    directory = tempfile.mkdtemp()

    for i, angle in enumerate(range(0, 360, 360//FRAMES)):
        # Standard viewing angle: ~35 degrees, straight down the diagonal of a cube
        matrix = Matrix()
        matrix.scale(scale, scale, scale)
        matrix.rotate(math.atan2(1, math.sqrt(2)), 0, 0)
        matrix.rotate(0, math.radians(angle), 0)
        matrix.rotate(math.radians(45), math.radians(45), 0)
        matrix.translate(-0.5, -0.5, -0.5)
        
        filename = os.path.join(directory, "output_{0:02}.png".format(i))
        filenames.append(filename)
        
        im = Image(SIZE, SIZE)
        chunkrenderer.render_block(block, matrix, im)
        im.save(filename, FORMAT_PNG)

    print("Converting to gif...")
    subprocess.call(["convert",
                     "-delay", "5",
                     "-dispose", "Background",
                     ] + filenames + [
                     out,
                     ])

    print("Cleaning up frame images from {0}".format(directory))
    for filename in filenames:
        os.unlink(filename)

    os.rmdir(directory)

if __name__ == "__main__":
    main()
