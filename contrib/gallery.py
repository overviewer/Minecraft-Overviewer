#!/usr/bin/env python3
"""
Outputs a huge image with all currently-supported block textures.
"""

import argparse

from PIL import Image
import sys
import os

# incantation to be able to import overviewer_core
if not hasattr(sys, "frozen"):
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.split(__file__)[0], '..')))


def main(outfile):
    from overviewer_core import textures
    t = textures.Textures()
    t.generate()

    blocks = {}

    for blockid in range(textures.max_blockid):
        for data in range(textures.max_data):
            tex = t.blockmap[blockid * textures.max_data + data]
            if tex:
                if blockid not in blocks:
                    blocks[blockid] = {}
                blocks[blockid][data] = tex

    columns = max(len(v) for v in blocks.values())
    rows = len(blocks)
    texsize = t.texture_size

    gallery = Image.new("RGBA", (columns * texsize, rows * texsize), t.bgcolor)

    for row, (blockid, textures) in enumerate(blocks.items()):
        for column, (data, tex) in enumerate(textures.items()):
            gallery.paste(tex[0], (column * texsize, row * texsize))

    gallery.save(outfile)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('file', metavar='output.png')
    args = parser.parse_args()
    main(args.file)
