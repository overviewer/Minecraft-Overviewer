"""
Outputs a huge image with all currently-supported block textures.
"""

from overviewer_core import textures
import sys
import Image

if len(sys.argv) != 2:
    print "usage: %s [output.png]" % (sys.argv[0],)
    sys.exit(1)

t = textures.Textures()
t.generate()

blocks = {}

for blockid in xrange(textures.max_blockid):
    for data in xrange(textures.max_data):
        tex = t.blockmap[blockid * textures.max_data + data]
        if tex:
            if not blockid in blocks:
                blocks[blockid] = {}
            blocks[blockid][data] = tex

columns = max(map(len, blocks.values()))
rows = len(blocks)
texsize = t.texture_size

gallery = Image.new("RGBA", (columns * texsize, rows * texsize), t.bgcolor)

row = 0
for blockid, textures in blocks.iteritems():
    column = 0
    for data, tex in textures.iteritems():
        gallery.paste(tex[0], (column * texsize, row * texsize))
        column += 1
    row += 1

gallery.save(sys.argv[1])
