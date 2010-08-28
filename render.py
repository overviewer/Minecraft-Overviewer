#!/usr/bin/python

import os
import sys
import os.path
from optparse import OptionParser
import re

import world

helptext = """
%prog [-c] <Path to World> <image out.png>
%prog -d [-c] <Path to World>"""

def remove_images(worlddir, cavemode):
    if cavemode:
        cavestr = "cave"
    else:
        cavestr = "nocave"
    imgre = r"img\.[^.]+\.[^.]+\.{0}\.\w+\.png$".format(cavestr)
    matcher = re.compile(imgre)

    for dirpath, dirnames, filenames in os.walk(worlddir):
        for f in filenames:
            if matcher.match(f):
                filepath = os.path.join(dirpath, f)
                print "Deleting {0}".format(filepath)
                os.unlink(filepath)

def confirm(imgfile):
    answer = raw_input("Overwrite existing image at %r? [Y/n]" % imgfile).strip()
    if not answer or answer.lower().startswith("y"):
        return True
    return False

def main():
    parser = OptionParser(usage=helptext)
    parser.add_option("-c", "--caves", dest="caves", help="Render only caves", action="store_true")
    parser.add_option("-d", "--delete-cache", dest="delete", help="Deletes the image files cached in your world directory", action="store_true")
    parser.add_option("-p", "--processes", dest="procs", help="How many chunks to render in parallel. A good number for this is 1 more than the number of cores in your computer. Default 2", default=2, action="store", type="int")

    options, args = parser.parse_args()

    if len(args) < 1:
        print "You need to give me your world directory"
        parser.print_help()
        sys.exit(1)
    worlddir = args[0]

    if options.delete:
        remove_images(worlddir, options.caves)
    else:
        if len(args) != 2:
            parser.error("What do you want to save the image as?")
        imageout = args[1]
        if not imageout.endswith(".png"):
            imageout = imageout + ".png"
        if os.path.exists(imageout) and not confirm(imageout):
            return
        imageobj = world.render_world(worlddir, options.caves, options.procs)
        print "Saving image..."
        imageobj.save(imageout)
        print "Saved as", imageout

if __name__ == "__main__":
    main()
