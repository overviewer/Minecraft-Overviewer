====================
Minecraft Overviewer
====================
By Andrew Brown

http://github.com/brownan/Minecraft-Overviewer

Generates large resolution images of a Minecraft map.

In short, this program reads in Minecraft world files and renders very large
resolution images. It performs a similar function to the existing Minecraft
Cartographer program.

I wrote this with an additional goal in mind: to generate large images that I
could zoom in and see details.

**New**: gmap.py generates tiles for a Google Map interface, so that people
with large worlds can still benefit!

Requirements
============
This program requires:

* Python 2.6 or 2.7 <http://python.org/download/>
* PIL (Python Imaging Library) <http://www.pythonware.com/products/pil/>
* Numpy <http://scipy.org/Download>

I developed and tested this on Linux. It has been reported to work on Windows
and Mac, but if something doesn't, let me know.

Using the Google Map Tile Generator
===================================
This is the new and preferred way to generate images of your map.

Disclaimers
-----------
Before you dive into using this, let it be known that there are a few minor
problems. First, it's slow. If your map is really large, this could take at
least half an hour, and for really large maps, several hours. Second, there's
no progress bar. You can watch the tiles get generated, but the program gives
no feedback at this time on how far it is.

There are probably some other minor glitches along the way, hopefully they will
be fixed soon. See the `Bugs`_ section below.

Running
-------
To generate a set of Google Map tiles, use the gmap.py script like this:

    python gmap.py <Path to World> <Output Directory>

The output directory must already exist. This will generate a set of image
tiles for your world. When it's done, it will put an index.html file in the
same directory that you can use to view it.

Note that this program renders each chunk of your world as an intermediate step
and stores the images in your world directory as a cache. You usually don't
need to worry about this, but if you want to delete them, see the section below
about `Deleting the Cache`_.

Using the Large Image Renderer
==============================
The Large Image Renderer creates one large image of your world. This was
originally the only option, but would crash and use too much memory for very
large worlds. You may still find a use for it though.

Right now there's only a console interface. Here's how to use it:

To render a world, run the renderer.py script like this:

    python renderer.py <Path to World> <image out.png>

The <Path to world> is the path to the directory containing your world files. 

Cave mode
---------
Cave mode renders all blocks that have no sunlight hitting them. Additionally,
blocks are given a colored tint according to how deep they are. Red are closest
to bedrock, green is close to sea level, and blue is close to the sky.

Cave mode is like normal mode, but give it the "-c" flag. Like this:

    python renderer.py -c <Path to World> <image out.png>

Deleting the Cache
------------------
The Overviewer keeps a cache of each world chunk it renders stored within your
world directory. When you generate a new image of the same world, it will only
re-render chunks that have changed, speeding things up a lot.

If you want to delete these images, run the renderer.py script with the -d flag:

    python renderer.py -d <Path to World>

To delete the cave mode images, run it with -d and -c

    python renderer.py -d -c <Path to World>

You may want to do this for example to save space. Or perhaps you've changed
texture packs and want to force it to re-render all chunks.

Using More Cores
----------------
The Overviewer will render each chunk separately in parallel. You can tell it
how many processes to start with the -p option. This is set to a default of 2,
which will use 2 processes to render chunks, and 1 to render the final image.

To bump that up to 3 processes, use a command in this form:

    python renderer.py -p 3 <Path to World> <image out.png>

Bugs
====
This program has bugs. They are mostly minor things, I wouldn't have released a
completely useless program. However, there are a number of things that I want
to fix or improve.

For a current list of issues, visit
http://github.com/brownan/Minecraft-Overviewer/issues

Feel free to comment on issues, report new issues, and vote on issues that are
important to you, so I can prioritize accordingly.

An incomplete list of things I want to fix soon is:

* Rendering non-cube blocks, such as torches, flowers, mine tracks, fences,
  doors, and the like. Right now they are either not rendered at all, or
  rendered as if they were a cube, so it looks funny.

* Water transparency. There are a couple issues involved with that, and I want
  to fix them.

* Add lighting

* Speed up the tile rendering. I can parallelize that process, and add more
  caches to the tiles so subsequent renderings go faster.

* I want to add some indication of progress to the tile generation.
