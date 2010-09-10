====================
Minecraft Overviewer
====================
By Andrew Brown

http://github.com/brownan/Minecraft-Overviewer

Generates large resolution images of a Minecraft map.

In short, this program reads in Minecraft world files and renders very large
resolution images. It performs a similar function to the existing Minecraft
Cartographer program but with a slightly different goal in mind: to generate
large resolution images such that one can zoom in and see details.

Features
========

* Renders large resolution images of your world, such that you can zoom in and
  see details

* Outputs a Google Map powered interface that is memory efficient, both in
  generating and viewing.

* Utilizes 2 levels of caching to speed up subsequent renderings of your world.

* Throw the output directory up on a web server to share your Minecraft world
  with everyone!

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
least half an hour, and for really large maps, several hours (Subsequent runs
will be quicker since it only re-renders tiles that have changed). Second,
there's no progress bar. You can watch the tiles get generated, but the program
gives no direct feedback at this time on how far along it is.

There are probably some other minor glitches along the way, hopefully they will
be fixed soon. See the `Bugs`_ section below.

Running
-------
To generate a set of Google Map tiles, use the gmap.py script like this::

    python gmap.py <Path to World> <Output Directory>

The output directory will be created if it doesn't exist. This will generate a
set of image tiles for your world in the directory you choose. When it's done,
it will put an index.html file in the same directory that you can use to view
it.

Note that this program renders each chunk of your world as an intermediate step
and stores the images in your world directory as a cache. You usually don't
need to worry about this, but if you want to delete them, see the section below
about `Deleting the Cache`_.

Also note that this program outputs hash files alongside the tile images in the
output directory. These files are used to quickly determine if a tile needs to
be re-generated on subsequent runs of the program on the same world. This
greatly speeds up the rendering.

Using more Cores
----------------
Adding the "-p" option will utilize more cores to generate the chunk files.
This can speed up rendering quite a bit. However, the tile generation routine
is currently serial and not written to take advantage of multiple cores. This
option will only affect the chunk generation (which is around half the process)

Example::

    python gmap.py -p 5 <Path to World> <Output Directory>

Crushing the Output Tiles
-------------------------
Image files taking too much disk space? Try using pngcrush. On Linux and
probably Mac, if you have pngcrush installed, this command will go and crush
all your images in the given destination. This took the total disk usage of the
render for my world from 85M to 67M.

::

    find /path/to/destination -name "*.png" -exec pngcrush {} {}.crush \; -exec mv {}.crush {} \;

If you're on Windows, I've gotten word that this command line snippet works
provided pngout is installed and on your path. Note that the % symbols will
need to be doubled up if this is in a batch file.

::

    FOR /R c:\path\to\tiles\folder %v IN (*.png) DO pngout %v /y

Viewing the Results
-------------------
The output is two things: an index.html file, and a directory hierarchy full of
images. To view your world, simply open index.html in a web browser. Internet
access is required to load the Google Maps API files, but you otherwise don't
need anything else.

You can throw these files up to a web server to let others view your map. You
do not need a Google Maps API key (as was the case with older versions of the
API), so just copying the directory to your web server should suffice.

Tip: Since Minecraft worlds rarely produce perfectly square worlds, there will
be blank and non-existent tiles around the borders of your world. The Google
Maps API has no way of knowing this until it requests them and the web server
returns a 404 Not Found. If this doesn't bother you, then fine, stop reading.
Otherwise: you can avoid a lot of 404s to your logs by configuring your web
server to redirect all 404 requests in that directory to a single 1px
"blank.png". This may or may not save on bandwidth, but it will probably save
on log noise.

Using the Large Image Renderer
==============================
The Large Image Renderer creates one large image of your world. This was
originally the only option, but uses a large amount of memory and generates
unwieldy large images. It is still included in this package in case someone
finds it useful, but the preferred method is the Google Map tile generator.

Be warned: For even moderately large worlds this may eat up all your memory,
take a long time, or even just outright crash. It allocates an image large
enough to accommodate your entire world and then draws each block on it. It
would not be surprising to need gigabytes of memory for extremely large
worlds.

To render a world, run the renderer.py script like this::

    python renderer.py <Path to World> <image out.png>

The <Path to world> is the path to the directory containing your world files. 

Cave mode
---------
Cave mode renders all blocks that have no sunlight hitting them. Additionally,
blocks are given a colored tint according to how deep they are. Red are closest
to bedrock, green is close to sea level, and blue is close to the sky.

Cave mode is like normal mode, but give it the "-c" flag. Like this::

    python renderer.py -c <Path to World> <image out.png>

Deleting the Cache
------------------
The Overviewer keeps a cache of each world chunk it renders stored within your
world directory. When you generate a new image of the same world, it will only
re-render chunks that have changed, speeding things up a lot.

If you want to delete these images, run the renderer.py script with the -d flag::

    python renderer.py -d <Path to World>

To delete the cave mode images, run it with -d and -c

::

    python renderer.py -d -c <Path to World>

You may want to do this for example to save space. Or perhaps you've changed
texture packs and want to force it to re-render all chunks.

Using More Cores
----------------
The Overviewer will render each chunk separately in parallel. You can tell it
how many processes to start with the -p option. This is set to a default of 2,
which will use 2 processes to render chunks, and 1 to render the final image.

To bump that up to 3 processes, use a command in this form::

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

* Speed up the tile rendering. I can parallelize that process.

* I want to add some indication of progress to the tile generation.

* Some kind of graphical interface.
