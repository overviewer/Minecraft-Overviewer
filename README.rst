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

* Renders efficiently in parallel, using as many simultaneous processes as you
  want!

* Utilizes 2 levels of caching to speed up subsequent renderings of your world.

* Throw the output directory up on a web server to share your Minecraft world
  with everyone!

Requirements
============
This program requires:

* Python 2.6 or 2.7 <http://python.org/download/>
* PIL (Python Imaging Library) <http://www.pythonware.com/products/pil/>
* Numpy <http://scipy.org/Download>

I develop and test this on Linux, but need help testing it on Windows and Mac.
If something doesn't work, let me know.

Using the Google Map Tile Generator
===================================

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
you will find an index.html file in the same directory that you can use to view
it.

Using more Cores
----------------
Adding the "-p" option will utilize more cores during processing.  This can
speed up rendering quite a bit. The default is set to the same number of cores
in your computer, but you can adjust it.

Example to run 5 worker processes in parallel::

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

* Some kind of graphical interface.
