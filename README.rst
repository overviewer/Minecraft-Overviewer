====================
Minecraft Overviewer
====================
By Andrew Brown and contributors

http://github.com/brownan/Minecraft-Overviewer

Generates large resolution images of a Minecraft map.

In short, this program reads in Minecraft world files and renders very large
resolution images. It performs a similar function to the existing Minecraft
Cartographer program but with a slightly different goal in mind: to generate
large resolution images such that one can zoom in and see details.

See some examples here!
http://github.com/brownan/Minecraft-Overviewer/wiki/Map-examples

(To contact me, send me a message on Github)

Features
========

* Renders large resolution images of your world, such that you can zoom in and
  see details

* Customizable textures! Pulls textures straight from your installed texture
  pack!

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
* Either the Minecraft client installed, or a terrain.png file. See the
  `Textures`_ section below.

I develop and test this on Linux, but need help testing it on Windows and Mac.
If something doesn't work, let me know.

Using the Overviewer
====================

Disclaimers
-----------
Before you dive into using this, just be aware that, for large maps, there is a
*lot* of data to parse through and process. If your world is very large, expect
the initial render to take at least an hour, possibly more. (Since Minecraft
maps are practically infinite, the maximum time this could take is also
infinite!)

If you press ctrl-C, it will stop. The next run will pick up where it left off.

Once your initial render is done, subsequent renderings will be MUCH faster due
to all the caching that happens behind the scenes. Just use the same output
directory and it will only update the tiles it needs to.

There are probably some other minor glitches along the way, hopefully they will
be fixed soon. See the `Bugs`_ section below.

Textures
--------
The Overviewer uses actual textures to render your world. However, I don't
include textures in the package. You will need to do one of two things before
you can use the Overviewer:

* Make sure the Minecraft client is installed. The Overviewer will find the
  installed minecraft.jar and extract the textures from it.

* Install a texture file yourself. This file is called "terrain.png" and is
  normally found in your minecraft.jar file (not "Minecraft.jar", the launcher,
  but rather the file that's downloaded by the launcher and installed into a
  hidden directory). You can also get this file from any of the third party
  texture packs out there.

Running
-------
To generate a set of Google Map tiles, use the gmap.py script like this::

    python gmap.py [OPTIONS] <World Number / Path to World> <Output Directory>

The output directory will be created if it doesn't exist. This will generate a
set of image tiles for your world in the directory you choose. When it's done,
you will find an index.html file in the same directory that you can use to view
it.

**Important note about Caches**

The Overviewer will put a cached image for every chunk *directly in your world
directory by default*. If you do not like this behavior, you can specify
another location with the --cachedir option. See below for details.

Options
-------

-h, --help
    Shows the list of options and exits

--cachedir=CACHEDIR
    By default, the Overviewer will save in your world directory one image
    file for every chunk in your world. If you do backups of your world,
    you may not want these images in your world directory.

    Use this option to specify an alternate location to put the rendered
    chunk images. You must specify this same directory each rendering so
    that it doesn't have to render every chunk from scratch every time.

    Example::

        python gmap.py --cachedir=<chunk cache dir> <world> <output dir>

--imgformat=FORMAT
    Set the output image format used for the tiles. The default is 'png',
    but 'jpg' is also supported. Note that regardless of what you choose,
    Overviewer will still use PNG for cached images to avoid recompression
    artifacts.

-p PROCS, --processes=PROCS
    Adding the "-p" option will utilize more cores during processing.  This
    can speed up rendering quite a bit. The default is set to the same
    number of cores in your computer, but you can adjust it.

    Example to run 5 worker processes in parallel::

        python gmap.py -p 5 <Path to World> <Output Directory>

-z ZOOM, --zoom=ZOOM
    The Overviewer by default will detect how many zoom levels are required
    to show your entire map. This option sets it manually.

    *You do not normally need to set this option!*

    This is equivalent to setting the dimensions of the highest zoom level. It
    does not actually change how the map is rendered, but rather *how much of
    the map is rendered.* (Calling this option "zoom" may be a bit misleading,
    I know)
   
    To be precise, it sets the width and height of the highest zoom level, in
    tiles. A zoom level of z means the highest zoom level of your map will be
    2^z by 2^z tiles.

    This option map be useful if you have some outlier chunks causing your map
    to be too large, or you want to render a smaller portion of your map,
    instead of rendering everything.

    This will render your map with 7 zoom levels::

        python gmap.py -z 7 <Path to World> <Output Directory>

    Remember that each additional zoom level adds 4 times as many tiles as
    the last. This can add up fast, zoom level 10 has over a million tiles.
    Tiles with no content will not be rendered, but they still take a small
    amount of time to process.

-d, --delete
    This option changes the mode of execution. No tiles are rendered, and
    instead, cache files are deleted.

    Explanation: The Overviewer keeps two levels of cache: it saves each
    chunk rendered as a png, and it keeps a hash file along side each tile
    in your output directory. Using these cache files allows the Overviewer
    to skip rendering of any tile image that has not changed.

    By default, the chunk images are saved in your world directory. This
    example will remove them::
    
        python gmap.py -d <World # / Path to World / Path to cache dir>

    You can also delete the tile cache as well. This will force a full
    re-render, useful if you've changed texture packs and want your world
    to look uniform. Here's an example::

        python gmap.py -d <# / path> <Tile Directory>

    Be warned, this will cause the next rendering of your map to take
    significantly longer, since it is having to re-generate the files you just
    deleted.

--chunklist=CHUNKLIST
    Use this option to specify manually a list of chunks to consider for
    updating. Without this option, every chunk is checked for update and if
    necessary, re-rendered. If this option points to a file containing, 1 per
    line, the path to a chunk data file, then only those in the list will be
    considered for update.

    It's up to you to build such a list. On Linux or Mac, try using the "find"
    command. You could, for example, output all chunk files that are older than
    a certain date. Or perhaps you can incrementally update your map by passing
    in a subset of chunks each time. It's up to you!

--lighting
    This option enables map lighting, using lighting information stored by
    Minecraft inside the chunks. This will make your map prettier, at the cost
    of update speed.
    
    Note that for existing, unlit maps, you may want to clear your cache
    (with -d) before updating the map to use lighting. Otherwise, only updated
    chunks will have lighting enabled.

--night
    This option enables --lighting, and renders the world at night.

Viewing the Results
-------------------
Within the output directory you will find two things: an index.html file, and a
directory hierarchy full of images. To view your world, simply open index.html
in a web browser. Internet access is required to load the Google Maps API
files, but you otherwise don't need anything else.

You can throw these files up to a web server to let others view your map. You
do *not* need a Google Maps API key (as was the case with older versions of the
API), so just copying the directory to your web server should suffice. You are,
however, bound by the Google Maps API terms of service.

http://code.google.com/apis/maps/terms.html

Crushing the Output Tiles
-------------------------
Image files taking too much disk space? Try using pngcrush. On Linux and
probably Mac, if you have pngcrush installed, this command will go and crush
all your images in the given destination. This took the total disk usage of the
render for my world from 85M to 67M.

::

    find /path/to/destination -name "*.png" -exec pngcrush {} {}.crush \; -exec mv {}.crush {} \;

Or if you prefer a more parallel solution, try something like this::

    find /path/to/destination -print0 | xargs -0 -n 1 -P <nr_procs> sh -c 'pngcrush $0 temp.$$ && mv temp.$$ $0'

If you're on Windows, I've gotten word that this command line snippet works
provided pngout is installed and on your path. Note that the % symbols will
need to be doubled up if this is in a batch file.

::

    FOR /R c:\path\to\tiles\folder %v IN (*.png) DO pngout %v /y

Bugs
====
This program has bugs. They are mostly minor things, I wouldn't have released a
completely useless program. However, there are a number of things that I want
to fix or improve.

For a current list of issues, visit
http://github.com/brownan/Minecraft-Overviewer/issues

Feel free to comment on issues, report new issues, and vote on issues that are
important to you, so I can prioritize accordingly.

An incomplete list of things I want to do soon is:

* Improve efficiency

* Rendering non-cube blocks, such as torches, flowers, mine tracks, fences,
  doors, and the like. Right now they are either not rendered at all, or
  rendered as if they were a cube, so it looks funny.

* Some kind of graphical interface.

* A Windows exe for easier access for Windows users.
