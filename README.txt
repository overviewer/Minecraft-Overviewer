====================
Minecraft Overviewer
====================
By Andrew Brown

Generates large resolution images of a Minecraft map.

In short, this program reads in Minecraft world files and renders very large
resolution images. It performs a similar function to the existing Minecraft
Cartographer program.

I wrote this with an additional goal in mind: to generate large images that I
could zoom in and see details.

Using the Overviewer
====================

Requirements
------------
This program requires:

* Python 2.6
* PIL (Python Imaging Library)
* Numpy

I developed and tested this on Linux. It has been reported to work on Windows
and Mac, but if something doesn't, let me know. 

Running
-------
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
* This program is memory intensive. Obviously if you have a 20000x10000 pixel
  image, it's going to take up quite a bit of room. This program may not work
  if you have a gigantic world. The program may crash, or even if the image is
  successfully generated, your image viewer may crash or refuse to display it.
  I am working on a solution to this involving a google maps like interface
  where the world is split up into tiles.

* Some types of block are not rendered correctly yet. This includes torches,
  mushrooms, flowers, and anything that is not a traditional "block". Some are
  still rendered, but look funny. Others are not rendered at all currently.

* Water transparency is not working yet. I'm trying to come up with a good
  solution, but I think it has to do with the image blending algorithm in the
  Python Imaging Library. There may not be an easy solution to this.
