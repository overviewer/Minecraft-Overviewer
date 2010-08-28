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

I developed and tested this on Linux. It ought to work on Windows and Mac, but
I haven't tried it.

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

If you want to delete these images, to save space or perhaps you've changed
texture packs, run the renderer.py script with the -d flag:

    python renderer.py -d <Path to World>

To delete the cave mode images, run it with -d and -c

    python renderer.py -d -c <Path to World>

Using More Cores
----------------
The Overviewer will render each chunk separately in parallel. You can tell it
how many processes to start with the -p option. This is set to a default of 2,
which will use 2 processes to render chunks, and 1 to render the final image.

    python renderer.py -p 3 <Path to World> <image out.png>

Bugs
====
* This program is memory intensive. Obviously if you have a 20000x10000 pixel
  image, it's going to take up quite a bit of room. This program may not work
  if you have a gigantic world. I am working on a solution to this, possibly
  splitting up the final image so it's not as big. Even if the image is
  successfully generated, my image viewer has quite some trouble showing it.

* Some types of block are not rendered correctly yet. This includes torches,
  mushrooms, flowers, and anything that is not a traditional "block". They are
  still rendered, but look funny.

* Water transparency is not working yet. I'm trying to come up with a good
  solution, but I think it has to do with the image blending algorithm in the
  Python Imaging Library. There may not be an easy solution to this.
