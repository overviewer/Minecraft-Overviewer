================
Quickstart Guide
================

This guide is aimed at new users that want to get started using Minecraft
Overviewer. It is *not* meant to explain everything, but it should help you
generate your first map.

Getting the Overviewer
======================

Head to our `Github Homepage <https://github.com/overviewer/Minecraft-Overviewer>`_. You can either download the Windows package if you're running Windows, install the Debian package if you're running Debian or Ubuntu, or Git-clone the source. Building from source should be as simple as a `python setup.py build` but for more information, see `Building the Overviewer from Source <building.html>`_.

Quick-link for Git Source. (Clone this)
    git://github.com/overviewer/Minecraft-Overviewer.git

Rendering your First Map
========================

Overviewer is a command-line application, and so it needs to be run from the command line. If you installed Overviewer from a package manager, the command is ``overviewer.py``. If you downloaded it manually, open a terminal window and navigate to wherever you downloaded Overviewer. For pre-compiled Windows builds, the command is ``overviewer.exe``. For other systems, it's ``./overviewer.py``.

To generate your map, run::

    overviewer.exe WorldName path\to\output\                 # on windows, or
    ./overviewer.py WorldName path/to/output/                 # on other systems

where ``WorldName`` is the name of the world you want to render, and
``path/to/output`` is the place where you want to store the rendered world. The
first render can take a while, depending on the size of your world. You can, if
you want to, provide a path to the world you want to render, instead of
providing a world name and having Overviewer auto-discover the world path.

When the render is done, open up *index.html* using your web-browser of choice. Pretty cool, huh? You can even upload this map to a web server to share with others! Simply upload the entire folder to a web server and point your users to index.html!

Incremental updates are just as easy, and a lot faster. If you go and change something inside your world, run the command again and Overviewer will automatically rerender only what's needed.

Running Overviewer on a Server
------------------------------

There are special considerations when running Overviewer on a server. For
information on how to do this, see `Running Overviewer on a Server`_.

.. _Running Overviewer on a Server: https://github.com/overviewer/Minecraft-Overviewer/wiki/Running-Overviewer-on-a-Server

Extra Features
==============

Overviewer has a lot of features beyond generating the simple map we started with. Here's information on two of them.

Render Modes
------------

Overviewer supports many different rendermodes. Run `./overviewer.py --list-rendermodes` to get a list. Two of the most popular rendermodes are *lighting* and *night*, which draw shadows for the corresponding time of day. To tell Overviewer what rendermode to use, run

    ./overviewer.py --rendermodes=lighting WorldName output/dir/

You can also specify multiple rendermodes at once, and Overviewer will render
them all and let you toggle between them on the generated web page. To get both
*lighting* and *night* on the same page, run::

    ./overviewer.py --rendermodes=lighting,night WorldName output/dir/

Biomes
------

Minecraft Overviewer has support for using the biome info from the `Minecraft
Biome Extractor`_. If you run the biome extractor on your world, during the
next run Overviewer will automatically recognize the biome info and use it to
colorize your grass and leaves appropriately. This will only appear on updated
chunks, though; to colorize the entire world you will need to rerender from
scratch by deleting the old render.

**Note**: as of Minecraft 1.8, you currently need to use a patched Biome
Extractor that can be found `here <http://www.minecraftforum.net/topic/76063-minecraft-biome-extractor-add-biome-support-to-your-mapper/page__st__140__gopid__8431028#entry8431028>`_, or `here on GitHub
<https://github.com/overviewer/minecraft-biome-extractor>`_.

.. _Minecraft Biome Extractor: http://www.minecraftforum.net/viewtopic.php?f=25&t=80902
