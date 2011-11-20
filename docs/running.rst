======================
Running the Overviewer
======================

Rendering your First Map
========================

Overviewer is a command-line application, and so it needs to be run from the
command line. If you installed Overviewer from a package manager, the command is
``overviewer.py``. If you downloaded it manually, open a terminal window and
navigate to wherever you downloaded Overviewer. For pre-compiled Windows builds,
the command is ``overviewer.exe``. For other systems, it's ``overviewer.py``.

The basic usage for Windows is::

    overviewer.exe [options] <World> <Output Dir>

And similarly for other systems::

    overviewer.py [options] <World> <Output Dir>

**World**
    World can be one of several things.

    1. The path to your Minecraft world on your hard drive
    2. The name of a single player world on your current system. Note that if it
       has spaces, you will need to put the world name in quotes.
    3. If your single-player world name is in the format "World #" (e.g. "World
       1"), you can just specify the number.

**Output Dir**
    This is the directory you would like to put the rendered tiles and
    supporting HTML and javascript files. You should use the same output
    directory each time; the Overviewer will automatically re-render only the
    tiles that need rendering on subsequent runs.

**options**
    See the :doc:`options` page for a list of options you can
    specify.

For example, on Windows if your Minecraft server runs out of ``c:\server\`` and you want
to put the rendered map in ``c:\mcmap\``, run this::

    overviewer.exe c:\server\world c:\mcmap

For Mac or Linux builds from source, you would run something like this with the
current directory in the top level of the source tree::

    ./overviewer.py /opt/minecraft/server/world /opt/minecraft/mcmap

The first render can take a while, depending on the size of your world.

When the render is done, open up *index.html* using your web-browser of choice.
Pretty cool, huh? You can even upload this map to a web server to share with
others! Simply upload the entire folder to a web server and point your users to
index.html!

Incremental updates are just as easy, and a lot faster. If you go and change
something inside your world, run the command again and Overviewer will
automatically re-render only what's needed.

.. _installing-textures:

Installing the Textures
=======================
If you're running on a machine without the Minecraft client installed, you will
need to provide the terrain.png file manually for the Overviewer to use in
rendering your world. This is common for servers.

All Overviewer needs is a terrain.png file. If the Minecraft client is
installed, it will use the terrain.png that comes with Minecraft. If the
Minecraft client is not installed or you wish to use a different terrain.png,
for example a custom texture pack, read on.

You have several options:

* If you have the Minecraft client installed, the Overviewer will automatically
  use those textures. This is a good solution since the Minecraft Launcher will
  always keep this file up-to-date and you don't have to do anything extra.

  * If you're running the Overviewer on a server, you can still put the
    minecraft.jar file (not the launcher) into the correct location and the
    Overviewer will find and use it, even if the rest of the client files are
    missing. On Linux, try a command like this::

        wget -N http://s3.amazonaws.com/MinecraftDownload/minecraft.jar -P ~/.minecraft/bin/

* You can manually extract the terrain.png from minecraft.jar or your favorite
  texture pack. If you've built the Overviewer from source, simply place the
  file in the same directory as overviewer.py or overviewer.exe. For
  installations, you will need to specify the path... see the next bullet.

* You can put a terrain.png file anywhere you want and point to its
  location with the :option:`--textures-path` option. This should
  point to the directory containing the terrain.png, not to the file
  itself.

* Alternately, you can download any texture pack ZIP you like and
  point to this directly with :option:`--textures-path`.

Note: the :option:`--check-terrain` option is useful for debugging terrain.png issues.
For example::

    $ ./overviewer.py --check-terrain
    2011-09-26 21:51:46,494 [INFO] Found terrain.png in '/home/achin/.minecraft/bin/minecraft.jar'
    2011-09-26 21:51:46,497 [INFO] Hash of terrain.png file is: `6d53f9e59d2ea8c6f574c9a366f3312cd87338a8` 

::

    $ ./overviewer.py --check-terrain --textures-path=/tmp
    2011-09-26 21:52:52,143 [INFO] Found terrain.png in '/tmp/terrain.png'
    2011-09-26 21:52:52,145 [INFO] Hash of terrain.png file is: `6d53f9e59d2ea8c6f574c9a366f3312cd87338a8`

Running on a Live Map
=====================
If you're running the Overviewer on a live server or a single player world
that's running, read this section.

Minecraft doesn't really like it when other programs go snooping around in a
live world, so running Overviewer on a live world usually creates a few errors,
usually "corrupt chunk" errors. You *can* do this, but it's not a supported way
of running Overviewer.

To get around this, you can copy your live world somewhere else, and render the
copied world instead. If you're already making backups of your world, you can
use the backups to make the render. Many people even use their backups to run
Overviewer on a different machine than the one running the Minecraft server.

There used to be a few things to be careful about, but right now there's only
one important thing left.

Preserving Modification Times
-----------------------------

The important thing to be careful about when copying world files to another
location is file modification times, which Overviewer uses to figure out what
parts of the map need updating. If you do a straight copy, usually this will
update the modification times on all the copied files, causing Overviewer to
re-render the entire map. To copy files on Unix, while keeping these
modification times intact, use ``cp -p``. For people who render from backups,
GNU ``tar`` automatically handles modification times correctly. ``rsync -a``
will handle this correctly as well. If you use some other tool, you'll have to
figure out how to do this yourself.

Biome Support
=============

Minecraft Overviewer has support for using the biome info from the `Minecraft
Biome Extractor`_. If you run the biome extractor on your world, during the
next run Overviewer will automatically recognize the biome info and use it to
colorize your grass and leaves appropriately. This will only appear on updated
chunks, though; to colorize the entire world you will need to re-render from
scratch by using :option:`--forcerender`

.. note::

    as of Minecraft 1.8, you currently need to use a patched Biome Extractor
    that can be found `here
    <http://www.minecraftforum.net/topic/76063-minecraft-biome-extractor-add-biome-support-to-your-mapper/page__st__140__gopid__8431028#entry8431028>`_,
    or `here on GitHub
    <https://github.com/overviewer/minecraft-biome-extractor>`_.

.. _Minecraft Biome Extractor: http://www.minecraftforum.net/viewtopic.php?f=25&t=80902
