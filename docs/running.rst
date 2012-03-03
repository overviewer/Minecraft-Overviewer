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

What follows in this section is a few examples to get you started. For full
usage, see the :ref:`usage` section.

So, let's render your first map! Let's say you want to render your single player
world called "My World". Let's also say you want to save it c:\mcmap. You
would type into your command prompt the following::

    overviewer.exe "My World" c:\mcmap

If you're on Linux or a Mac, you could do something like one of the following::

    overviewer.py "My World" /home/username/mcmap

or

::

    overviewer.py "My World" /Users/username/mcmap

Those will look for a single player world by that name. You can also specify the
path to the world you want to render. This is useful for rendering servers.

Let's say you have a server installed in /home/username/mcserver. This command
will render the default dimension (in the case of Bukkit multiworld servers, the
default world is used. You can also specify the directory to the specific world
you want to render).

::

    overviewer.py /home/username/mcserver /home/username/mcmap

After you enter one of the commands, The Overviewer should start rendering your
map. When the render is done, open up *index.html* using your web-browser of
choice.  Pretty cool, huh? You can even upload this map to a web server to share
with others! Simply upload the entire folder to a web server and point your
users to index.html!

Incremental updates are just as easy, and a lot faster. If you go and change
something inside your world, run the command again and The Overviewer will
automatically re-render only what's needed.

Specifying a different rendermode
---------------------------------
There are a few built-in rendermodes for you to choose from. Each will render
your map differently. For example, if you want smooth lighting (which looks
really good), you would add ``--rendermodes=smooth-lighting`` to your command.
e.g.

::

    overviewer.py --rendermodes=smooth-lighting /home/username/mcserver /home/username/mcmap

The rendermodes you have to choose from are:

* normal (the default)
* lighting
* smooth-lighting
* cave

You can specify more than one. Just separate them with a comma!

.. _usage:

Usage
=====

For this section, we assume the executable is ``overviewer.py``. Replace that
with ``overviewer.exe`` for windows. 

Overviewer usage::

    overviewer.py [--rendermodes=...] [options] <World> <Output Dir>
    overviewer.py --config=<config file> [options]

The first form is for basic or quick renderings without having to create a
config file. It is intentionally limited because the amount of configuration was
becoming unmanageable for the command line.

The second, preferred usage involves creating a configuration file which
specifies all the options including what to render, where to place the output,
and all the settings. See :ref:`configfile` for details on that.

For example, on Windows if your Minecraft server runs out of ``c:\server\`` and you want
to put the rendered map in ``c:\mcmap\``, run this::

    overviewer.exe c:\server\world c:\mcmap

For Mac or Linux builds from source, you would run something like this with the
current directory in the top level of the source tree::

    ./overviewer.py /opt/minecraft/server/world /opt/minecraft/mcmap

The first render can take a while, depending on the size of your world.

.. _options:

Options
-------

These options change the way the render works, and are intended to be things you
only have to use once-in-a-while.

.. cmdoption:: --forcerender

    Forces The Overviewer to re-render every tile regardless of whether it
    thinks it needs updating or not. This is similar to deleting your output
    directory and rendering from scratch.

    This is the default mode for first-time renders. This option overrides
    :option:`--check-tiles` and :option:`--no-tile-checks`

.. cmdoption:: --check-tiles

    Forces The Overviewer to check each tile on disk and compare its
    modification time to the modification time of the part of the world that
    tile renders. This is slightly slower than the default, but can be useful if
    there are some tiles that somehow got skipped.

    This option is the default when The Overviewer detects the last render was
    interrupted midway through. This option overrides :option:`--forcerender`
    and :option:`--no-tile-checks`

.. cmdoption:: --no-tile-checks

    With this option, The Overviewer will not do any checking of tiles on disk
    to determine what tiles need updating. Instead, it will look at the time
    that the last render was performed, and render parts of the map that were
    changed since then. This is the fastest option, but could cause problems if
    the clocks of the Minecraft server and the machine running The Overviewer
    are not in sync.

    This option is the default unless the condition for :option:`--forcerender`
    or :option:`--check-tiles` is in effect.  This option overrides
    :option:`--forcerender` and :option:`--check-tiles`.

.. cmdoption:: -p <procs>, --processes <procs>

    This specifies the number of worker processes to spawn on the local machine
    to do work. It defaults to the number of CPU cores you have, if not
    specified.

    This option can also be specified in the config file as :ref:`processes <processes>`

.. _installing-textures:

Installing the Textures
=======================

If Overviewer is running on a machine with the Minecraft client installed, it
will automatically use the default textures from Minecraft.

If, however, you're running on a machine without the Minecraft client installed,
or if you want to use different textures, you will need to provide the textures
manually. This is common for servers.

If you want or need to provide your own textures, you have several options:

* If you're running the Overviewer on a server, you can still put the
  minecraft.jar file (not the launcher) into the correct location and the
  Overviewer will find and use it, thinking the client is installed, even if the
  rest of the client files are missing. On Linux, try a command like this::

      wget -N http://s3.amazonaws.com/MinecraftDownload/minecraft.jar -P ~/.minecraft/bin/

* You can manually extract the terrain.png from minecraft.jar or your favorite
  texture pack. If you've built the Overviewer from source or are using the
  windows exe, place the file in the same directory as overviewer.py or
  overviewer.exe.

* Specify any terrain.png or texture pack you want with the
  :ref:`texture_pack<option_texture_pack>` option.

If you copy your world before you render it
-------------------------------------------

The important thing to be careful about when copying world files to another
location is file modification times, which Overviewer uses to figure out what
parts of the map need updating. If you do a straight copy, usually this will
update the modification times on all the copied files, causing Overviewer to
re-render the entire map. To copy files on Unix, while keeping these
modification times intact, use ``cp -p``. For people who render from backups,
GNU ``tar`` automatically handles modification times correctly. ``rsync -a
--delete`` will handle this correctly as well. If you use some other tool,
you'll have to figure out how to do this yourself.
