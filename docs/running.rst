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

The following options change the way The Overviewer generates or updates the
map, and are intended to be things you only have to use in special situations.
You should not normally have to specify these options; the default is
typically correct.

.. cmdoption:: --no-tile-checks

    With this option, The Overviewer will determine which tiles to render by
    looking at the saved last-render timestamp and comparing it to the
    last-modified time of the chunks of the world. It builds a tree of tiles
    that need updating and renders only those tiles.

    This option does not do *any* checking of tile mtimes on disk, and thus is
    the cheapest option: only rendering what needs updating while minimising
    disk IO.

    The caveat is that the *only* thing to trigger a tile update is if Minecraft
    updates a chunk. Any other reason a tile may have for needing re-rendering
    is not detected. This means that changes in your render configuration will
    not be reflected in your world except in updated chunks. It could also cause
    problems if the system clock of the machine running Minecraft is not stable.

    **This option is the default** unless :option:`--forcerender` or
    :option:`--check-tiles` is in effect.  This option conflicts with
    :option:`--forcerender` and :option:`--check-tiles`.

.. cmdoption:: --check-tiles

    Forces The Overviewer to check each tile on disk and check to make sure it
    is up to date. This also checks for tiles that shouldn't exist and deletes
    them.

    This is functionally equivalent to :option:`--no-tile-checks` with the
    difference that each tile is individually checked. It is therefore useful if
    the tiles are not consistent with the last-render timestamp that is
    automatically stored. This option was designed to handle the case where the
    last render was interrupted -- some tiles have been updated but others
    haven't, so each one is checked before it is rendered.

    This is slightly slower than :option:`--no-tile-checks` due to the
    additional disk-io involved in reading tile mtimes from the filesystem

    Since this option also checks for erroneous tiles, **It is also useful after
    you delete sections of your map, e.g. with worldedit, to delete tiles that
    should no longer exist.** Overviewer greatly overestimates tiles to be
    rendered and time needed to complete.

    The caveats with this option are the same as for :option:`--no-tile-checks`
    with the additional caveat that tile timestamps in the filesystem must be
    preserved. If you copy tiles or make changes to them with an external tool
    that modifies mtimes of tiles, it could cause problems with this option.

    This option is automatically activated when The Overviewer detects the last
    render was interrupted midway through. This option conflicts with
    :option:`--forcerender` and :option:`--no-tile-checks`

.. cmdoption:: --forcerender

    Forces The Overviewer to re-render every tile regardless of whether it
    thinks it needs updating or not. It does no tile mtime checks, and therefore
    ignores the last render time of the world, the last modification times of
    each chunk, and the filesystem mtimes of each tile. It unconditionally
    renders every tile that exists.

    The caveat with this option is that it does *no* checks, period. Meaning it
    will not detect tiles that do exist, but shouldn't (this can happen if your
    world shrinks for some reason. For that specific case,
    :option:`--check-tiles` is actually the appropriate mode).

    This option is useful if you have changed a render setting and wish to
    re-render every tile with the new settings.

    This option is automatically activated for first-time renders. This option
    conflicts with :option:`--check-tiles` and :option:`--no-tile-checks`

.. cmdoption:: --genpoi

    .. note::
        Don't use this flag without first reading :ref:`signsmarkers`!

    Generates the POI markers for your map. This option does not do any tile/map
    generation, and ONLY generates markers. See :ref:`signsmarkers` on how to
    configure POI options.

.. cmdoption:: -p <procs>, --processes <procs>

    This specifies the number of worker processes to spawn on the local machine
    to do work. It defaults to the number of CPU cores you have, if not
    specified.

    This option can also be specified in the config file as :ref:`processes <processes>`

.. cmdoption:: --skip-scan

    .. note::
        Don't use this flag without first reading :ref:`signsmarkers`!

    When generating POI markers, this option prevents scanning for entities and
    tile entities, and only creates the markers specified in the config file.
    This considerably speeds up the POI marker generation process if no entities
    or tile entities are being used for POI markers. See :ref:`signsmarkers` on
    how to configure POI options.

.. cmdoption:: -v, --verbose

    Activate a more verbose logging format and turn on debugging output. This
    can be quite noisy but also gives a lot more info on what The Overviewer is
    doing.

.. cmdoption:: -q, --quiet

    Turns off one level of logging for quieter output. You can specify this more
    than once. One ``-q`` will suppress all INFO lines. Two will suppress all
    INFO and WARNING lines. And so on for ERROR and CRITICAL log messages.

    If :option:`--verbose<-v>` is given, then the first ``-q`` will counteract
    the DEBUG lines, but not the more verbose logging format. Thus, you can
    specify ``-v -q`` to get only INFO logs and higher (no DEBUG) but with the
    more verbose logging format.

.. cmdoption:: --update-web-assets

    Update web assets, including custom assets, without starting a render.
    This won't update overviewerConfig.js, but will recreate overviewer.js

.. _installing-textures:

Installing the Textures
=======================

.. note::
    This procedure has changed with Minecraft 1.6's Resource Pack update. The
    latest versions of Overviewer are not compatible with Minecraft 1.5 client
    resources.

If Overviewer is running on a machine with the Minecraft client installed, it
will automatically use the default textures from Minecraft.

.. note::
    Overviewer will only search for installed client *release* versions, not
    snapshots. If you want to use a snapshot client jar for the textures,
    you must specify it manually with the :ref:`texturepath<option_texturepath>`
    option.

If, however, you're running on a machine without the Minecraft client installed,
or if you want to use different textures, you will need to provide the textures
manually. This is common for servers.

If you want or need to provide your own textures, you have several options:

* The easy solution is to download the latest client jar to the location the
  launcher would normally install it. Overviewer will find it and use it.

  You can use the following commands to download the client jar on Linux or Mac.
  Run the first line in a terminal, changing the version string to the latest as appropriate
  (these docs may not always be updated to reflect the latest). Then paste the second line
  into your terminal to download the latest version. ``${VERSION}`` will be replaced
  by the acutal version string from the first line.

  ::

    VERSION=1.12
    wget https://s3.amazonaws.com/Minecraft.Download/versions/${VERSION}/${VERSION}.jar -P ~/.minecraft/versions/${VERSION}/

  If that's too confusing for you, then just take this single line and paste it into
  a terminal to get 1.12 textures::

    wget https://s3.amazonaws.com/Minecraft.Download/versions/1.12/1.12.jar -P ~/.minecraft/versions/1.12/

* You can also just run the launcher to install the client.

* You can transfer the client jar to the correct place manually, from a computer
  that does have the client, to your server. The correct places are:

  * For Linux: ``~/.minecraft/versions/<version>/<version>.jar``

  * For Mac: ``~/Library/Application Support/minecraft/versions/<version>/<version>.jar``

  * For Windows: ``%APPDATA%/.minecraft/versions/<version>/<version>.jar``

* You can download and use a custom resource pack. Download the resource pack
  file and specify the path to it with the
  :ref:`texturepath<option_texturepath>` option.

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

HTTPS support
-------------

In order to support displaying maps over HTTPS, Overviewer loads the Google
maps API and JQuery over HTTPS. This avoids security warnings for HTTPS
sites, and is not expected to cause problems for users.

If this change causes problems, take a look at the
:ref:`custom web assets<customwebassets>` option. This allows you to
provide a custom index.html which loads the required Javascript libraries
over HTTP.
