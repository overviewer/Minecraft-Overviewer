=======
Options
=======

.. contents::
    :local:

Command line options
====================

.. cmdoption:: -h, --help

    Shows the list of options and exits

.. cmdoption:: --advanced-help

    Display help - including advanced options

Useful Options
--------------
.. cmdoption:: --rendermodes=MODE1[,MODE2,...]

    Use this option to specify which render mode to use, such as lighting or
    night. Use --list-rendermodes to get a list of available rendermodes, and
    a short description of each. If you provide more than one mode (separated
    by commas), Overviewer will render all of them at once, and provide a
    toggle on the resulting map to switch between them.
    
    If for some reason commas do not work for your shell (like if you're using
    Powershell on Windows), you can also use a colon ':' or a forward slash '/'
    to separate the modes.

    See the `Render Modes`_ section for more information.

.. cmdoption:: --list-rendermodes

    List the available render modes, and a short description of each.

.. cmdoption:: --north-direction=NORTH_DIRECTION

    Specifies which corner of the screen north will point to.
    Valid options are: lower-left, upper-left, upper-right, lower-right.
    If you do not specify this option, it will default to whatever direction
    the existing map uses. For new maps, it defaults to lower-left for
    historical reasons.

.. cmdoption:: --settings=PATH

    Use this option to load settings from a file. For more information see the
    `Settings File`_ section below.

Less Useful Options
-------------------

.. cmdoption:: -p PROCS, --processes=PROCS

    Adding the "-p" option will utilize more cores during processing.  This
    can speed up rendering quite a bit. The default is set to the same
    number of cores in your computer, but you can adjust it.

    Example to run 5 worker processes in parallel::

        python overviewer.py -p 5 <Path to World> <Output Directory>

.. cmdoption:: -d, --delete

    This option changes the mode of execution. No tiles are rendered, and
    instead, files are deleted.

    *Note*: Currently only the overviewer.dat file is deleted when you run with
    this option

.. cmdoption:: --forcerender

    Force re-rendering the entire map (or the given regionlist). This
    is an easier way to completely re-render without deleting the map.

.. cmdoption:: --regionlist=regionlist

    Use this option to specify manually a list of regions to consider for
    updating. Without this option, every chunk in every region is checked for
    update and if necessary, re-rendered. If this option points to a file
    containing, 1 per line, the path to a region data file, then only those
    in the list will be considered for update.

    It's up to you to build such a list. On Linux or Mac, try using the "find"
    command. You could, for example, output all region files that are older than
    a certain date. Or perhaps you can incrementally update your map by passing
    in a subset of regions each time. It's up to you!



Settings File
=============

You can optionally store settings in a file named settings.py (or really,
anything you want).  It is a regular python script, so you can use any python
functions or modules you want. To use a settings file, use the --settings
command line option.

For a sample settings file, look at 'sample.settings.py'. Note that this file
is not meant to be used directly, but instead it should be used as a
collection of examples to guide writing your own.

Here's a (possibly incomplete) list of available settings, which are available
in settings.py. Note that you can also set command-line options in a similar
way.

imgformat=FORMAT
    Set the output image format used for the tiles. The default is 'png',
    but 'jpg' is also supported.

zoom=ZOOM
    The Overviewer by default will detect how many zoom levels are required
    to show your entire map. This option sets it manually.

    *You do not normally need to set this option!*

    This is equivalent to setting the dimensions of the highest zoom level. It
    does not actually change how the map is rendered, but rather *how much of
    the map is rendered.* Setting this option too low *will crop your map.*
    (Calling this option "zoom" may be a bit misleading, I know)
   
    To be precise, it sets the width and height of the highest zoom level, in
    tiles. A zoom level of z means the highest zoom level of your map will be
    2^z by 2^z tiles.

    This option map be useful if you have some outlier chunks causing your map
    to be too large, or you want to render a smaller portion of your map,
    instead of rendering everything.

    Remember that each additional zoom level adds 4 times as many tiles as
    the last. This can add up fast, zoom level 10 has over a million tiles.
    Tiles with no content will not be rendered, but they still take a small
    amount of time to process.

web_assets_hook
    This option lets you define a function to run after the web assets have
    been copied into the output directory, but before any tile rendering takes
    place. This is an ideal time to do any custom postprocessing for
    markers.js or other web assets.
    
    This function should accept one argument: a QuadtreeGen object.

web_assets_path
    This option lets you provide alternative web assets to use when
    rendering. The contents of this folder will be copied into the output folder
    during render, and will overwrite any default files already copied by
    Overviewer. See the web_assets folder included with Overviewer for the
    default assets.

textures_path
    This is like web_assets_path, but instead it provides an alternative texture
    source. Overviewer looks in here for terrain.png and other textures before
    it looks anywhere else.

north_direction
    Specifies which corner of the screen north will point to.
    Valid options are: lower-left, upper-left, upper-right, lower-right.

Render Modes
============

.. _rendermode-options: https://github.com/agrif/Minecraft-Overviewer/tree/rendermode-options

Rendermode options are a new way of changing how existing render modes
work, by passing in values at startup. For example, you can change how
dark the 'night' mode is, or enable lighting in 'cave' mode.

Options and Rendermode Inheritance
----------------------------------

Each mode will accept its own options, as well as the options for
parent modes; for example the 'night' mode will also accept options
listed for 'lighting' and 'normal'. Also, if you set an option on a
mode, all its children will also have that option set. So, setting the
'edge_opacity' option on 'normal' will also set it on 'lighting' and
'night'.

Basically, each mode inherits available options and set options from
its parent.

Eventually the :option:`--list-rendermodes` option will show parent
relationships. Right now, it looks something like this:

* normal

  * lighting

    * night
    * cave

* overlay

  * spawn
  * mineral

How to Set Options
------------------

Available options for each mode are listed below, but once you know
what to set you'll have to edit *settings.py* to set them. Here's an
example::

    rendermode_options = {
        'lighting': {
            'edge_opacity': 0.5,
        },

        'cave': {
            'lighting': True,
            'depth_tinting': False,
        },
    }

As you can see, each entry in ``rendermode_options`` starts with the mode name
you want to apply the options to, then a dictionary containing each option. So
in this example, 'lighting' mode has 'edge_opacity' set to 0.5, and 'cave' mode
has 'lighting' turned on and 'depth_tinting' turned off.

Defining Custom Rendermodes
---------------------------

Sometimes, you want to render two map layers with the same mode, but with two
different sets of options. For example, you way want to render a cave mode with
depth tinting, and another cave mode with lighting and no depth tinting. In this
case, you will want to define a 'custom' render mode that inherits from 'cave'
and uses the options you want. For example::

    custom_rendermodes = {
        'cave-lighting': {
            'parent': 'cave',
            'label': 'Lit Cave',
            'description': 'cave mode, with lighting',
            'options': {
                'depth_tinting': False,
                'lighting': True,
            }
        },
    }

    rendermode = ['cave', 'cave-lighting']

Each entry in ``custom_rendermodes`` starts with the mode name, and is followed
by a dictionary of mode information, such as the parent mode and description
(for your reference), a label for use on the map, as well as the options to
apply.

Every custom rendermode you define is on exactly equal footing with the built-in
modes: you can put them in the ``rendermode`` list to render them, you can
inherit from them in other custom modes, and you can even add options to them
with ``rendermode_options``, though that's a little redundant.

Option Listing
--------------

Soon there should be a way to pull out supported options from Overviewer
directly, but for right now, here's a reference of currently supported options.

normal
~~~~~~

* **edge_opacity** - darkness of the edge lines, from 0.0 to 1.0 (default: 0.15)
* **min_depth** - lowest level of blocks to render (default: 0)
* **max_depth** - highest level of blocks to render (default: 127)
* **height_fading** - darken or lighten blocks based on height (default: False)

lighting
~~~~~~~~

all the options available in 'normal', and...

* **shade_strength** - how dark to make the shadows, from 0.0 to 1.0 (default: 1.0)

night
~~~~~

'night' mode has no options of its own, but it inherits options from
'lighting'.

cave
~~~~

all the options available in 'normal', and...

* **depth_tinting** - tint caves based on how deep they are (default: True)
* **only_lit** - only render lit caves (default: False)
* **lighting** - render caves with lighting enabled (default: False)

mineral
~~~~~~~

The mineral overlay supports one option, **minerals**, that has a fairly
complicated format. **minerals** must be a list of ``(blockid, (r, g, b))``
tuples that tell the mineral overlay what blocks to look for. Whenever a block
with that block id is found underground, the surface is colored with the given
color.

See the *settings.py* example below for an example usage of **minerals**.

Example *settings.py*
---------------------

This *settings.py* will render three layers: a normal 'lighting' layer, a 'cave'
layer restricted to between levels 40 and 55 to show off a hypothetical subway
system, and a 'mineral' layer that has been modified to show underground rail
tracks instead of ore.

::

    rendermode = ['lighting', 'subway-cave', 'subway-overlay']

    custom_rendermodes = {
        'subway-cave' : {'parent' : 'cave', 
                         'label' : 'Subway',
                         'description' : 'a subway map, based on the cave rendermode',
                         'options' : {
                             'depth_tinting' : False,
                             'lighting' : True,
                             'only_lit' : True,
                             'min_depth' : 40,
                             'max_depth' : 55,
                         }
        },
        'subway-overlay' : {'parent' : 'mineral',
                            'label' : 'Subway Overlay',
                            'description' : 'an overlay showing the location of minecart tracks',
                            'options' : {'minerals' : [
                                (27, (255, 234, 0)),
                                (28, (255, 234, 0)),
                                (66, (255, 234, 0)),
                            ]}
        },
    }

    rendermode_options = {
        'lighting' : {'edge_opacity' : 0.5},
    #    'night' : {'shade_strength' : 0.5},
    #    'cave' : {'only_lit' : True, 'lighting' : True, 'depth_tinting' : False},
    }
