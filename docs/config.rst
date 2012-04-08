.. _configfile:

======================
The Configuration File
======================

Using a configuration file is now the preferred way of running The Overviewer.
You will need to create a blank file and specify it when running The Overviewer
like this::

    overviewer.py --config=path/to/my_configfile


The config file is formatted in Python syntax. If you aren't familiar with
Python, don't worry, it's pretty simple. Just follow the examples.

A Simple Example
================

::

    worlds["My world"] = "/home/username/server/world"

    renders["normalrender"] = {
        "world": "My world",
        "title": "Normal Render of My World",
    }

    outputdir = "/home/username/mcmap"

This defines a single world, and a single render of that world. You can see
there are two main sections.

The ``worlds`` dictionary
    Define items in the ``worlds`` dictionary as shown to tell The Overviewer
    where to find your worlds. The keys to this dictionary ("My world" in the
    example) is a name you give, and is referenced later in the render
    dictionary. If you want to render more than one world, you would put more
    lines like this one. Otherwise, one is sufficient.

The ``renders`` dictionary
    Each item here declares a "render" which is a map of one dimension of one
    world rendered with the given options. If you declare more than one render,
    then you will get a dropdown box to choose which map you want to look at
    when viewing the maps.

    You are free to declare as many renders as you want with whatever options
    you want. For example, you are allowed to render multiple worlds, or even
    render the same world multiple times with different options.

.. note::

    Since this is Python syntax, keep in mind you need to put quotation marks
    around your strings. ``worlds[My world]`` will not work. It must be
    ``worlds["My world"]``

A more complicated example
==========================
::

    worlds["survival"] = "/home/username/server/survivalworld"
    worlds["creative"] = "/home/username/server/creativeworld"

    renders["survivalday"] = {
        "world": "survival",
        "title": "Survival Daytime",
        "rendermode": smooth_lighting,
        "dimension": "overworld",
    }

    renders["survivalnight"] = {
        "world": "survival",
        "title": "Survival Daytime",
        "rendermode": smooth_night,
        "dimension": "overworld",
    }

    renders["survivalnether"] = {
        "world": "survival",
        "title": "Survival Nether",
        "rendermode": nether_smooth_lighting,
        "dimension": "nether",
    }

    renders["creative"] = {
        "world": "creative",
        "title": "Creative",
        "rendermode": smooth_lighting,
        "dimension": "overworld",
    }

    outputdir = "/home/username/mcmap"
    texturepath = "/home/username/my_texture_pack.zip"

This config defines four maps for render. Two of them are of the survival
world's overworld, one is for the survival's nether, and one is for the creative
world.

Notice here we explicitly set the dimension property on each render. If
dimension is not specified, the default or overworld dimension is used. It is
necessary e.g. for the nether render.

Also note here we specify some different rendermodes. A rendermode refers to how
the map is rendered. The Overviewer can render a map in many different ways, and
there are many preset rendermodes, and you can even create your own (more on
that later).

And finally, note the usage of the ``texturepath`` option. This specifies a
texture pack to use for the rendering. Also note that it is set at the top level
of the config file, and therefore applies to every render. It could be set on
individual renders to apply to just those renders.

.. note::

    See the ``sample_config.py`` file included in the repository for another
    example.

Config File Specifications
==========================

The config file is a python file and is parsed with python's execfile() builtin.
This means you can put arbitrary logic in this file. The Overviewer gives the
execution of the file a local dict with a few pre-defined items (everything in
the overviewer_core.rendermodes module).

If the above doesn't make sense, just know that items in the config file take
the form ``key = value``. Two items take a different form:, ``worlds`` and
``renders``, which are described below.

``worlds``
    This is pre-defined as an empty dictionary. The config file is expected to
    add at least one item to it.

    Keys are arbitrary strings used to identify the worlds in the ``renders``
    dictionary.

    Values are paths to worlds (directories with a level.dat)

    e.g.::

        worlds['myworld'] = "/path/to/myworld"

    **You must specify at least one world**

``renders``
    This is also pre-defined as an empty dictionary. The config file is expected
    to add at least one item to it.

    Keys are strings that are used as the identifier for this render in the
    javascript, and also as the directory name for the tiles, but it's
    essentially up to you. It thus is recommended to make it a string with no
    spaces or special characters, only alphanumeric characters.

    Values are dictionaries specifying the configuration for the render. Each of
    these render dictionaries maps strings naming configuration options to their
    values. Valid keys and their values are listed in the :ref:`renderdict`
    section.

    e.g.::

        renders['myrender'] = {
                'world': 'myworld',
                'title': 'Minecraft Server Title',
                }

    **You must specify at least one render**

``outputdir = "<output directory path>"``
    This is the path to the output directory where the rendered tiles will
    be saved.

    e.g.::

        outputdir = "/path/to/output"

    **Required**

.. _processes:

``processes = num_procs``
    This specifies the number of worker processes to spawn on the local machine
    to do work. It defaults to the number of CPU cores you have, if not
    specified.

    This can also be specified with :option:`--processes <-p>`

    e.g.::

        processes = 2

.. _observer:

``observer = <observer object>``
    This lets you configure how the progress of the render is reported. The
    default is to display a progress bar, unless run on Windows or with stderr
    redirected to a file. The default value will probably be fine for most
    people, but advanced users may want to make their own progress reporter (for
    a web service or something like that) or you may want to force a particular
    observer to be used. The observer object is expected to have at least ``start``,
    ``add``, ``update``, and ``finish`` methods.

    e.g.::

        observer = ProgressBarObserver()

.. _outputdir:


.. _renderdict:

Render Dictonary Keys
---------------------

The render dictionary is a dictionary mapping configuration key strings to
values. The valid configuration keys are listed below.

.. note::

    Any of these items can be specified at the top level of the config file to
    set the default for every render. For example, this line at the top of the
    config file will set the world for every render to 'myworld' if no world is
    specified::

        world = 'myworld'

    Then you don't need to specify a ``world`` key in the render dictionaries::

        render['arender'] = {
                'title': 'This render doesn't explicitly declare a world!',
                }

``world``
    Specifies which world this render corresponds to. Its value should be a
    string from the appropriate key in the worlds dictionary.

    **Required**

``title``
    This is the display name used in the user interface. Set this to whatever
    you want to see displayed in the Map Type control (the buttons in the upper-
    right).

    **Required**

.. _option_dimension:

``dimension``
    Specified which dimension of the world should be rendered. Each Minecraft
    world has by default 3 dimensions: The Overworld, The Nether, and The End.
    Bukkit servers are a bit more complicated, typically worlds only have a
    single dimension, in which case you can leave this option off.

    The value should be a string. It should either be one of "overworld",
    "nether", "end", or the directory name of the dimension within the world.
    e.g. "DIM-1"

    .. note::

        If you choose to render your nether dimension, you must also use a
        nether :ref:`rendermode<option_rendermode>`. Otherwise you'll
        just end up rendering the nether's ceiling.

    **Default:** ``"overworld"``

.. _option_rendermode:

``rendermode``
    This is which rendermode to use for this render. There are many rendermodes
    to choose from. This can either be a rendermode object, or a string, in
    which case the rendermode object by that name is used.

    e.g.::

        "rendermode": "normal",

    Here are the rendermodes and what they do:

    ``"normal"``
        A normal render with no lighting. This is the fastest option.

    ``"lighting"``
        A render with per-block lighting, which looks similar to Minecraft
        without smooth lighting turned on. This is slightly slower than the
        normal mode.

    ``"smooth_lighting"``
        A render with smooth lighting, which looks similar to Minecraft with
        smooth lighting turned on.

        *This option looks the best* but is also the slowest.

    ``"night"``
        A "nighttime" render with blocky lighting.

    ``"smooth_night"``
        A "nighttime" render with smooth lighting

    ``"nether"``
        A normal lighting render of the nether. You can apply this to any
        render, not just nether dimensions. The only difference between this and
        normal is that the ceiling is stripped off, so you can actually see
        inside.

        .. note::

            Selecting this rendermode doesn't automatically render your nether
            dimension.  Be sure to also set the
            :ref:`dimension<option_dimension>` option to 'nether'.

    ``"nether_lighting"``
        Similar to "nether" but with blocky lighting.

    ``"nether_smooth_lighting"``
        Similar to "nether" but with smooth lighting.

    ``"cave"``
        A cave render with depth tinting (blocks are tinted with a color
        dependent on their depth, so it's easier to tell overlapping caves
        apart)

    **Default:** ``"normal"``

    .. note::

        The value for the 'rendermode' key can be either a *string* or
        *rendermode object* (strings simply name one of the built-in rendermode
        objects). The actual object type is a list of *rendermode primitive*
        objects.  See :ref:`customrendermodes` for more information.

``northdirection``
    This is direction that north will be rendered. This north direction will
    match the established north direction in the game where the sun rises in the
    east and sets in the west.

    Here are the valid north directions:

    * ``"upper-left"``
    * ``"upper-right"``
    * ``"lower-left"``
    * ``"lower-right"``

    **Default:** ``"upper-left"``

``rerenderprob``
    This is the probability that a tile will be rerendered even though there may
    have been no changes to any blocks within that tile. Its value should be a
    floating point number between 0.0 and 1.0.

    **Default:** ``0``

``imgformat``
    This is which image format to render the tiles into. Its value should be a
    string containing "png", "jpg", or "jpeg".

    **Default:** ``"png"``

``imgquality``
    This is the image quality used when saving the tiles into the JPEG image
    format. Its value should be an integer between 0 and 100.

    **Default:** ``95``

``bgcolor``
    This is the background color to be displayed behind the map. Its value
    should be either a string in the standard HTML color syntax or a 4-tuple in
    the format of (r,b,g,a). The alpha entry should be set to 0.

    **Default:** ``#1a1a1a``

.. _option_texture_pack:

``texturepath``
    This is a where a specific texture pack can be found to be used during this render.
    It can be either a folder or a directory. Its value should be a string.

.. _crop:

``crop``
    You can use this to render a small subset of your map, instead of the entire
    thing. The format is (min x, min z, max x, max z).

    The coordinates are block coordinates. The same you get with the debug menu
    in-game and the coordinates shown when you view a map.

    Example that only renders a 1000 by 1000 square of land about the origin::

        renders['myrender'] = {
                'world': 'myworld',
                'title': "Cropped Example",
                'crop': (-500, -500, 500, 500),
        }

    This option performs a similar function to the old ``--regionlist`` option
    (which no longer exists). It is useful for example if someone has wandered
    really far off and made your map too large. You can set the crop for the
    largest map you want to render (perhaps ``(-10000,-10000,10000,10000)``). It
    could also be used to define a really small render showing off one
    particular feature, perhaps from multiple angles.

    .. warning::

        If you decide to change the bounds on a render, you may find it produces
        unexpected results. It is recommended to not change the crop settings
        once it has been rendered once.

        For an expansion to the bounds, because chunks in the new bounds have
        the same mtime as the old, tiles will not automatically be updated,
        leaving strange artifacts along the old border. You may need to use
        :option:`--forcerender` to force those tiles to update.  (You can use
        the ``forcerender`` option on just one render by adding ``'forcerender':
        True`` to that render's configuration)

        For reductions to the bounds, you will need to render your map at least
        once with the :option:`--check-tiles` mode activated, and then once with
        the :option:`--forcerender` option. The first run will go and delete tiles that
        should no longer exist, while the second will render the tiles around
        the edge properly. Also see :ref:`this faq entry<cropping_faq>`.

        Sorry there's no better way to handle these cases at the moment. It's a
        tricky problem and nobody has devoted the effort to solve it yet.

``forcerender``
    This is a boolean. If set to ``True`` (or any non-false value) then this
    render will unconditionally re-render every tile regardless of whether it
    actually needs updating or not.

    The :option:`--forcerender` command line option acts similarly, but with
    one important difference. Say you have 3 renders defined in your
    configuration file. If you use :option:`--forcerender`, then all 3 of those
    renders get re-rendered completely. However, if you just need one of them
    re-rendered, that's unnecessary extra work.

    If you set ``'forcerender': True,`` on just one of those renders, then just
    that one gets re-rendered completely. The other two render normally (only
    tiles that need updating are rendered).

    You probably don't want to leave this option in your config file, it is
    intended to be used temporarily, such as after a setting change, to
    re-render the entire map with new settings. If you leave it in, then
    Overviewer will end up doing a lot of unnecessary work rendering parts of
    your map that may not have changed.

    Example::

        renders['myrender'] = {
                'world': 'myworld',
                'title': "Forced Example",
                'forcerender': True,
        }

``changelist``
    This is a string. It names a file where it will write out, one per line, the
    path to tiles that have been updated. You can specify the same file for
    multiple (or all) renders and they will all be written to the same file. The
    file is cleared when The Overviewer starts.

    This option is useful in conjunction with a simple upload script, to upload
    the files that have changed.

    .. warning::

        A solution like ``rsync -a --delete`` is much better because it also
        watches for tiles that should be *deleted*, which is impossible to
        convey with the changelist option. If your map ever shrinks or you've
        removed some tiles, you may need to do some manual deletion on the
        remote side.

.. _option_markers:

``markers``
    This controls the display of markers, signs, and other points of interest
    in the output HTML.  It should be a list of filter functions.

    .. note::

       Setting this configuration option alone does nothing.  In order to get
       markers and signs on our map, you must also run the genPO script.  See
       the :doc:`Signs and markers<signs>` section for more details and documenation.


    **Default:** ``[]`` (an empty list)

``showspawn``
    This is a boolean, and defaults to ``True``. If set to ``False``, then the spawn
    icon will not be displayed on the rendered map.

.. _customrendermodes:

Custom Rendermodes and Rendermode Primitives
============================================

We have generalized the rendering system. Every rendermode is made up of a
sequence of *rendermode primitives*. These primitives add some functionality to
the render, and stacked together, form a functional rendermode.  Some rendermode
primitives have options you can change. You are free to create your own
rendermodes by defining a list of rendermode primitives.

There are 9 rendermode primitives. Each has a helper class defined in
overviewer_core.rendermodes, and a section of C code in the C extension.

A list of rendermode primitives defines a rendermode. During rendering, each
rendermode primitive is applied in sequence. For example, the lighting
rendermode consists of the primitives "Base" and "Lighting". The Base primitive
draws the blocks with no lighting, and determines which blocks are occluded
(hidden). The Lighting primitive then draws the appropriate shading on each
block.

More specifically, each primitive defines a draw() and an is_occluded()
function. A block is rendered if none of the primitives determine the block is
occluded. A block is rendered by applying each primitives' draw() function in
sequence.

The Rendermode Primitives
-------------------------

Base
    This is the base of all non-overlay rendermodes. It renders each block
    according to its defined texture, and applies basic occluding to hidden
    blocks.

    **Options**

    biomes
        Whether to render biome coloring or not. Default: True.

        Set to False to disable biomes::

            nobiome_smooth_lighting = [Base(biomes=False), EdgeLines(), SmoothLighting()]

Nether
    This doesn't affect the drawing, but occludes blocks that are connected to
    the ceiling.

HeightFading
    Draws a colored overlay on the blocks that fades them out according to their
    height.

Depth
    Only renders blocks between the specified min and max heights.

    **Options**

    min
        lowest level of blocks to render. Default: 0

    max
        highest level of blocks to render. Default: 255

EdgeLines
    Draw edge lines on the back side of blocks, to help distinguish them from
    the background.

    **Options**

    opacity
        The darkness of the edge lines, from 0.0 to 1.0. Default: 0.15

Cave
    Occlude blocks that are in direct sunlight, effectively rendering only
    caves.

    **Options**

    only_lit
        Only render lit caves. Default: False

DepthTinting
    Tint blocks a color according to their depth (height) from bedrock. Useful
    mainly for cave renders.

Lighting
    Applies lighting to each block.

    **Options**

    strength
        how dark to make the shadows. from 0.0 to 1.0. Default: 1.0

    night
        whether to use nighttime skylight settings. Default: False

    color
        whether to use colored light. Default: False

SmoothLighting
    Applies smooth lighting to each block.

    **Options**

    (same as Lighting)

ClearBase
    Forces the background to be transparent. Use this in place of Base
    for rendering pure overlays.

    .. warning::

        Overlays are currently not functional in this branch of code. We are
        working on them. Please inquire in :ref:`IRC<help>` for more information.

SpawnOverlay
    Color the map red in areas where monsters can spawn. Either use
    this on top of other modes, or on top of ClearBase to create a
    pure overlay.

MineralOverlay
    Color the map according to what minerals can be found
    underneath. Either use this on top of other modes, or on top of
    ClearBase to create a pure overlay.

    **Options**

    minerals
        A list of (blockid, (r, g, b)) tuples to use as colors. If not
        provided, a default list of common minerals is used.

Defining Custom Rendermodes
---------------------------
Each rendermode primitive listed above is a Python *class* that is automatically
imported in the context of the config file (They come from
overviewer_core.rendermodes). To define your own rendermode, simply define a
list of rendermode primitive *objects* like so::

    my_rendermode = [Base(), EdgeLines(), SmoothLighting()]

If you want to specify any options, they go as parameters to the rendermode
primitive object's constructor::

    my_rendermode = [Base(), EdgeLines(opacity=0.2),
            SmoothLighting(strength=0.5, color=True)]

Then you can use your new rendermode in your render definitions::

    render["survivalday"] = {
        "world": "survival",
        "title": "Survival Daytime",
        "rendermode": my_rendermode,
        "dimension": "overworld",
    }

Note the lack of quotes around ``my_rendermode``. This is necessary since you
are referencing the previously defined list, not one of the built-in
rendermodes.

Built-in Rendermodes
--------------------
The built-in rendermodes are nothing but pre-defined lists of rendermode
primitives for your convenience. Here are their definitions::

    normal = [Base(), EdgeLines()]
    lighting = [Base(), EdgeLines(), Lighting()]
    smooth_lighting = [Base(), EdgeLines(), SmoothLighting()]
    night = [Base(), EdgeLines(), Lighting(night=True)]
    smooth_night = [Base(), EdgeLines(), SmoothLighting(night=True)]
    nether = [Base(), EdgeLines(), Nether()]
    nether_lighting = [Base(), EdgeLines(), Nether(), Lighting()]
    nether_smooth_lighting = [Base(), EdgeLines(), Nether(), SmoothLighting()]
    cave = [Base(), EdgeLines(), Cave(), DepthTinting()]
