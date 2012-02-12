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

    render["normalrender"] = {
        "worldname": "My world",
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
    Each item here declares a "render" which is a map of a world rendered with a
    set of options. If you have more than one, when viewing the maps, you will
    get a dropdown box to choose which map you want to look at.

    You can render the same world multiple times with different options, or
    render multiple worlds.

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
        "worldname": "survival",
        "title": "Survival Daytime",
        "rendermode": smooth_lighting,
        "dimension": "overworld",
    }

    renders["survivalnight"] = {
        "worldname": "survival",
        "title": "Survival Daytime",
        "rendermode": smooth_night,
        "dimension": "overworld",
    }

    renders["survivalnether"] = {
        "worldname": "survival",
        "title": "Survival Nether",
        "rendermode": nether_smooth_lighting,
        "dimension": "nether",
    }

    renders["survivalspawnoverlay"] = {
        "worldname": "survival",
        "title": "Spawn Overlay",
        "rendermode": spawn_overlay,
        "dimension": "overworld",
        "overlay": ["survivalday", "survivalnight"],
    }

    renders["creative"] = {
        "worldname": "creative",
        "title": "Creative",
        "rendermode": smooth_lighting,
        "dimension": "overworld",
    }

    outputdir = "/home/username/mcmap"
    textures = "/home/username/my_texture_pack.zip"

This config defines four maps for render, and one overlay. Two of them are of
the survival world's overworld, one is for the survival's nether, and one is for
a creative world. The overlay is the "spawn_overlay" (which highlights areas
that are dark enough for monsters to spawn) and it will be available when
viewing the survivalday and survivalnight maps.

Notice here we explicitly set the dimension property on each render. If
dimension is not specified, the default or overworld dimension is used.

Also note here we specify some different rendermodes. A rendermode refers to how
the map is rendered. The Overviewer can render a map in many different ways, and
there are many preset rendermodes, and you can even create your own (more on
that later).

And finally, note the usage of the ``textures`` option. This specifies a texture
pack to use for the rendering.

Config File Specifications
==========================

The config file is a python file and is parsed with python's execfile() builtin.
This means you can put arbitrary logic in this file. The Overviewer gives the
execution of the file a local dict with a few pre-defined items (everything in
the overviewer_core.rendermodes module).

After the config file is evaluated, the ``worlds`` and ``renders`` dictionaries,
along with other global level configuration options, are used to configure The
Overviewer's rendering.

``worlds``
    This is pre-defined as an empty dictionary. The config file is expected to
    add at least one item to it.

    Keys are arbitrary strings used to identify the worlds in the ``renders``
    dictionary.

    Values are paths to worlds (directories with a level.dat)

``renders``
    This is also pre-defined as an empty dictionary. The config file is expected
    to add at least one item to it.

    Keys are strings that are used as the identifier for this render in the
    javascript, and also as the directory name for the tiles. It thus is
    recommended to make it a string with no spaces or special characters, only
    alphanumeric characters.

    Values are dictionaries specifying the configuration for the render. Each of
    these render dictionaries maps strings naming configuration options to their
    values. Valid keys and their values are listed below.

Render Dictonary Keys
---------------------

``worldname``
    Specifies which world this render corresponds to. Its value should be a
    string from the appropriate key in the worlds dictionary.

    **Required**

``dimension``
    Specified which dimension of the world should be rendered. Each Minecraft
    world has by default 3 dimensions: The Overworld, The Nether, and The End.
    Bukkit servers are a bit more complicated, typically worlds only have a
    single dimension, in which case you can leave this option off.

    The value should be a string. It should either be one of "overworld",
    "nether", "end", or the directory name of the dimension within the world.
    e.g. "DIM-1"

    **Default: "overworld"**

``title``
    This is the display name used in the user interface. Set this to whatever
    you want to see displayed in the Map Type control (the buttons in the upper-
    right).

``rendermode``
    This is which rendermode to use for this render. There are many rendermodes
    to choose from. This can either be a rendermode object, or a string, in
    which case the rendermode object by that name is used.

    Here are the rendermodes and what they do:

    normal
        A normal render with no lighting. This is the fastest option.

    lighting
        A render with per-block lighting, which looks similar to Minecraft
        without smooth lighting turned on. This is slightly slower than the
        normal mode.

    smooth_lighting
        A render with smooth lighting, which looks similar to Minecraft with
        smooth lighting turned on.

        *This option looks the best* but is also the slowest.

    night
        A "nighttime" render with blocky lighting.

    smooth_night
        A "nighttime" render with smooth lighting

    nether
        A normal lighting render of the nether. You can apply this to any
        render, not just nether dimensions. The only difference between this and
        normal is that the ceiling is stripped off, so you can actually see
        inside.
        
    nether_lighting
        Similar to "nether" but with blocky lighting.

    nether_smooth_lighting
        Similar to "nether" but with smooth lighting.
    
    Technical note: The actual object type for this option is a list of
    *rendermode primitive* objects. See :ref:`customrendermodes` for more
    information.

    **Default: normal**

Global Options
--------------
These values are set directly in the config file. Example::

    texture_pack = "/home/username/minecraft/my_texture_pack.zip"

.. _option_texture_pack:

``texture_pack = "<texture pack path>"``
    This is a string indicating the path to the texture pack to use for
    rendering.

.. _processes:

``processes = num_procs``
    This specifies the number of worker processes to spawn on the local machine
    to do work. It defaults to the number of CPU cores you have, if not
    specified.
 
    This can also be specified with :option:`--processes <-p>`

.. _outputdir:

``outputdir = "<output directory path>"``
    This is the path to the output directory where the rendered tiles will
    be saved.

TODO: More to come here

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
        highest level of blocks to render. Default: 127

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
