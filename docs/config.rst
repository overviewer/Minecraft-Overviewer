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

.. note::

    You should *always* use forward slashes ("/"), even on
    Windows.  This is required because the backslash ("\\") has special meaning
    in Python.  

Examples
========

The following examples should give you an idea of what a configuration file looks
like, and also teach you some neat tricks.

A Simple Example
----------------

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
--------------------------
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
        "title": "Survival Nighttime",
        "rendermode": smooth_night,
        "dimension": "overworld",
    }

    renders["survivalnether"] = {
        "world": "survival",
        "title": "Survival Nether",
        "rendermode": nether_smooth_lighting,
        "dimension": "nether",
    }

    renders["survivalnethersouth"] = {
        "world": "survival",
        "title": "Survival Nether",
        "rendermode": nether_smooth_lighting,
        "dimension": "nether",
        "northdirection" : "lower-right",
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
texture pack (also called a resource pack) to use for the rendering. Also note
that it is set at the top level of the config file, and therefore applies to
every render. It could be set on individual renders to apply to just those
renders.

.. note::

    See the ``sample_config.py`` file included in the repository for another
    example.

A dynamic config file
---------------------

It might be handy to dynamically retrieve parameters. For instance, if you
periodically render your last map backup which is located in a timestamped
directory, it is not convenient to edit the config file each time to fit the
new directory name.

Using environment variables, you can easily retrieve a parameter which has
been set by, for instance, your map backup script. In this example, Overviewer
is called from a *bash* script, but it can be done from other shell scripts
and languages.

::

    #!/bin/bash
    
    ## Add these lines to your bash script
    
    # Setting up an environment variable that child processes will inherit.
    # In this example, the map's path is not static and depends on the
    # previously set $timestamp var.
    MYWORLD_DIR=/path/to/map/backup/$timestamp/YourWorld
    export MYWORLD_DIR
    
    # Running the Overviewer
    overviewer.py --config=/path/to/yourConfig.py

.. note::

    The environment variable will only be local to the process and its child
    processes. The Overviewer, when run by the script, will be able to access
    the variable since it becomes a child process.

::

    ## A config file example
    
    # Importing the os python module
    import os
    
    # Retrieving the environment variable set up by the bash script
    worlds["My world"] = os.environ['MYWORLD_DIR']

    renders["normalrender"] = {
        "world": "My world",
        "title": "Normal Render of My World",
    }

    outputdir = "/home/username/mcmap"

Config File Specifications
==========================

The config file is a python file and is parsed with python's execfile() builtin.
This means you can put arbitrary logic in this file. The Overviewer gives the
execution of the file a local dict with a few pre-defined items (everything in
the overviewer_core.rendermodes module).

If the above doesn't make sense, just know that items in the config file take
the form ``key = value``. Two items take a different form:, ``worlds`` and
``renders``, which are described below.

General
-------

``worlds``
    This is pre-defined as an empty dictionary. The config file is expected to
    add at least one item to it.

    Keys are arbitrary strings used to identify the worlds in the ``renders``
    dictionary.

    Values are paths to worlds (directories with a level.dat)

    e.g.::

        worlds['myworld'] = "/path/to/myworld"

    **You must specify at least one world**

    *Reminder*: Always use forward slashes ("/"), even on Windows.

``renders``
    This is also pre-defined as an empty dictionary. The config file is expected
    to add at least one item to it. By default, it is an ordered dictionary; the
    order you add entries to it will determine the default render in the output
    map and the order the buttons appear in the map UI.

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

.. _outputdir:

``outputdir = "<output directory path>"``
    This is the path to the output directory where the rendered tiles will
    be saved.

    e.g.::

        outputdir = "/path/to/output"

    *Reminder*: Always use forward slashes ("/"), even on Windows.

    **Required**

.. _googleapikey:

``google_api_key = "your-google-api-key"``
    This is the key issued by Google to use their Google Maps Javascript
    APIs.

    **Required**

.. _processes:

``processes = num_procs``
    This specifies the number of worker processes to spawn on the local machine
    to do work. It defaults to the number of CPU cores you have, if not
    specified.

    This can also be specified with :option:`--processes <-p>`

    e.g.::

        processes = 2

Observers
~~~~~~~~~

.. _observer:

``observer = <observer object>``
    This lets you configure how the progress of the render is reported. The
    default is to display a progress bar, unless run on Windows or with stderr
    redirected to a file. The default value will probably be fine for most
    people, but advanced users may want to make their own progress reporter (for
    a web service or something like that) or you may want to force a particular
    observer to be used. The observer object is expected to have at least ``start``,
    ``add``, ``update``, and ``finish`` methods.

    If you want to specify an observer manually, try something like:
    ::

        from observer import ProgressBarObserver
        observer = ProgressBarObserver()

    There are currently three observers available: ``LoggingObserver``, 
    ``ProgressBarObserver`` and ``JSObserver``. 

    ``LoggingObserver``
         This gives the normal/older style output and is the default when output
         is redirected to a file or when running on Windows

    ``ProgressBarObserver``
        This is used by default when the output is a terminal. Displays a text based
        progress bar and some statistics.

    ``JSObserver(outputdir[, minrefresh][, messages])``
        This will display render progress on the output map in the bottom right
        corner of the screen. ``JSObserver``.

        * ``outputdir="<output directory path"``
            Path to overviewer output directory. For simplicity, specify this 
            as ``outputdir=outputdir`` and place this line after setting
            ``outputdir = "<output directory path>"``.
            
            **Required**
        
        * ``minrefresh=<seconds>``
            Progress information won't be written to file or requested by your
            web browser more frequently than this interval. 

        * ``messages=dict(totalTiles=<string>, renderCompleted=<string>, renderProgress=<string>)``
            Customises messages displayed in browser. All three messages must be
            defined similar to the following:

            * ``totalTiles="Rendering %d tiles"``
              The ``%d`` format string will be replaced with the total number of
              tiles to be rendered.

            * ``renderCompleted="Render completed in %02d:%02d:%02d"``
              The three format strings  will be replaced with the number of hours.
              minutes and seconds taken to complete this render.

            * ``renderProgress="Rendered %d of %d tiles (%d%% ETA:%s)""``
              The four format strings will be replaced with the number of tiles
              completed, the total number of tiles, the percentage complete, and the ETA.
	

            Format strings are explained here: http://docs.python.org/library/stdtypes.html#string-formatting
            All format strings must be present in your custom messages.

        ::

            from observer import JSObserver
            observer = JSObserver(outputdir, 10)
                
		
    ``MultiplexingObserver(Observer[, Observer[, Observer ...]])``
        This observer will send the progress information to all Observers passed
        to it.
        
        * All Observers passed must implement the full Observer interface.
        
        ::
        
            ## An example that updates both a LoggingObserver and a JSObserver
            # Import the Observers
            from observer import MultiplexingObserver, LoggingObserver, JSObserver
            
            # Construct the LoggingObserver
            loggingObserver = LoggingObserver()
            
            # Construct a basic JSObserver
            jsObserver = JSObserver(outputdir) # This assumes you have set the outputdir previous to this line
            
            # Set the observer to a MultiplexingObserver 
            observer = MultiplexingObserver(loggingObserver, jsObserver)
            
    ``ServerAnnounceObserver(target, pct_interval)``
        This Observer will send its progress and status to a Minecraft server
        via ``target`` with a Minecraft ``say`` command.
        
        * ``target=<file handle to write to>``
            Either a FIFO file or stdin. Progress and status messages will be written to this handle.
            
            **Required**
        
        * ``pct_interval=<update rate, in percent>``
            Progress and status messages will not be written more often than this value.
            E.g., a value of ``1`` will make the ServerAnnounceObserver write to its target
            once for every 1% of progress.
            
            **Required**

    ``RConObserver(target, password[, port][, pct_interval])``
        This Observer will announce render progress with the server's ``say``
        command through RCon.

        * ``target=<address>``
            Address of the target Minecraft server.

            **Required**

        * ``password=<rcon password>``
            The server's rcon password.

            **Required**

        * ``port=<port number>``
            Port on which the Minecraft server listens for incoming RCon connections.

            **Default:** ``25575``

        * ``pct_interval=<update rate, in percent>``
            Percentage interval in which the progress should be announced, the same as
            for ``ServerAnnounceObserver``.

            **Default:** ``10``
            
            

Custom web assets
~~~~~~~~~~~~~~~~~

.. _customwebassets:

``customwebassets = "<path to custom web assets>"``
    This option allows you to speciy a directory containing custom web assets
    to be copied to the output directory. Any files in the custom web assets 
    directory overwrite the default files.

    If you are providing a custom index.html, the following strings will be replaced:

    * ``{title}``
      Will be replaced by 'Minecraft Overviewer'

    * ``{time}``
      Will be replaced by the current date and time when the world is rendered
      e.g. 'Sun, 12 Aug 2012 15:25:40 BST'

    * ``{version}``
      Will be replaced by the version of Overviewer used
      e.g. '0.9.276 (5ff9c50)' 

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

        renders['arender'] = {
                'title': 'This render doesn't explicitly declare a world!',
                }

General
~~~~~~~

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

    .. note::

        For the end, you will most likely want to turn down the strength of
        the shadows, as you'd otherwise end up with a very dark result.
        
        e.g.::
            
            end_lighting = [Base(), EdgeLines(), Lighting(strength=0.5)]
            end_smooth_lighting = [Base(), EdgeLines(), SmoothLighting(strength=0.5)]

    **Default:** ``"overworld"``

Rendering
~~~~~~~~~

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
    This is direction or viewpoint angle with which north will be rendered. This north direction will
    match the established north direction in the game where the sun rises in the
    east and sets in the west.

    Here are the valid north directions:

    * ``"upper-left"``
    * ``"upper-right"``
    * ``"lower-left"``
    * ``"lower-right"``

    **Default:** ``"upper-left"``

.. _option_overlay:

``overlay``
    This specifies which renders that this render will be displayed on top of. 
    It should be a list of other renders.  If this option is confusing, think
    of this option's name as "overlay_on_to".

    If you leave this as an empty list, this overlay will be displayed on top
    of all renders for the same world/dimension as this one.

    As an example, let's assume you have two renders, one called "day" and one 
    called "night".  You want to create a Biome Overlay to be displayed on top
    of the "day" render.  Your config file might look like this:

    ::

        outputdir = "output_dir"


        worlds["exmaple"] = "exmaple"

        renders['day'] = {
            'world': 'exmaple',
            'rendermode': 'smooth_lighting',
            'title': "Daytime Render",
        }
        renders['night'] = {
            'world': 'exmaple',
            'rendermode': 'night',
            'title': "Night Render",
        }

        renders['biomeover'] = {
            'world': 'exmaple',
            'rendermode': [ClearBase(), BiomeOverlay()],
            'title': "Biome Coloring Overlay",
            'overlay': ['day']
        }

    **Default:** ``[]`` (an empty list)

.. _option_texturepath:

``texturepath``
    This is a where a specific texture or resource pack can be found to use
    during this render. It can be a path to either a folder or a zip/jar file
    containing the texture resources. If specifying a folder, this option should
    point to a directory that *contains* the assets/ directory (it should not
    point to the assets directory directly or any one particular texture image).

    Its value should be a string: the path on the filesystem to the resource
    pack.

.. _crop:

``crop``
    You can use this to render one or more small subsets of your map. The format
    of an individual crop zone is (min x, min z, max x, max z); if you wish to
    specify multiple crop zones, you may do so by specifying a list of crop zones,
    i.e. [(min x1, min z1, max x1, max z1), (min x2, min z2, max x2, max z2)]

    The coordinates are block coordinates. The same you get with the debug menu
    in-game and the coordinates shown when you view a map.

    Example that only renders a 1000 by 1000 square of land about the origin::

        renders['myrender'] = {
                'world': 'myworld',
                'title': "Cropped Example",
                'crop': (-500, -500, 500, 500),
        }

    Example that renders two 500 by 500 squares of land::

        renders['myrender'] = {
                'world': 'myworld',
                'title': "Multi cropped Example",
                'crop': [(-500, -500, 0, 0), (0, 0, 500, 500)]
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

Image options
~~~~~~~~~~~~~

``imgformat``
    This is which image format to render the tiles into. Its value should be a
    string containing "png", "jpg", or "jpeg".

    **Default:** ``"png"``

``imgquality``
    This is the image quality used when saving the tiles into the JPEG image
    format. Its value should be an integer between 0 and 100.

    **Default:** ``95``

``optimizeimg``

    .. warning::
        Using image optimizers will increase render times significantly.

    This option specifies which additional tools overviewer should use to
    optimize the filesize of rendered tiles.
    The tools used must be placed somewhere where overviewer can find them, for
    example the "PATH" environment variable or a directory like /usr/bin.

    The option is a list of Optimizer objects, which are then executed in
    the order in which they're specified::
        
        # Import the optimizers we need
        from optimizeimages import pngnq, optipng

        worlds["world"] = "/path/to/world"

        renders["daytime"] = {
            "world":"world",
            "title":"day",
            "rendermode":smooth_lighting,
            "optimizeimg":[pngnq(sampling=1), optipng(olevel=3)],
        }

    .. note::
        Don't forget to import the optimizers you use in your config file, as shown in the
        example above.
    
    Here is a list of supported image optimization programs:

    ``pngnq``
        pngnq quantizes 32-bit RGBA images into 8-bit RGBA palette PNGs. This is
        lossy, but reduces filesize significantly. Available settings:
        
        ``sampling``
            An integer between ``1`` and ``10``, ``1`` samples all pixels, is slow and yields
            the best quality. Higher values sample less of the image, which makes
            the process faster, but less accurate.

            **Default:** ``3``

        ``dither``
            Either the string ``"n"`` for no dithering, or ``"f"`` for Floyd
            Steinberg dithering. Dithering helps eliminate colorbanding, sometimes
            increasing visual quality.

            .. warning::
                With pngnq version 1.0 (which is what Ubuntu 12.04 ships), the
                dithering option is broken. Only the default, no dithering,
                can be specified on those systems.

            **Default:** ``"n"``

        .. warning::
            Because of several PIL bugs, only the most zoomed in level has transparency
            when using pngnq. The other zoom levels have all transparency replaced by
            black. This is *not* pngnq's fault, as pngnq supports multiple levels of
            transparency just fine, it's PIL's fault for not even reading indexed
            PNGs correctly.

    ``optipng``
        optipng tunes the deflate algorithm and removes unneeded channels from the PNG,
        producing a smaller, lossless output image. It was inspired by pngcrush.
        Available settings:

        ``olevel``
            An integer between ``0`` (few optimizations) and ``7`` (many optimizations).
            The default should be satisfactory for everyone, higher levels than the default
            see almost no benefit.

            **Default:** ``2``

    ``pngcrush``
        pngcrush, like optipng, is a lossless PNG recompressor. If you are able to do so, it
        is recommended to use optipng instead, as it generally yields better results in less
        time.
        Available settings:

        ``brute``
            Either ``True`` or ``False``. Cycles through all compression methods, and is very slow.

            .. note::
                There is practically no reason to ever use this. optipng will beat pngcrush, and
                throwing more CPU time at pngcrush most likely won't help. If you think you need
                this option, then you are most likely wrong.

            **Default:** ``False``

    ``jpegoptim``
        jpegoptim can do both lossy and lossless JPEG optimisation. If no options are specified,
        jpegoptim will only do lossless optimisations.
        Available settings:

        ``quality``
            A number between 0 and 100 that corresponds to the jpeg quality level. If the input
            image has a lower quality specified than the output image, jpegoptim will only do
            lossless optimisations.
            
            If this option is specified and the above condition does not apply, jpegoptim will
            do lossy optimisation.

            **Default:** ``None`` *(= Unspecified)*

        ``target_size``
            Either a percentage of the original filesize (e.g. ``"50%"``) or a target filesize
            in kilobytes (e.g. ``15``). jpegoptim will then try to reach this as its target size.

            If specified, jpegoptim will do lossy optimisation.

            .. warning::
                This appears to have a greater performance impact than just setting ``quality``.
                Unless predictable filesizes are a thing you need, you should probably use ``quality``
                instead.

            **Default:** ``None`` *(= Unspecified)*

    **Default:** ``[]``

Zoom
~~~~

These options control the zooming behavior in the JavaScript output.

``defaultzoom``
    This value specifies the default zoom level that the map will be
    opened with. It has to be greater than 0, which corresponds to the
    most zoomed-out level. If you use ``minzoom`` or ``maxzoom``, it
    should be between those two.

    **Default:** ``1``

``maxzoom``
    This specifies the maximum, closest in zoom allowed by the zoom
    control on the web page. This is relative to 0, the farthest-out
    image, so setting this to 8 will allow you to zoom in at most 8
    times. This is *not* relative to ``minzoom``, so setting
    ``minzoom`` will shave off even more levels. If you wish to
    specify how many zoom levels to leave off, instead of how many
    total to use, use a negative number here. For example, setting
    this to -2 will disable the two most zoomed-in levels.

    .. note::

            This does not change the number of zoom levels rendered, but allows
            you to neglect uploading the larger and more detailed zoom levels if bandwidth
            usage is an issue.

    **Default:** Automatically set to most detailed zoom level

``minzoom``
    This specifies the minimum, farthest away zoom allowed by the zoom
    control on the web page. For example, setting this to 2 will
    disable the two most zoomed-out levels.

    .. note::

            This does not change the number of zoom levels rendered, but allows
            you to have control over the number of zoom levels accessible via the
            slider control.

    **Default:** 0 (zero, which does not disable any zoom levels)

Other HTML/JS output options
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``showlocationmarker``
    Allows you to specify whether to show the location marker when accessing a URL
    with coordinates specified.

    **Default:** ``True``

``base``
    Allows you to specify a remote location for the tile folder, useful if you
    rsync your map's images to a remote server. Leave a trailing slash and point
    to the location that contains the tile folders for each render, not the
    tiles folder itself. For example, if the tile images start at
    http://domain.com/map/world_day/ you want to set this to http://domain.com/map/

.. _option_markers:

``markers``
    This controls the display of markers, signs, and other points of interest
    in the output HTML.  It should be a list of dictionaries.  

    .. note::

       Setting this configuration option alone does nothing.  In order to get
       markers and signs on our map, you must also run the genPO script.  See
       the :doc:`Signs and markers<signs>` section for more details and documenation.

    **Default:** ``[]`` (an empty list)


``poititle``
    This controls the display name of the POI/marker dropdown control.

    **Default:** "Signs"

``showspawn``
    This is a boolean, and defaults to ``True``. If set to ``False``, then the spawn
    icon will not be displayed on the rendered map.

``bgcolor``
    This is the background color to be displayed behind the map. Its value
    should be either a string in the standard HTML color syntax or a 4-tuple in
    the format of (r,b,g,a). The alpha entry should be set to 0.

    **Default:** ``#1a1a1a``

Map update behavior
~~~~~~~~~~~~~~~~~~~

.. _rerenderprob:

``rerenderprob``
    This is the probability that a tile will be rerendered even though there may
    have been no changes to any blocks within that tile. Its value should be a
    floating point number between 0.0 and 1.0.

    **Default:** ``0``


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

``renderchecks``
    This is an integer, and functions as a more complex form of
    ``forcerender``. Setting it to 1 enables :option:`--check-tiles`
    mode, setting it to 2 enables :option:`--forcerender`, and 3 tells
    Overviewer to keep this particular render in the output, but
    otherwise don't update it. It defaults to 0, which is the usual
    update checking mode.

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
    
    **Options**
    
    sealevel
        sealevel of the word you're rendering. Note that the default,
        128, is usually *incorrect* for most worlds. You should
        probably set this to 64. Default: 128

Depth
    Only renders blocks between the specified min and max heights.

    **Options**

    min
        lowest level of blocks to render. Default: 0

    max
        highest level of blocks to render. Default: 255

Exposed
    Only renders blocks that are exposed (adjacent to a transparent block).
    
    **Options**
    
    mode
        when set to 1, inverts the render mode, only drawing unexposed blocks. Default: 0
        
NoFluids
    Don't render fluid blocks (water, lava).

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

Hide
    Hide blocks based on blockid. Blocks hidden in this way will be
    treated exactly the same as air.

    **Options**

    blocks
        A list of block ids, or (blockid, data) tuples to hide.

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

SpawnOverlay
    Color the map red in areas where monsters can spawn. Either use
    this on top of other modes, or on top of ClearBase to create a
    pure overlay.

    **Options**

    overlay_color
        custom color for the overlay in the format (r,g,b,a). If not
        defined a red color is used.

SlimeOverlay
    Color the map green in chunks where slimes can spawn. Either use
    this on top of other modes, or on top of ClearBase to create a
    pure overlay.

    **Options**

    overlay_color
        custom color for the overlay in the format (r,g,b,a). If not
        defined a green color is used.

MineralOverlay
    Color the map according to what minerals can be found
    underneath. Either use this on top of other modes, or on top of
    ClearBase to create a pure overlay.

    **Options**

    minerals
        A list of (blockid, (r, g, b)) tuples to use as colors. If not
        provided, a default list of common minerals is used.

        Example::

            MineralOverlay(minerals=[(64,(255,255,0)), (13,(127,0,127))])

StructureOverlay
    Color the map according to patterns of blocks. With this rail overlays
    or overlays for other small structures can be realized. It can also be
    a MineralOverlay with alpha support.

    This Overlay colors according to a patterns that are specified as
    multiple tuples of the form ``(relx, rely, relz, blockid)``. So
    by specifying ``(0, -1, 0, 4)`` the block below the current one has to
    be a cobblestone.

    One color is then specified as
    ``((relblockid1, relblockid2, ...), (r, g, b, a))`` where the
    ``relblockid*`` are relative coordinates and the blockid as specified
    above. The ``relblockid*`` must match all at the same time for the
    color to apply.

    Example::

        StructureOverlay(structures=[(((0, 0, 0, 66), (0, -1, 0, 4)), (255, 0, 0, 255)),
                                     (((0, 0, 0, 27), (0, -1, 0, 4)), (0, 255, 0, 255))])

    In this example all rails(66) on top of cobblestone are rendered in
    pure red. And all powerrails(27) are rendered in green.

    If ``structures`` is not provided, a default rail coloring is used.

BiomeOverlay
    Color the map according to the biome at that point. Either use on
    top of other modes or on top of ClearBase to create a pure overlay.

    **Options**

    biomes
        A list of ("biome name", (r, g, b)) tuples to use as colors. Any
        biome not specified won't be highlighted. If not provided then 
        a default list of biomes and colors is used.

        Example::

            BiomeOverlay(biomes=[("Forest", (0, 255, 0)), ("Desert", (255, 0, 0))])

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

    renders["survivalday"] = {
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
