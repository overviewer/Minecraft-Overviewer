.. _signsmarkers:

=================
Signs and Markers
=================

The Overviewer can display signs, markers, and other points of interest on your
map.  This works a little differently than it has in the past, so be sure to read
these docs carefully.

In these docs, we use the term POI (or point of interest) to refer to entities and
tileentities.


Configuration File
==================


Filter Functions
----------------

A filter function is a python function that is used to figure out if a given POI
should be part of a markerSet of not, and to control how it is displayed.  
The function should accept one argument (a dictionary, also know as an associative
array), and return a string representing the text to be displayed.  For example::

    def signFilter(poi):
        if poi['id'] == 'Sign' or poi['id'] == 'minecraft:sign':
            return "\n".join([poi['Text1'], poi['Text2'], poi['Text3'], poi['Text4']])

If a POI doesn't match, the filter can return None (which is the default if a python 
functions runs off the end without an explicit 'return').

The single argument will either a TileEntity, or an Entity taken directly from 
the chunk file.  It could also be a special entity representing a player's location
or a player's spawn.  See below for more details.

In this example, this function returns all 4 lines from the sign
if the entity is a sign.
For more information of TileEntities and Entities, see
the `Chunk Format <http://www.minecraftwiki.net/wiki/Chunk_format>`_ page on
the Minecraft Wiki.

A more complicated filter function can construct a more customized display text::

    def chestFilter(poi):
        if poi['id'] == "Chest":
            return "Chest with %d items" % len(poi['Items'])

It is also possible to return a tuple from the filter function to specify a hovertext
different from the text displayed in the info window. The first entry of the tuple will
be used as the hover text, the second will be used as the info window content::

    def chestFilter(poi):
        if poi['id'] == "Chest":
            return ("Chest", "Chest with %d items" % len(poi['Items']))

Because of the way the config file is loaded, if you need to import a function or module
for use in your filter function, you need to explicitly load it into the global namespace::

    global escape
    from cgi import escape
    def signFilter(poi):
        if poi['id'] == 'Sign':
            return "\n".join(map(escape, [poi['Text1'], poi['Text2'], poi['Text3'], poi['Text4']]))

Since writing these filters can be a little tedious, a set of predefined filters
functions are provided.  See the :ref:`predefined_filter_functions` section for
details.


Special POIs
------------

There are currently two special types of POIs.  They each have a special id:

PlayerSpawn
  Used to indicate the spawn location of a player.  The player's name is set
  in the ``EntityId`` key, and the location is in the x,y,z keys

Player
  Used to indicate the last known location of a player.  The player's name is set
  in the ``EntityId`` key, and the location is in the x,y,z keys.

.. note::
  The player location is taken from level.dat (in the case of a single-player world) 
  or the player.dat files (in the case of a multi-player server).  The locations are 
  only written to these files when the world is saved, so this won't give you real-time
  player location information. 

Here's an example that displays icons for each player::

    def playerIcons(poi):
        if poi['id'] == 'Player':
            poi['icon'] = "https://overviewer.org/avatar/%s" % poi['EntityId']
            return "Last known location for %s" % poi['EntityId']

Note how each POI can get a different icon by setting ``poi['icon']``. These icons must exist in either
the output folder, or in your custom web assets folder. If the icon file does not exist in the correct 
location, your markers will be shown without an icon - making them invisible!

Manual POIs
-----------

It is also possible to manually define markers. Each render can have a render dictionary key
called ``manualpois``, which is a list of dicts. Each dict represents a marker, and is required
to have at least the attributes ``x``, ``y``, ``z`` and ``id``, with the coordinates being Minecraft
world coordinates. (i.e. what you see in-game when you press F3)

An example which adds two POIs with the id "town", and then uses a filter function to filter for them::

    def townFilter(poi):
        if poi['id'] == 'Town':
            return poi['name']

            
    renders['myrender'] = {
        'world':'myworld',
        'title':'Example',
        'manualpois':[
                       {'id':'Town',
                        'x':200,
                        'y':64,
                        'z':200,
                        'name':'Foo'},
                       {'id':'Town',
                        'x':-300,
                        'y':85,
                        'z':-234,
                        'name':'Bar'}],
        'markers': [dict(name="Towns", filterFunction=townFilter)],
    }

Here is a more complex example where not every marker of a certain id has a certain key::

    def townFilter(poi):
        if poi['id'] == 'Town':
            try:
                return (poi['name'], poi['description'])
            except KeyError:
                return poi['name'] + '\n'

            
    renders['myrender'] = {
        'world':'myworld',
        'title':'Example',
        'manualpois':[
                       {'id':'Town',
                        'x':200,
                        'y':64,
                        'z':200,
                        'name':'Foo',
                        'description':'Best place to eat hamburgers'},
                       {'id':'Town',
                        'x':-300,
                        'y':85,
                        'z':-234,
                        'name':'Bar'}],
        'markers': [dict(name="Towns", filterFunction=townFilter, icon="icons/marker_town.png")],
        ### Note: The 'icon' parameter allows you to specify a custom icon, as per
        ###       standard markers. This icon must exist in the same folder as your
        ###       custom webassets or in the same folder as the generated index.html
        ###       in this case, we use the marker_town.png icon which comes with
        ###       the Overviewer by default, located in a subdirectory of web_assets.
    }
    

Render Dictionary Key
---------------------

Each render can specify a list of zero or more filter functions.  Each of these
filter functions become a selectable item in the 'Signs' drop-down menu in the
rendered map.  Previously, this used to be a list of functions.  Now it is a list
of dictionaries.  For example::

    renders['myrender'] = {
            'world': 'myworld',
            'title': "Example",
            'markers': [dict(name="All signs", filterFunction=signFilter),
                        dict(name="Chests", filterFunction=chestFilter, icon="chest.png", createInfoWindow=False)]
    }


The following keys are accepted in the marker dictionary:

``name``
    This is the text that is displayed in the 'Signs' dropdown.

``filterFunction``
    This is the filter function.  It must accept at least 1 argument (the POI to filter),
    and it must return either None or a string.

``icon``
    Optional.  Specifies the icon to use for POIs in this group.  If omitted, it defaults
    to a signpost icon.  Note that each POI can have different icon by setting the key 'icon'
    on the POI itself (this can be done by modifying the POI in the filter function.  See the
    example above)

``createInfoWindow``
    Optional. Specifies whether or not the icon displays an info window on click. Defaults to True

``checked``
    Optional.  Specifies whether or not this marker group will be checked(visible) by default when
    the map loads.  Defaults to False

Generating the POI Markers
==========================

.. note::
    Markers will not be updated or added during a regular overviewer.py map render!
    You must use one of the following options to generate your markers.

The --genpoi option
-------------------
Running overviewer.py with the :option:`--genpoi` option flag will generate your 
POI markers. For example::

     /path/to/overviewer.py --config /path/to/your/config/file.conf --genpoi

.. note::
    A --genpoi run will NOT generate a map render, it will only generate markers.

If all went well, you will see a "Markers" button in the upper-right corner of
your map.

genPOI.py
---------

The genPOI.py script is also provided, and can be used directly. For example:: 
    
    /path/to/overviewer/genpoi.py --config=/path/to/your/config.file



This will generate the necessary JavaScript files needed in your config file's
outputdir.

Options
-------

genPOI comes with a few options of its own.

.. cmdoption:: -c <file>, --config=<file>

    The config file to use for the genPOI operation. This must be the same
    config file that you use for your normal rendering runs.

.. cmdoption:: -q, --quiet

    Outputs less information onto the terminal while running.

.. cmdoption:: --skip-scan

    Skip scanning the world for entities and tile entities. Useful if you only
    want to generate markers for players or through manual POIs, as you can
    speed up the genPOI operation considerably.

.. cmdoption:: --skip-players

    Skip reading and retrieving player data during genPOI runs. This is useful
    if you don't plan on generating markers for the player locations.

.. _predefined_filter_functions:

Predefined Filter Functions
===========================

TODO write some filter functions, then document them here

Marker Icons Overviewer ships by default
========================================

Overviewer comes with multiple small icons that you can use for your markers.
You can find them in the ``overviewer_core/data/web_assets/icons`` directory.

If you want to make your own in the same style, you can use the provided
``marker_base_plain.svg`` and ``marker_base_plain_red.svg`` as template, with
a vector editing software such as `Inkscape <http://inkscape.org>`_.
