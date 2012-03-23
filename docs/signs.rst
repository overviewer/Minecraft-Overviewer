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
should be part of a markerSet of not.  The function should accept one argument
(a dictionary, also know as an associative array), and return a boolean::

    def signFilter(poi):
        "All signs"
        return poi['id'] == 'Sign'

The single argument will either a TileEntity, or an Entity taken directly from 
the chunk file.  In this example, this function returns true only if the type
of entity is a sign.  For more information of TileEntities and Entities, see
the `Chunk Format <http://www.minecraftwiki.net/wiki/Chunk_format>`_ page on
the Minecraft Wiki.

.. note::
   The doc string ("All signs" in this example) is important.  It is the label
   that appears in your rendered map

A more advanced filter may also look at other entity fields, such as the sign text::

    def goldFilter(poi):
        "Gold"
        return poi['id'] == 'Sign' and (\
            'gold' in poi['Text1'] or
            'gold' in poi['Text2'])
           
This looks for the word 'gold' in either the first or second line of the signtext.

Since writing these filters can be a little tedious, a set of predefined filters
functions are provided.  See the :ref:`predefined_filter_functions` section for
details.

Render Dictionary Key
---------------------

Each render can specify a list of zero or more filter functions.  Each of these
filter functions become a selectable item in the 'Signs' drop-down menu in the
rendered map.  For example::

    renders['myrender'] = {
            'world': 'myworld',
            'title': "Example",
            'markers': [allFilter, anotherFilter],
    }




Generating the POI Markers
==========================

genPOI.py
---------

In order to actually generate the markers and add them to your map, the script 
genPOI.py must be run. For example::

    genpoi.py --config=/path/to/your/config.file

.. note::
    Markers will not be updated or added during a regular overviewer.py 
    map render!

This will generate the necessary JavaScript files needed in your config file's
outputdir.

Options
-------

genPOI.py has a single option:: --config. You should use the same configfile as 
used for your normal renders.


.. _predefined_filter_functions:

Predefined Filter Functions
===========================

TODO write some filter functions, then document them here

