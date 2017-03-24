==========================
Frequently Asked Questions
==========================

.. contents::
    :local:

General Questions
=================

Does the Overviewer work with mod blocks?
-----------------------------------------

The Overviewer will render the world, but none of the blocks added by mods
will be visible. Currently, the blocks Overviewer supports are hardcoded, and
because there is no official Minecraft modding API as of the time of writing,
supporting mod blocks is not trivial.

Can I view Overviewer maps without having an internet connection?
-----------------------------------------------------------------

Not at the moment. The Overviewer relies on the Google maps API to display
maps, which your browser needs to load from Google. However, switching away
from Google Maps is something that will most likely be looked into in the
future.

When my map expands, I see remnants of another zoom level
---------------------------------------------------------

When your map expands ("Your map seems to have expanded beyond its previous
bounds") you may see tiles at a zoom level that shouldn't be there, usually
around the borders. This is probably not a bug, but is typically caused by
copying the map tiles from their render destination to another location (such as
a web server).

When you're copying the rendered map, you need to be sure files that *don't*
exist in the source are *deleted* in the destination.

Explanation: When Overviewer re-arranges tiles to make room for another zoom
level, it moves some tiles tiles at a particular zoom level and places them at a
higher zoom level. The tiles that used to be at that zoom level should no longer
exist there, but if you're copying tiles, there is no mechanism to *delete*
those files at the copy destination.

If that explanation doesn't make full sense, then just know that you must do one
of the following:

* Render the tiles directly to the destination

* Copy the tiles from the render destination in a way that deletes extra files,
  such as using ``rsync`` with ``--delete``

* Erase and re-copy the files at the final destination when the map expands.
  Map expansions double the width and height of the map, so you will eventually
  hit a map size that is unlikely to need another level.

You've added a new feature or changed textures, but it's not showing up on my map!
----------------------------------------------------------------------------------

Some new features will only show up in newly-rendered areas. Use the
:option:`--forcerender` option to update the entire map. If you have a really
large map and don't want to re-render everything, take a look at
the :ref:`rerenderprob<rerenderprob>` configuration option.

The background color of the map is black, and I don't like it!
--------------------------------------------------------------

You can change the background color by specifying a new one in the configuration
file. See the :doc:`config` page for more details.

I downloaded the Windows version but when I double-click it, the window closes real fast.
-----------------------------------------------------------------------------------------

The Overviewer is a command line program and must be run from a command line. It
is necessary to become at least a little familiar with a command line to run The
Overviewer (if you have no interest in this, perhaps this isn't the mapping
program for you). A brief guide is provided on the
:doc:`win_tut/windowsguide` page.

Unfortunately, A full tutorial of the Windows command line is out of scope for this
documentation; consult the almighty Google for tutorials and information on
the Windows command line. (If you would like to contribute a short tutorial to
these docs, please do!)

Batch files are another easy way to run the Overviewer without messing with
command lines, but information on how to do this has also not been written. 

On a related note, we also welcome contributions for a graphical interface for
the Overviewer.

The Overviewer is eating up all my memory!
------------------------------------------

We have written The Overviewer with memory efficiency in mind. On even the
largest worlds we have at our disposal to test with, it should not be taking
more than a gigabyte or two. It varies of course, that number is only an
estimate, but most computers with a reasonable amount of RAM should run just
fine.

If you are seeing exorbitant memory usage, then it is likely either a bug or a
subtly corrupted world. Please file an issue or come talk to us on IRC so we can
take a look! See :ref:`help`.

How can I log The Overviewer's output to a file?
------------------------------------------------

If you are on a UNIX-like system like MacOSX or Linux, you can use shell redirection
to write the output into a file::

    overviewer.py --config=myconfig.py > renderlog.log 2>&1

What this does is redirect the previous commands standard output to the file "renderlog.log",
and redirect the standard error to the standard output. The file will be overwritten each time
you run this command line; to simply append the output to the file, use two greater than signs::

    overviewer.py --config=myconfig.py >> renderlog.log 2>&1


.. _cropping_faq:

I've deleted some sections of my world but they still appear in the map
-----------------------------------------------------------------------
Okay, so making edits to your world in e.g. worldedit has some caveats,
especially regarding deleting sections of your world.

This faq also applies to using the :ref:`crop<crop>` option.

Under normal operation with vanilla Minecraft and no external tools fiddling
with the world, Overviewer performs correctly, rendering areas that have
changed, and everything is good.

Often with servers one user will travel reeeeally far out and cause a lot of
extra work for the server and for The Overviewer, so you may be tempted to
delete parts of your map. This can cause problems, so read on to learn what you
can do about it.

First some explanation: Until recently (Mid May 2012) The Overviewer did not
have any facility for detecting parts of the map that should no longer exist.
Remember that the map is split into small tiles. When Overviewer starts up, the
first thing it does is calculate which tiles should exist and which should be
updated. This means it does not check or even look at tiles that should not
exist. This means that parts of your world which have been deleted will hang
around on your map because Overviewer won't even look at those tiles and notice
they shouldn't be there. You may even see strange artifacts around the border as
tiles that should exist get updated.

Now, with the :option:`--check-tiles` option, The Overviewer *will* look for and
remove tiles that should no longer exist. So you can render your map once with
that option and all those extra tiles will get removed automatically. However,
this is only half of the solution. The other half is making sure the tiles along
the border are re-rendered, or else it will look like your map is being cut off.

Explanation: The tiles next to the ones that were removed are tiles that should
continue to exist, but parts of them have chunks that no longer exist. Those
tiles then should be re-rendered to show that. However, since tile updates are
triggered by the chunk last-modified timestamp changing, and the chunks that
still exist have *not* been updated, those tiles will not get re-rendered.

The consequence of this is that your map will end up looking cut-off around the
new borders that were created by the parts you deleted. You can fix this one of
two ways.

1. You can run a render with :option:`--forcerender`. This has the unfortunate
   side-effect of re-rendering *everything* and doing much more work than is
   necessary.

2. Manually navigate the tile directory hierarchy and manually delete tiles
   along the edge. Then run once again with :option:`--check-tiles` to re-render
   the tiles you just deleted. This may not be as bad as it seems. Remember each
   zoom level divides the world into 4 quadrants: 0, 1, 2, and 3 are the upper
   left, upper right, lower left, and lower right. It shouldn't be too hard to
   navigate it manually to find the parts of the map that need re-generating.

3. The third non-option is to not worry about it. The problem will fix itself if
   people explore near there, because that will force that part of the map to
   update.

My map is zoomed out so far that it looks (almost) blank
--------------------------------------------------------

We see this quite a bit, and seems to stem from a bug in the Minecraft terrain
generation.

Explanation: Minecraft generates chunks of your world as it needs them. When
Overviewer goes to render your map, it looks at how big the world is, and
calculates how big the maps needs to be in order to fit it all in.
Occasionally, we see that Minecraft has generated a few chunks of the world
extremely far away from the main part of the world. These erroneous chunks have
most likely not been explored [*]_ and should not exist.

There are two solutions. The preferred is to delete the offending chunks. Open
up your region folder of your world and look at the region file names. They are
numbered ``r.##.##.mcr`` where ``##`` is a number. The two numbers indicate the
coordinates of that region file. Look for region files with coordinates much
larger in magnitude than any others. Most likely you will find around 1–3
region files with coordinates much larger than any others. Delete or otherwise
remove those files, and re-render your map.

The other option is to use the :ref:`crop<crop>` option to tell Overviewer not
to render all of your map, but instead to only render the specified region.

As always, if you need assistance, come chat with us on :ref:`irc<help>`.

.. [*] They could also have been triggered by an accidential teleport where the coordinates were typed in manually.

I want to put manual POI definitions or other parts of my config into a seperate file
-------------------------------------------------------------------------------------

This can be achieved by creating a module and then importing it in
your config.  First, create a file containing your markers
definitions. We'll call it ``manualmarkers.py``.

::

    mymarkers = [{'id':'town', 'x':200, 'y':64, 'z':-400, 'name':'Pillowcastle'},
                 {'id':'town', 'x':500, 'y':70, 'z': 100, 'name':'brownotopia' }]


The final step is to import the very basic module you've just created
into your config.  In your config, do the following

::

    import sys
    sys.path.append("/wherever/your/manualmarkers/is/") # Replace this with your path to manualmarkers.py,
                                                        # so python can find it
    
    from manualmarkers import *                         # import our markers
    
    # all the usual config stuff goes here
    
    renders["myrender"] = {
        "title" : "foo",
        "world" : "someworld",
        "manualpois" : mymarkers,                         # IMPORTANT! Variable name from manualmarkers.py
        # and here goes the list of the filters, etc.
    }

Now you should be all set.
