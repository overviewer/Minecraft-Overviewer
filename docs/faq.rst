==========================
Frequently Asked Questions
==========================

.. contents::
    :local:

General Questions
=================

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

You've added a few feature or changed textures, but it's not showing up on my map!
----------------------------------------------------------------------------------

Some new features will only show up in newly-rendered areas. Use the
:option:`--forcerender` option to update the entire map. If you have a really
large map and don't want to re-render everything, take a look at
the :option:`--stochastic-render` option.

How do I use this on CentOS 5?
------------------------------

CentOS 5 comes with Python 2.4, but the Overviewer needs 2.6 or higher. See the
special instructions at :ref:`centos`

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
