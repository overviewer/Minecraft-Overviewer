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

The full map doesn't display even when fully zoomed out!
--------------------------------------------------------

Are you using the :option:`-z` or :option:`--zoom <-z>` option on your
commandline or in settings.py? If so, try removing it, or increasing the value
you set.  It's quite likely you don't need it at all. See the documentation for
the :option:`zoom <-z>` option.

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
program for you).

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
take a look. See :ref:`help`.

We have had a few reports of The Overviewer eating all a system's RAM but we
have been unable to figure out why or duplicate the issue. Any help or evidence
you can provide us will help us figure this out!
