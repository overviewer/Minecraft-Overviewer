==========================
Frequently Asked Questions
==========================

.. contents::
    :local:

General Questions
=================

The full map doesn't display even when fully zoomed out!
--------------------------------------------------------

Are you using the :option:`-z` or :option:`--zoom <-z>` option on your
commandline or in settings.py? If so, try removing it, or increasing the value
you set.  It's quite likely you don't need it at all. See the documentation for
the :option:`zoom <-z>` option.

You've added a few feature, but it's not showing up on my map!
--------------------------------------------------------------

Some new features will only show up in newly-rendered areas. Use the
:option:`--forcerender` option to update the entire map.

How do I use this on CentOS 5?
------------------------------

CentOS 5 comes with Python 2.4, but the Overviewer needs 2.6 or higher. See the
special instructions at :ref:`centos`

The background color of the map is black, and I don't like it!
--------------------------------------------------------------

You can change this by using the :option:`--bg-color` command line option, or
``bg_color`` in settings.py. See the :doc:`options` page for more details.

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
