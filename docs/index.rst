..
    Hey! You! Read these docs at http://docs.overviewer.org
    Go there now!

========================
The Minecraft Overviewer
========================

See also the `Github Homepage`_

.. _Github Homepage: https://github.com/overviewer/Minecraft-Overviewer

Introduction
============
The Minecraft Overviewer is a command-line tool for rendering high-resolution
maps of Minecraft worlds. It generates a set of static html and image files and
uses the Google Maps API to display a nice interactive map.

The Overviewer has been in active development for over a year and has many
features, including day and night lighting, cave rendering, mineral overlays,
and many plugins for even more features! It is written mostly in Python with
critical sections in C as an extension module.

For a simple example of what your renders will look like, head over to `The
"Exmaple" Map <http://overviewer.org/example/>`_. For more user-contributed
examples, see `The Example Wiki Page <https://github.com/overviewer/Minecraft-Overviewer/wiki/Map-examples>`_.

Features
========

* Renders large resolution images of your world, such that you can zoom in and
  see details

* Customizable textures! Pulls textures straight from your installed texture
  pack!

* Outputs a Google Map powered interface that is memory efficient, both in
  generating and viewing.

* Renders efficiently in parallel, using as many simultaneous processes as you
  want!

* Utilizes caching to speed up subsequent renderings of your world.

* Throw the output directory up on a web server to share your Minecraft world
  with everyone!

Requirements
============
This is a quick list of what's required to run The Overviewer. It runs on
Windows, Mac, and Linux as long as you have these software packages installed:

* Python 2.6 or 2.7 (we are *not* yet compatible with Python 3.x)

* PIL (Python Imaging Library)

* Numpy

* Either a Minecraft Client installed or a terrain.png for the textures.

The first three are included in the Windows download. Also, there are additional
requirements for compiling it (like a compiler). More details are available in
either the :doc:`Building <building>` or :doc:`Installing <installing>` pages.

Getting Started
===============

The Overviewer works with Linux, Mac, and Windows! We provide Windows and Debian
built executables for your convenience. Find them as well as the full sources on
our `Github Homepage`_.

**If you are running Windows, Debian, or Ubuntu and would like the pre-built
packages and don't want to have to compile anything yourself**, head to the
:doc:`installing` page.

**If you would like to build the Overviewer from source yourself (it's not that
bad)**, head to the :doc:`Building <building>` page.

**For all other platforms** you will need to build it yourself.
:doc:`building`.


Help
====

**IF YOU NEED HELP COMPILING OR RUNNING THE OVERVIEWER** feel free to chat with
us live in IRC: #overviewer on Freenode. There's usually someone on there that
can help you out. Not familiar with IRC? `Use the web client
<http://webchat.freenode.net/?channels=overviewer>`_. (If there's no immediate
response, wait around or try a different time of day; we have to sleep sometime)

Also check our :doc:`Frequently Asked Questions <faq>` page.

If you think you've found a bug or other issue, file an issue on our `Issue
Tracker <https://github.com/overviewer/Minecraft-Overviewer/issues>`_. Filing or
commenting on an issue sends a notice to our IRC channel, so the response time
is often very good!

Documentation Contents
======================

.. toctree::
   :maxdepth: 2

   installing
   building
   running
   options
   faq
   design/designdoc


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

