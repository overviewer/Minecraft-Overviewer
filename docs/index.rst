========================
The Minecraft Overviewer
========================

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

Download
========

The Overviewer works with Linux, Mac, and Windows! We provide Windows and Debian
built executables for your convenience. Find them as well as the full sources on
our `Github Homepage`_.

**If you are running Windows, Debian, or Ubuntu and would like the pre-built
packages and don't want to have to compile anything yourself**, head to the
:doc:`installing` page.

**If you would like to build the Overviewer from source yourself (it's not that
bad)**, head to the :doc:`Building <building>` page.

.. _Github Homepage: https://github.com/overviewer/Minecraft-Overviewer

Help
====
**IF YOU NEED HELP COMPILING OR RUNNING THE OVERVIEWER** feel free to pop in
IRC: #overviewer on freenode. Not familiar with IRC? `Use the web client
<http://webchat.freenode.net/?channels=overviewer>`_. There's usually someone on
there that can help you out.

If you think you've found a bug or other issue, file an issue on our `Issue
Tracker <https://github.com/overviewer/Minecraft-Overviewer/issues>`_. Filing or
commenting on an issue sends a notice to our IRC channel, so the response time
is often very good!

Documentation Contents
======================

.. toctree::
   :maxdepth: 2

   building
   installing
   running
   options
   faq
   design/designdoc


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

