====================================
Minecraft Overviewer  |Build Status|
====================================
By Andrew Brown and contributors (see CONTRIBUTORS.rst).

Documentation:
    http://docs.overviewer.org

Github code repository:
    https://github.com/overviewer/Minecraft-Overviewer

Travis-CI:
    https://travis-ci.org/overviewer/Minecraft-Overviewer

Blog:
    https://overviewer.org/blog/


The Minecraft Overviewer is a command-line tool for rendering high-resolution
maps of Minecraft worlds. It generates a set of static html and image files and
uses the Google Maps API to display a nice interactive map.

The Overviewer has been in active development for several years and has many
features, including day and night lighting, cave rendering, mineral overlays,
and many plugins for even more features! It is written mostly in Python with
critical sections in C as an extension module.

Getting Started
---------------
All documentation has been consolidated at our documentation site. For
information on downloading, compiling, installing, and running The Overviewer,
visit the docs site.

http://docs.overviewer.org

A few helpful tips are below, but everyone is going to want to visit the
documentation site for the most up-to-date and complete set of instructions!

Alternatively, the docs are also in the docs/ directory of the source download.
Look in there if you can't access the docs site.

Examples
--------
See examples of The Overviewer in action!

https://github.com/overviewer/Minecraft-Overviewer/wiki/Map-examples

Disclaimers
-----------
Before you dive into using this, just be aware that, for large maps, there is a
*lot* of data to parse through and process. If your world is very large, expect
the initial render to take at least an hour, possibly more. (Since Minecraft
maps are practically infinite, the maximum time this could take is also
infinite!)

If you press ctrl-C, it will stop. The next run will pick up where it left off.

Once your initial render is done, subsequent renderings will be MUCH faster due
to all the caching that happens behind the scenes. Just use the same output
directory and it will only update the tiles it needs to.

There are probably some other minor glitches along the way, hopefully they will
be fixed soon. See the `Bugs`_ section below.

Viewing the Results
-------------------
Within the output directory you will find two things: an index.html file, and a
directory hierarchy full of images. To view your world, simply open index.html
in a web browser. Internet access is required to load the Google Maps API
files, but you otherwise don't need anything else.

You can throw these files up to a web server to let others view your map. To
ensure the map works when viewed through the Internet, you'll need a Google
Maps API key, which you can `get for free from Google`_. Please note that you
are also bound by the `Google Maps API Terms of Service`_.

.. _get for free from Google: https://developers.google.com/maps/documentation/javascript/get-api-key
.. _Google Maps API Terms of Service: https://developers.google.com/maps/terms

Bugs
====

For a current list of issues, visit
https://github.com/overviewer/Minecraft-Overviewer/issues

Feel free to comment on issues, report new issues, and vote on issues that are
important to you.

.. |Build Status| image:: https://secure.travis-ci.org/overviewer/Minecraft-Overviewer.svg?branch=master
